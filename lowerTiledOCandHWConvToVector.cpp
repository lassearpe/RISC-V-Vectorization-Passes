#include "mlir/IR/Builders.h"                           // OpBuilder, create ops
#include "mlir/IR/BuiltinOps.h"                         // func::FuncOp
#include "mlir/Pass/Pass.h"                             // PassWrapper, OperationPass
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily
#include "mlir/Dialect/Linalg/IR/Linalg.h"              // linalg conv ops
#include "mlir/Dialect/Tensor/IR/Tensor.h"              // tensor::EmptyOp, transpose
#include "mlir/IR/PatternMatch.h"                       // PatternRewriter, OpRewritePattern
#include "mlir/IR/BuiltinTypes.h"                       // RankedTensorType
#include "mlir/IR/Value.h"                              // Value (SSA representation)
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

/**-----------------------------------------------------------------------------------------------------

// PASS DESCRIPTION:

// [N,H,W,C]

// [KH,KW,IC,OC]

// Implements:
// N
//     0...OC_TILE step
//         0...H_TILE step
//             0...W_TILE step
//             Load accumulator acc[0,h,0,oc]
//                 KH
//                     KW
//                         Load weight vector
//                         h in H_TILE
//                             IC
//                                 Load input
//                                 Broadcast input
//                                 FMA
//             Store Accumulator

// Using the khArgs,kwArgs,icArgs -> iterArgs. -- keep vectors in registers through the entire KH x KW x IC reduction.

// kernelVec loaded once and used H_tile times per IC iteration.

// Outer loops (OC,H,W): Tensor (because 2d tiling)
// inner loops (KH,KW,IC):Vector Register level.
// transferwrite up shall turn the finished vectors into a tensor update.

// Uses a tail loop for remainder handling. Beware: Atm only handles a remainder of 1.

// TAIL REMAINDER LOOP LOOKS LIKE A "SLOW" SOLUTION!
*/

namespace LasseThesis
{
    namespace transforms
    {

        using namespace mlir;

        namespace
        {

            struct VectoizedOCandHWTiledConv : public OpRewritePattern<mlir::linalg::Conv2DNhwcHwcfOp>
            {
                using OpRewritePattern<mlir::linalg::Conv2DNhwcHwcfOp>::OpRewritePattern; // constructor forwarding

                LogicalResult matchAndRewrite(
                    mlir::linalg::Conv2DNhwcHwcfOp convOp /*This is the OP we are matching on*/,
                    PatternRewriter &rewriter) const override
                {

                    if (!convOp->getAttr("tiled"))
                    {
                        return failure();
                    }

                    if (convOp->getAttr("vectorized"))
                    {
                        return failure();
                    }

                    /****************************
                     * Get necessary information to build the pass
                     ***************************/

                    Location loc = convOp.getLoc();

                    auto input = convOp.getDpsInputOperand(0)->get();
                    auto kernel = convOp.getDpsInputOperand(1)->get();
                    auto output = convOp.getDpsInitOperand(0)->get();

                    DenseIntElementsAttr strides = convOp.getStrides();
                    auto extractHStride = *(strides.begin());
                    auto extractWStride = *(strides.begin() + 1);
                    int64_t shInt = extractHStride.getZExtValue();
                    int64_t swInt = extractWStride.getZExtValue();

                    auto inputType = cast<RankedTensorType>(input.getType());
                    auto kernelType = cast<RankedTensorType>(kernel.getType());
                    auto outputType = cast<RankedTensorType>(output.getType());

                    int64_t H = outputType.getDimSize(1);
                    int64_t W = outputType.getDimSize(2);
                    int64_t OC = outputType.getDimSize(3);
                    int64_t IC = inputType.getDimSize(3);
                    int64_t kH = kernelType.getDimSize(0);
                    int64_t kW = kernelType.getDimSize(1);
                    // int64_t F = kernelType.getDimSize(3); // Same as totalIC

                    /***********************************************************
                     * Creating the microkernel directly via the vector dialect.
                     *************************************************************/

                    // Get the tile size from the former pass.
                    int64_t ocTileSize = convOp->getAttrOfType<IntegerAttr>("oc_tile_size").getInt();
                    if (!ocTileSize)
                        return failure();

                    int64_t hTileSize = convOp->getAttrOfType<IntegerAttr>("h_tile_size").getInt();
                    if (!hTileSize)
                        return failure();

                    int64_t wTileSize = convOp->getAttrOfType<IntegerAttr>("w_tile_size").getInt();
                    if (!wTileSize)
                        return failure();

                    // Value hUB = rewriter.create<arith::ConstantIndexOp>(loc, H);
                    Value hTileSizeValue = rewriter.create<arith::ConstantIndexOp>(loc, hTileSize);
                    Value wTileSizeValue = rewriter.create<arith::ConstantIndexOp>(loc, wTileSize);

                    // Saleable dims == False, as we are getting our dims as a CLI argument.
                    VectorType vectorType = VectorType::get({ocTileSize}, rewriter.getF32Type(), /*scaleable dims*/ {false});

                    // We need to project onto the accumulator tile with an Affine mapping.
                    AffineMap projectionMap = AffineMap::get(/*dim count*/ 4, 0, {rewriter.getAffineDimExpr(3)}, convOp.getContext());

                    Value OCTile = rewriter.create<arith::ConstantIndexOp>(loc, ocTileSize);
                    Value hTile = rewriter.create<arith::ConstantIndexOp>(loc, hTileSize);
                    Value wTile = rewriter.create<arith::ConstantIndexOp>(loc, wTileSize);
                    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0); // reuseable 0 index constant
                    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1); // reuseable 1 index constant
                    Value ocUB = rewriter.create<arith::ConstantIndexOp>(loc, OC);
                    Value icUB = rewriter.create<arith::ConstantIndexOp>(loc, IC);
                    Value hUB = rewriter.create<arith::ConstantIndexOp>(loc, H);
                    Value wUB = rewriter.create<arith::ConstantIndexOp>(loc, W);
                    Value khUB = rewriter.create<arith::ConstantIndexOp>(loc, kH);
                    Value kwUB = rewriter.create<arith::ConstantIndexOp>(loc, kW);
                    Value cst0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
                    Value sh = rewriter.create<arith::ConstantIndexOp>(loc, shInt);
                    Value sw = rewriter.create<arith::ConstantIndexOp>(loc, swInt);

                    // CALCULATE YOUR UB FOR THE "MAIN" LOOP, AND THE REMAINDER HEREOF
                    Value HRem = rewriter.create<arith::RemSIOp>(loc, hUB, hTileSizeValue);
                    Value hMainUB = rewriter.create<arith::SubIOp>(loc, hUB, HRem);

                    Value WRem = rewriter.create<arith::RemSIOp>(loc, wUB, wTileSizeValue);
                    Value wMainUB = rewriter.create<arith::SubIOp>(loc, wUB, WRem);

                    int64_t hRemInt = H % hTileSize;
                    int64_t hMainUBInt = H - hRemInt;

                    int64_t wRemInt = W % wTileSize;
                    int64_t wMainUBInt = W - wRemInt;

                    /********************
                     * Build the loop nest.
                     * SCF for-loops uses a lambda function internally to nested inner loops.
                     ********************/

                    auto OCLoop = rewriter.create<scf::ForOp>(
                        loc, c0, ocUB, OCTile,
                        ValueRange(output),
                        [&](OpBuilder &b, Location loc, Value oc, ValueRange ocArgs)
                        {
                            // Value OCOut = ocArgs[0];

                            auto hLoopMain = b.create<scf::ForOp>(
                                loc, c0, hMainUB, hTile,
                                ocArgs,
                                [&](OpBuilder &b, Location loc, Value h, ValueRange hArgs)
                                {
                                    // Value hOut = hArgs[0];

                                    auto wLoopMain = b.create<scf::ForOp>(
                                        loc, c0, wMainUB, wTile,
                                        hArgs,
                                        [&](OpBuilder &b, Location loc, Value w, ValueRange wArgs)
                                        {
                                            Value currentWOut = wArgs[0];

                                            // Initialize accumulators for this specific w position
                                            SmallVector<Value> accs;

                                            int accumulatorsNeeded = hTileSize * wTileSize;

                                            // Unroll accumulators
                                            for (int64_t i = 0; i < accumulatorsNeeded; i++)
                                            {
                                                Value acc = b.create<arith::ConstantOp>(
                                                    loc, vectorType, DenseElementsAttr::get(vectorType, 0.0f));
                                                accs.push_back(acc);
                                            }

                                            auto kHLoop = b.create<scf::ForOp>(
                                                loc, c0, khUB, c1,
                                                ValueRange{accs},
                                                [&](OpBuilder &b, Location loc, Value kh, ValueRange kHArgs)
                                                {
                                                    // Value kHAcc = khArgs[0];

                                                    auto kWLoop = b.create<scf::ForOp>(
                                                        loc, c0, kwUB, c1,
                                                        kHArgs,
                                                        [&](OpBuilder &b, Location loc, Value kw, ValueRange kWArgs)
                                                        {
                                                            // Value kWAcc = kWArgs[0];

                                                            // Value hOffset = b.create<arith::AddIOp>(
                                                            //     loc,
                                                            //     h,
                                                            //     kh);

                                                            // Value wOffset = b.create<arith::AddIOp>(
                                                            //     loc,
                                                            //     w,
                                                            //     kw);

                                                            auto icLoop = b.create<scf::ForOp>(
                                                                loc, c0, icUB, c1,
                                                                kWArgs,
                                                                [&](OpBuilder &b, Location loc, Value ic, ValueRange icArgs)
                                                                {
                                                                    Value icAcc = icArgs[0];

                                                                    // We load 1 weight for each kh,kw,ic position. Therefore, we
                                                                    // load an amount of weights corresponding to our OC-tile.
                                                                    // Because the current OC-tile always needs this fixed amount of weights.

                                                                    Value kernelVec = b.create<vector::TransferReadOp>(
                                                                        loc,
                                                                        vectorType,
                                                                        kernel,
                                                                        ValueRange{kh, kw, ic, oc},
                                                                        std::optional<Value>(cst0),
                                                                        AffineMapAttr::get(projectionMap),
                                                                        // mask,
                                                                        b.getBoolArrayAttr({true}));

                                                                    // Now, we reuse these weights over the HW-tiles.

                                                                    // FETCH NEW INPUT FOR EACH H POS AND PERFORM FMA FOR THIS CHANNEL TILE.

                                                                    SmallVector<Value> nextAccs(accumulatorsNeeded);

                                                                    // Value hIndexStrided = b.create<arith::MulIOp>(loc, b.create<arith::ConstantIndexOp>(loc, h_t), sh);

                                                                    for (int64_t h_t = 0; h_t < hTileSize; h_t++)
                                                                    {
                                                                        for (int64_t w_t = 0; w_t < wTileSize; w_t++)
                                                                        {
                                                                            int64_t idx = h_t * wTileSize + w_t;

                                                                            Value hIndexTile = b.create<arith::AddIOp>(loc, b.create<arith::ConstantIndexOp>(loc, h_t), h);
                                                                            Value wIndexTile = b.create<arith::AddIOp>(loc, b.create<arith::ConstantIndexOp>(loc, w_t), w);

                                                                            Value hIndexStrided = b.create<arith::MulIOp>(loc, hIndexTile, sh);
                                                                            Value wIndexStrided = b.create<arith::MulIOp>(loc, wIndexTile, sw);

                                                                            Value hOffset = b.create<arith::AddIOp>(loc, hIndexStrided, kh);
                                                                            Value wOffset = b.create<arith::AddIOp>(loc, wIndexStrided, kw);

                                                                            Value scalarIn = b.create<tensor::ExtractOp>(
                                                                                loc, b.getF32Type(), input,
                                                                                ValueRange{c0, hOffset, wOffset, ic}); // offsets inside the tiles given by h_t and w_t.

                                                                            Value broadcastIn = b.create<vector::BroadcastOp>(loc, vectorType, scalarIn);
                                                                            nextAccs[idx] = b.create<vector::FMAOp>(loc, vectorType, kernelVec, broadcastIn, icArgs[idx]);
                                                                        }
                                                                    }

                                                                    b.create<scf::YieldOp>(loc, nextAccs);
                                                                });

                                                            b.create<scf::YieldOp>(loc, icLoop.getResults());
                                                        });
                                                    b.create<scf::YieldOp>(loc, kWLoop.getResults());
                                                });

                                            // Value finalAcc = kHLoop.getResult(0);

                                            auto finishedOutput = currentWOut;

                                            /************************
                                             * Storing the results. *
                                             ************************/

                                            for (int h_t = 0; h_t < hTileSize; h_t++)
                                            {
                                                for (int w_t = 0; w_t < wTileSize; w_t++)
                                                {
                                                    int64_t idx = h_t * wTileSize + w_t;

                                                    Value hIndex = b.create<arith::AddIOp>(loc, h, b.create<arith::ConstantIndexOp>(loc, h_t));
                                                    Value wIndex = b.create<arith::AddIOp>(loc, w, b.create<arith::ConstantIndexOp>(loc, w_t));

                                                    auto newWOutput = b.create<vector::TransferWriteOp>(
                                                        loc,
                                                        kHLoop.getResult(idx), // Value
                                                        finishedOutput,        // Dest
                                                        ValueRange{c0, hIndex, wIndex, oc},
                                                        AffineMapAttr::get(projectionMap),
                                                        // mask,
                                                        b.getBoolArrayAttr({true}));

                                                    finishedOutput = newWOutput.getResult();
                                                }
                                            }

                                            b.create<scf::YieldOp>(loc, ValueRange{finishedOutput});
                                        });

                                    // Track the tensor state.
                                    auto currentTensorW = wLoopMain.getResult(0);
                                    Value finalTensor = currentTensorW;

                                    /***********************************
                                     ******* REMAINDER HANDLING*********
                                     ***********************************
                                     * SCALAR HANDLING - NOT TILED.
                                     * TRIED TILING, BUT SCALAR-VERSION HAD BETTER PERFORMANCE IN THE REMAINDER HANDLER.
                                     *
                                     * BEWARE: This is only relevant for the 7x7 spatial dim layers.
                                     * NOTE: It is implemented, so it can work for any of the layers -- but the implementation mitigates some performance gains!
                                     *
                                     * I believe this tail handling might be suboptimal performance wise. If we comment it out and compile, the layers get a significant performance boost (albeit only partial results).
                                     * This version only very slightly better than the non-spatially tiled version.
                                     * So there are performance to be gained, if one does this better. Maybe with padding and register masking, so it is unnecessary to generate the extra loop.
                                     * Unsure of the best way to improve this, left as potential work-to-do.
                                     *
                                     *
                                     ***********************************/

                                    /**************************
                                     * The "bottom" (h) remainder.
                                     *************************/
                                    if (hRemInt != 0)
                                    {
                                        // Loop over W again
                                        // But only to the "main" upper bound of the width.
                                        llvm::errs() << "\nH remainder activated\n";
                                        auto wLoopTail = b.create<scf::ForOp>(
                                            loc, c0, wUB, c1,
                                            ValueRange{finalTensor},
                                            [&](OpBuilder &b, Location loc, Value w, ValueRange wArgs)
                                            {
                                                // Value CurrentWOutTail = wArgs[0];

                                                // int accumulatorsNeededHRem = hTileSize;

                                                // SmallVector<Value> accs;

                                                // Accumulators needed for handling the tile of the Height-dimension.
                                                // for (int64_t i = 0; i < accumulatorsNeededHRem; i++)
                                                // {
                                                Value acc = b.create<arith::ConstantOp>(
                                                    loc, vectorType, DenseElementsAttr::get(vectorType, 0.0f));
                                                //     accs.push_back(acc);
                                                // }
                                                // Fresh accumulator for this (hTail, w) position
                                                // Value accW = acc;

                                                // KH loop
                                                auto khLoop = b.create<scf::ForOp>(
                                                    loc, c0, khUB, c1,
                                                    ValueRange{acc},
                                                    [&](OpBuilder &b, Location loc, Value kh, ValueRange khArgs)
                                                    {
                                                        // Value accKH = khArgs[0];

                                                        // Value accKW = kwArgs[0];
                                                        Value hOffset = b.create<arith::AddIOp>(loc, hMainUB, kh);
                                                        Value hIndexStrided = b.create<arith::MulIOp>(loc, hOffset, sh);

                                                        // KW loop
                                                        auto kwLoop = b.create<scf::ForOp>(
                                                            loc, c0, kwUB, c1,
                                                            khArgs,
                                                            [&](OpBuilder &b, Location loc, Value kw, ValueRange kwArgs)
                                                            {
                                                                /***************************************************************************
                                                                 * Should have an own internal loop to unroll, if the remainder is larger than 1!
                                                                 * However, currently, only a remainder of 1 in both directions is supported; relevant for 7x7 layers.
                                                                 ***************************************************************************/

                                                                // (wMain+kw)*stride
                                                                Value wOffset = b.create<arith::AddIOp>(loc, wMainUB, kw); // needs to start from the main, where we "left off".
                                                                Value wIndexStrided = b.create<arith::MulIOp>(loc, wOffset, sw);
                                                                auto icLoop = b.create<scf::ForOp>(
                                                                    loc, c0, icUB, c1,
                                                                    kwArgs,
                                                                    [&](OpBuilder &b, Location loc, Value ic, ValueRange icArgs)
                                                                    {
                                                                        Value kernelVec = b.create<vector::TransferReadOp>(
                                                                            loc, vectorType, kernel,
                                                                            ValueRange{kh, kw, ic, c0},
                                                                            /*paddingValue=*/b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getFloatAttr(b.getF32Type(), 0.0)),
                                                                            AffineMapAttr::get(projectionMap),
                                                                            b.getBoolArrayAttr({true}));

                                                                        Value scalarIn = b.create<tensor::ExtractOp>(
                                                                            loc, b.getF32Type(), input,
                                                                            ValueRange{c0, hIndexStrided, wIndexStrided, ic});

                                                                        // Broadcast to vector
                                                                        Value broadcastIn =
                                                                            b.create<vector::BroadcastOp>(loc, vectorType, scalarIn);

                                                                        // FMA
                                                                        acc = b.create<vector::FMAOp>(loc, vectorType, kernelVec, broadcastIn, icArgs[0]);
                                                                        b.create<scf::YieldOp>(loc, acc);
                                                                    });
                                                                b.create<scf::YieldOp>(loc, icLoop.getResults());
                                                            });
                                                        b.create<scf::YieldOp>(loc, kwLoop.getResult(0));
                                                    });

                                                // Value accFinal = khLoop.getResult(0);

                                                /******************************************************
                                                 * Supports remainders of arbitrary size.
                                                 ******************************************************/

                                                auto intermediateWOutputTail = wArgs[0];

                                                for (int i = 0; i < wTileSize; i++)
                                                {
                                                    Value wIndex = b.create<arith::AddIOp>(
                                                        loc, w, b.create<arith::ConstantIndexOp>(loc, i));

                                                    auto newWOutputTail = b.create<vector::TransferWriteOp>(
                                                        loc,
                                                        khLoop.getResult(0),
                                                        intermediateWOutputTail,
                                                        ValueRange{c0, hMainUB, wIndex, oc},
                                                        AffineMapAttr::get(projectionMap),
                                                        b.getBoolArrayAttr({true}));
                                                    intermediateWOutputTail = newWOutputTail.getResult();
                                                }

                                                b.create<scf::YieldOp>(loc, ValueRange{intermediateWOutputTail});
                                            });

                                        finalTensor = wLoopTail.getResult(0);
                                    }

                                    b.create<scf::YieldOp>(loc, finalTensor);
                                });

                            auto currentTensorH = hLoopMain.getResult(0);
                            Value finalTensor = currentTensorH;

                            /********************************
                             * The "W" scalar remainder handling
                             *
                             * NOTE: THIS VERSION SUPPORTS LARGER THAN 1 REMAINDER HANDLING.
                             *******************************/
                            if (wRemInt != 0)
                            {
                                llvm::errs() << "\nW remainder activated\n";

                                auto hLoopTail = b.create<scf::ForOp>(
                                    loc, c0, hUB, c1,
                                    ValueRange{finalTensor},
                                    [&](OpBuilder &b, Location loc, Value h, ValueRange hArgs)
                                    {
                                        auto wLoopTail = b.create<scf::ForOp>(
                                            loc, wMainUB, wUB, c1,
                                            hArgs,
                                            [&](OpBuilder &b, Location loc, Value w, ValueRange wArgs)
                                            {
                                                // Value acc = b.create<arith::ConstantOp>(
                                                //     loc, vectorType, DenseElementsAttr::get(vectorType, 0.0f));

                                                SmallVector<Value> accs;

                                                for (int64_t i = 0; i < wRemInt; i++)
                                                {
                                                    Value acc = b.create<arith::ConstantOp>(
                                                        loc, vectorType, DenseElementsAttr::get(vectorType, 0.0f));
                                                    accs.push_back(acc);
                                                }

                                                auto khLoop = b.create<scf::ForOp>(
                                                    loc, c0, khUB, c1,
                                                    ValueRange{accs},
                                                    [&](OpBuilder &b, Location loc, Value kh, ValueRange khArgs)
                                                    {
                                                        // (h+kh)*stride
                                                        Value hIndexStrided = b.create<arith::MulIOp>(loc, h, sh);
                                                        Value hOffset = b.create<arith::AddIOp>(loc, hIndexStrided, kh);

                                                        auto kwLoop = b.create<scf::ForOp>(
                                                            loc, c0, kwUB, c1,
                                                            khArgs,
                                                            [&](OpBuilder &b, Location loc, Value kw, ValueRange kwArgs)
                                                            {
                                                                auto icLoop = b.create<scf::ForOp>(
                                                                    loc, c0, icUB, c1,
                                                                    kwArgs,
                                                                    [&](OpBuilder &b, Location loc, Value ic, ValueRange icArgs)
                                                                    {
                                                                        Value kernelVec = b.create<vector::TransferReadOp>(
                                                                            loc, vectorType, kernel,
                                                                            ValueRange{kh, kw, ic, c0},
                                                                            /*paddingValue=*/b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getFloatAttr(b.getF32Type(), 0.0)),
                                                                            AffineMapAttr::get(projectionMap),
                                                                            b.getBoolArrayAttr({true}));

                                                                        for (int64_t w_t = 0; w_t < wRemInt; w_t++)
                                                                        {

                                                                            // Value hIndexTile = b.create<arith::AddIOp>(loc, b.create<arith::ConstantIndexOp>(loc, h_t), h);
                                                                            Value wIndexTile = b.create<arith::AddIOp>(loc, b.create<arith::ConstantIndexOp>(loc, w_t), w);
                                                                            Value wIndexStrided = b.create<arith::MulIOp>(loc, wIndexTile, sw);
                                                                            // (wMain+kw)*stride
                                                                            Value wOffset = b.create<arith::AddIOp>(loc, wIndexStrided, kw); // needs to start from the main, where we "left off".

                                                                            // Fetch scalar based on offset
                                                                            Value scalarIn = b.create<tensor::ExtractOp>(
                                                                                loc, b.getF32Type(), input,
                                                                                ValueRange{c0, hOffset, wOffset, ic});

                                                                            // Broadcast to vector
                                                                            Value broadcastIn =
                                                                                b.create<vector::BroadcastOp>(loc, vectorType, scalarIn);

                                                                            // FMA
                                                                            accs[w_t] = b.create<vector::FMAOp>(loc, vectorType, kernelVec, broadcastIn, icArgs[0]);
                                                                            // }
                                                                        }
                                                                        b.create<scf::YieldOp>(loc, accs);
                                                                    });

                                                                b.create<scf::YieldOp>(loc, icLoop.getResults());
                                                            });

                                                        // Value accAfterKW = kwLoop.getResult(0);
                                                        b.create<scf::YieldOp>(loc, kwLoop.getResults());
                                                    });
                                                auto intermediateHOutputTail = wArgs[0];

                                                auto newOutputTail = b.create<vector::TransferWriteOp>(
                                                    loc,
                                                    khLoop.getResult(0),
                                                    intermediateHOutputTail,
                                                    ValueRange{c0, h, w, oc},
                                                    AffineMapAttr::get(projectionMap),
                                                    b.getBoolArrayAttr({true}));

                                                intermediateHOutputTail = newOutputTail.getResult();
                                                // b.create<scf::YieldOp>(loc, khLoop.getResults());

                                                b.create<scf::YieldOp>(loc, ValueRange{intermediateHOutputTail});
                                            });
                                        b.create<scf::YieldOp>(loc, wLoopTail.getResult(0));
                                    });

                                finalTensor = hLoopTail.getResult(0);
                            }

                            /** THE BELOW IS UNNESSECARY, AS THE REMAINDER IS NOT TILED. */

                            // Value currentTensorHW = hLoopMain.getResult(0);

                            //         /**************************************************
                            //          * Handling of the final remaninder (of the remainder)
                            //          *
                            //          * In my current implementation, this is always 1x1 (as we only handle remainders of size 1 in 7x7 layers).
                            //          * .
                            //          **************************************************/
                            //         if (hRemInt != 0 && wRemInt != 0)
                            //         {

                            //             // Hardcoded for the 1x1 corner.
                            //             Value hCorner = rewriter.create<arith::ConstantIndexOp>(loc, hMainUBInt);
                            //             Value wCorner = rewriter.create<arith::ConstantIndexOp>(loc, wMainUBInt);

                            //             Value acc = rewriter.create<arith::ConstantOp>(
                            //                 loc, vectorType, DenseElementsAttr::get(vectorType, 0.0f));

                            //             auto khLoop = rewriter.create<scf::ForOp>(
                            //                 loc, c0, khUB, c1,
                            //                 ValueRange{acc},
                            //                 [&](OpBuilder &b, Location loc, Value kh, ValueRange khArgs)
                            //                 {
                            //                     Value accKH = khArgs[0];

                            //                     // KW loop
                            //                     auto kwLoop = b.create<scf::ForOp>(
                            //                         loc, c0, kwUB, c1,
                            //                         ValueRange{accKH},
                            //                         [&](OpBuilder &b, Location loc, Value kw, ValueRange kwArgs)
                            //                         {
                            //                             Value accKW = kwArgs[0];

                            //                             // Value hOffset = b.create<arith::AddIOp>(loc, h, kh);

                            //                             /***************************************************************************
                            //                              * BEWARE:
                            //                              * Should have an internal loop to unroll, if the remainder is larger than 1!
                            //                              ***************************************************************************/
                            //                             auto icLoop = b.create<scf::ForOp>(
                            //                                 loc, c0, icUB, c1,
                            //                                 kwArgs,
                            //                                 [&](OpBuilder &b, Location loc, Value ic, ValueRange icArgs)
                            //                                 {
                            //                                     Value accIC = icArgs[0];

                            //                                     Value kernelVec = b.create<vector::TransferReadOp>(
                            //                                         loc,
                            //                                         vectorType,
                            //                                         kernel,
                            //                                         ValueRange{kh, kw, ic, c0},
                            //                                         /*paddingValue=*/b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getFloatAttr(b.getF32Type(), 0.0)),
                            //                                         AffineMapAttr::get(projectionMap),
                            //                                         b.getBoolArrayAttr({true}));

                            //                                     Value scalarIn = b.create<tensor::ExtractOp>(
                            //                                         loc, b.getF32Type(), input,
                            //                                         ValueRange{c0, hCorner, wCorner, ic});

                            //                                     // Broadcast to vector
                            //                                     Value broadcastIn =
                            //                                         b.create<vector::BroadcastOp>(loc, vectorType, scalarIn);

                            //                                     // FMA
                            //                                     Value FMA = b.create<vector::FMAOp>(loc, vectorType, kernelVec, broadcastIn, icArgs[0]);

                            //                                     b.create<scf::YieldOp>(loc, FMA);
                            //                                 });

                            //                             // Value accAfterIC = ;
                            //                             b.create<scf::YieldOp>(loc, icLoop.getResult(0));
                            //                         });

                            //                     b.create<scf::YieldOp>(loc, kwLoop.getResult(0));
                            //                 });

                            //             // Value accFinal = ;

                            //             auto finalOut = b.create<vector::TransferWriteOp>(
                            //                 loc,
                            //                 khLoop.getResult(0),                  // Values (Yielded from nested loops.)
                            //                 finalTensor,                          // Destination
                            //                 ValueRange{c0, hCorner, wCorner, oc}, // Indices to store into.
                            //                 AffineMapAttr::get(projectionMap),
                            //                 b.getBoolArrayAttr({true}));

                            //             finalTensor = finalOut.getResult();
                            //         }
                            b.create<scf::YieldOp>(loc, ValueRange{finalTensor});
                        });

                    convOp->setAttr("vectorized", rewriter.getBoolAttr(true));
                    rewriter.replaceOp(convOp, OCLoop.getResults());

                    return success();
                }
            };
        }

        // Pass construction must be public (outside anonymous namespace)
        struct VectorizeOCandHWTiledConvPass
            : public PassWrapper<VectorizeOCandHWTiledConvPass, OperationPass<func::FuncOp>>
        {
            void runOnOperation() override
            {

                func::FuncOp funcOp = getOperation(); // Inside a func.func region.

                RewritePatternSet patterns(funcOp.getContext());              // Create pattern set for our convolution.
                patterns.add<VectoizedOCandHWTiledConv>(funcOp.getContext()); // Provide the pattern we defines above.

                // Apply the pattern greedily.
                if (failed(applyPatternsGreedily(
                        funcOp, std::move(patterns))))
                    signalPassFailure();
            }
        };

        // Everyting implementation related goes to the anonymous namespace, while the factory function stays public.
        /**Register the pass for passmanager. */
        std::unique_ptr<Pass> createLowerOCandHWTiledConvToVectorPass()
        {
            return std::make_unique<VectorizeOCandHWTiledConvPass>();
        }

    } // transforms
} // LasseThesis