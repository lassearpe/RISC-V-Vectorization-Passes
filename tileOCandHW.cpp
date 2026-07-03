#include "mlir/IR/Builders.h"                           // OpBuilder, create ops
#include "mlir/IR/BuiltinOps.h"                         // func::FuncOp
#include "mlir/Pass/Pass.h"                             // PassWrapper, OperationPass
#include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily
#include "mlir/Dialect/Linalg/IR/Linalg.h"              // linalg conv ops
#include "mlir/Dialect/Tensor/IR/Tensor.h"              // tensor::EmptyOp, transpose
#include "mlir/IR/PatternMatch.h"                       // PatternRewriter, OpRewritePattern
#include "mlir/IR/BuiltinTypes.h"                       // RankedTensorType
#include "mlir/IR/Value.h"                              // Value (SSA representation)
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"

/* -----------------------------------------------------------------------------------------------------
PASS DESCRIPTION:

After having changed data layout, we need to promote the accumulator
to outside the inner most loop in order to facilitate data reuse.

[N,H,W,C]

[KH,KW,IC,OC]

Implements:
N
    0...OC_TILE step
        0...H_TILE step
            Load accumulator acc[0,h,0,oc]
            W...TILE step
                KH
                    KW
                        Load weight vector
                        h in H_TILE
                            w in W_TILE
                            IC
                                Load input
                                Broadcast input
                                FMA
            Store Accumulator

*******VIGTIGT***********
RET OVENSTÅENDE TIL, HVAD DEN RENT FAKTISK ÆNDRER TIL
DU FANDT UD A, AT DU SKAL FORHOLDE DIG TIL DEN AKTUELLE STRUKTUR, E.G

N
H
W
OC
KH
KW
IC

også efterfølgende ændrer dette til det korrekte loop nest.


ONLY TILES OC IN PRINCIPLE, BUT HERE THE USER ALSO GIVES THE TILESIZE FOR THE SPATIAL TILING.





--------------------------------------------------------------------------------------------------------*/

std::vector<int> availableFactors(int64_t spatialDim, int64_t maxSpatialTile)
{

    // int64_t limit = std::min<int64_t>(
    //     std::sqrt(spatialDim),
    //     maxSpatialTile);
    std::vector<int> factors;

    for (int i = 1; (i <= maxSpatialTile); i++)
    {

        if (spatialDim % i == 0)
        {
            factors.push_back(i);

            // if (i != spatialDim / i)
            // {
            //     factors.push_back(spatialDim / i);
            // }
        }
    }
    sort(factors.begin(), factors.end());
    return factors;
};

namespace LasseThesis
{
    namespace transforms
    {

        using namespace mlir;

        namespace
        {

            struct OCandHTile : public OpRewritePattern<mlir::linalg::Conv2DNhwcHwcfOp>
            {
                using OpRewritePattern<mlir::linalg::Conv2DNhwcHwcfOp>::OpRewritePattern; // constructor forwarding

                LogicalResult matchAndRewrite(
                    mlir::linalg::Conv2DNhwcHwcfOp convOp /*This is the OP we are matching on*/,
                    PatternRewriter &rewriter) const override
                {

                    if (convOp->getAttr("tiled"))
                    {
                        return failure();
                    }

                    /********************************
                     * Get necessary output information
                     ********************************/

                    Location loc = convOp.getLoc();

                    Value output = convOp.getDpsInitOperand(0)->get(); // full output tensor
                    auto outputType = cast<RankedTensorType>(output.getType());
                    if (!outputType.hasStaticShape())
                        return failure();

                    int64_t H = outputType.getDimSize(1);
                    int64_t W = outputType.getDimSize(2);
                    int64_t totalOC = outputType.getDimSize(3);

                    /******************************************
                     ********** TILING "HEURISTIC "************
                     ******************************************/

                    // Make vector length a placeholder parameter.
                    // BEWARE -- THIS DEPENDS ON THE VECTOR LENGTH!
                    // Always argue to how the heuistic will behave, if the input is not a multiplier of the VL?
                    // int64_t defaultVL = 16;
                    int64_t VLEN = 512;
                    int64_t SEW = 32;
                    int64_t defaultVL = VLEN/SEW;

                    /*****************************
                     * "Maximum OC tile" heuristic
                     *  Important assumption: Always a multiple of the VL
                     ****************************/

                    int64_t maxLMUL = 8;
                    int64_t ocTileSize = totalOC;

                    while (!((ocTileSize / defaultVL) <= maxLMUL))
                    {
                        ocTileSize = ocTileSize / 2;
                    }

                    /*******************************************************************
                     * "Maximum HW tile" heuristic
                     * - Important: This is implemented on the vectorization pass, not here.
                     * I just thought it made good sense to have it decided here, together with the OC tiling.
                     ********************************************************************/

                    int64_t findSpatialFactor = H; // Assuming sqaure HW.



                    // int64_t registerPressure = ocTileSize; // Register pressure depends on the VL/OC tile. 
                    int64_t spatialTile = 1;

                    // Maintain register pressure inequality.
                    while ((ocTileSize) * (spatialTile + 2) <= (512-ocTileSize))  
                    {
                        spatialTile++;
                    }

                    llvm::errs() << spatialTile << " (spatial tile) \n";

                    // Generate spatial factors. 
                    std::vector<int> factors = availableFactors(H, spatialTile);

                    llvm::errs() << "factors:\n";
                    for (int x : factors)
                    {
                        llvm::errs() << x << " ";
                    }

                    // Choose maximum tile; always in one direction for simplicity. (W proved better than H).
                    int64_t largestFactor = factors.at(factors.size()-1); // largest factor

                    int64_t hTileSize = factors.begin()[0]; // 1 
                    int64_t wTileSize = largestFactor;

                    // Edge case: If 7x7 layer.
                    if (spatialTile == 2 && factors.size() == 1) {
                        wTileSize++;
                    }

                    llvm::errs() << "\nOC TILE SIZE:" << ocTileSize;
                    llvm::errs() << "\nH TILE SIZE:" << hTileSize;
                    llvm::errs() << "\nW TILE SIZE:" << wTileSize;

                    // Propagate it to the next pass. 
                    convOp->setAttr("oc_tile_size", IntegerAttr::get(rewriter.getI64Type(), ocTileSize));
                    convOp->setAttr("h_tile_size", IntegerAttr::get(rewriter.getI64Type(), 1));
                    convOp->setAttr("w_tile_size", IntegerAttr::get(rewriter.getI64Type(), 1));

                    /*******************************************************************
                     * Create the tiled convolution, and replace the old conv with this.
                     * (The spatial tile is set in the next pass.)
                     *******************************************************************/

                    SmallVector<int64_t> tileSizes = {0, 0, 0, ocTileSize, 0, 0, 0};
                    linalg::LinalgTilingOptions tileOptions;
                    tileOptions.setTileSizes(tileSizes);

                    FailureOr<linalg::TiledLinalgOp> tiledConvOr = linalg::tileLinalgOp(rewriter, convOp, tileOptions);
                    if (failed(tiledConvOr))
                        return failure();

                    linalg::LinalgOp tiledConv = cast<linalg::LinalgOp>(tiledConvOr->op);

                    /** Any downsteam comp. depending on the original convOp will now depend on the new tiledConvOp. */
                    // The "front" is the OUTERMOST loop created by the tiling process.
                    // (back() is then the innermost.)
                    rewriter.replaceOp(convOp, tiledConvOr->loops.front()->getResults());

                    // tiledConvOr->op->setAttr("tiled-and-promoted", rewriter.getBoolAttr(true));
                    tiledConv->setAttr("tiled", rewriter.getBoolAttr(true));

                    return success();
                }
            };
        } // Anonymous namespace

        // Pass construction must be public (outside anonymous namespace)
        struct OCandHWTilePass
            : public PassWrapper<OCandHWTilePass, OperationPass<func::FuncOp>>
        {
            void runOnOperation() override
            {

                func::FuncOp funcOp = getOperation();            // Inside a func.func region.
                RewritePatternSet patterns(funcOp.getContext()); // Create pattern set for our convolution.
                patterns.add<OCandHTile>(funcOp.getContext());   // Provide the pattern we defines above.

                // Apply the pattern greedily.
                if (failed(applyPatternsGreedily(
                        funcOp, std::move(patterns))))
                    signalPassFailure();
            }
        };

        // Everyting implementation related goes to the anonymous namespace, while the factory function stays public.
        /**Register the pass for passmanager. */
        std::unique_ptr<Pass> createOCandHWTilePass()
        {
            return std::make_unique<OCandHWTilePass>();
        }

    } // transforms
} // LasseThesis
