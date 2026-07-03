// #include "mlir/IR/Builders.h"                           // OpBuilder, create ops
// #include "mlir/IR/BuiltinOps.h"                         // func::FuncOp
// #include "mlir/Pass/Pass.h"                             // PassWrapper, OperationPass
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h" // applyPatternsAndFoldGreedily
// #include "mlir/Dialect/Linalg/IR/Linalg.h"              // linalg conv ops
// #include "mlir/Dialect/Tensor/IR/Tensor.h"              // tensor::EmptyOp, transpose
// #include "mlir/IR/PatternMatch.h"                       // PatternRewriter, OpRewritePattern
// #include "mlir/IR/BuiltinTypes.h"                       // RankedTensorType
// #include "mlir/IR/Value.h"                              // Value (SSA representation)
// #include "mlir/IR/AffineMap.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Linalg/Transforms/Transforms.h"
// #include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
// #include "mlir/Dialect/Vector/IR/VectorOps.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "RVV/RVVDialect.h"
// #include "RVV/Transforms.h"

// // #include "mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp"

// /* -----------------------------------------------------------------------------------------------------
// PASS DESCRIPTION:


// --------------------------------------------------------------------------------------------------------*/

// namespace LasseThesis
// {
//     namespace transforms
//     {
//         using namespace mlir;

//         namespace
//         {

//             struct TransferReadToRVVPattern : public OpRewritePattern<func::funcOp>
//             {
//                 using OpRewritePattern<mlir::vector::>::OpRewritePattern; // constructor forwarding


//                 LogicalResult matchAndRewrite(
//                     mlir::vector::TransferReadOp loadOp /*This is the OP we are matching on*/,
//                     PatternRewriter &rewriter) const override
//                 {
//                     Location loc = loadOp.getLoc();
//                     auto vectorType = loadOp.getVectorType();
                    

//                     rewriter.create<rvv::VLEOp>(
//                         loc, vecType, loadOp.getSource(), vsetvl
//                     )
//                 {

                 

//                     rewriter.replaceOp(convOp, hLoop->getResult(0));
//                     convOp->setAttr("vectorized", rewriter.getBoolAttr(false));

//                     return success();
//                 }
//             };
//         } // Anonymous namespace

//         // Pass construction must be public (outside anonymous namespace)
//         struct ConvertVectorToRVVPass
//             : public PassWrapper<Pass, OperationPass<func::FuncOp>>
//         {
//             void runOnOperation() override
//             {
//                 RewritePatternSet patterns(&getContext());   
//                 populateVectorToRVVPatterns(patterns);

//                 // Apply the pattern greedily.
//                 if (failed(applyPatternsGreedily(
//                         func, std::move(patterns))))
//                     signalPassFailure();
//             }
//         };

//         // Everyting implementation related goes to the anonymous namespace, while the factory function stays public.
//         /**Register the pass for passmanager. */
//         std::unique_ptr<Pass> createConvertVectorToRVV()
//         {
//             return std::make_unique<ConvertVectorToRVVPass>();
//         }
//     }
//     } // transforms
// } // LasseThesis
