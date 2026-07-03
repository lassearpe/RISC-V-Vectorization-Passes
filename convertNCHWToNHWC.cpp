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
#include "LasseThesis/Transforms/convertToNHWC.h"
#include "stablehlo/dialect/StablehloOps.h"

/* -----------------------------------------------------------------------------------------------------
PASS DESCRIPTION: Defines a pass that transforms the logical layout of the convolution from NCHW to NWCH.
Works in conjunction with a next pass reordering the linalg.generic.

Important: The final transpose allows MLIR to optimize....

--------------------------------------------------------------------------------------------------------*/

namespace LasseThesis
{
    namespace transforms
    {

        using namespace mlir;

        namespace
        {

            /** Define our own class that represent the op-level rewrite pattern
             * This is used for smaller, focused rewrites (e.g only specific ops).*/

            struct ConvertConv2DNchw : public OpRewritePattern<mlir::stablehlo::ConvolutionOp>
            {
                using OpRewritePattern<mlir::stablehlo::ConvolutionOp>::OpRewritePattern; // constructor forwarding

                /** Perform the matching and rewriting. Procedure:
                 * 1 Access input, outputs
                 * 2 Insert (in this case) transposes op
                 * 3 Create new NHWC conv op
                 * 4 Replace the old Op with the new Op (NHWC).
                 */
                LogicalResult matchAndRewrite(
                    mlir::stablehlo::ConvolutionOp convOp /*This is the OP we are matching on*/,
                    PatternRewriter &rewriter) const override
                {

                    /*
                        Gain the tensors of interest from the IR.
                        We are matching on the pattern provided in "input.mlir".
                        %result = stablehlo.convolution(%input (lhs), %kernel (rhs))
                    */

                    RankedTensorType resultType = cast<RankedTensorType>(convOp.getResult().getType());
                    RankedTensorType kernelType = cast<RankedTensorType>(convOp.getRhs().getType());
                    RankedTensorType inputType = cast<RankedTensorType>(convOp.getLhs().getType());

                    if (!inputType || !kernelType || !resultType)
                    {
                        return failure();
                    }

                    // Debugging prints
                    printf("Input type \n");
                    inputType.dump();
                    printf("kernel type:\n");
                    kernelType.dump();
                    printf("resulttype:\n");
                    resultType.dump();

                    // Base case. Means pass already have been applied.
                    if (convOp->getAttr("data_layout"))
                    {
                        return failure();
                    }

                    /* --- Get data necessary for transforming convOp: --- */
                    Location loc = convOp.getLoc();
                    MLIRContext *context = rewriter.getContext();
                    Value inputTensor = convOp.getLhs();
                    Value kernelTensor = convOp.getRhs();
                    Value outputTensor = convOp.getResult();
                    uint64_t batchCount = convOp.getBatchGroupCount();          // 1 always (inference)
                    uint64_t featureGroupCount = convOp.getFeatureGroupCount(); // 1 --- KAN VÆRE RELEVANT, HVIS PACKED/BLOCKED LAYOUT?

                    /*These values are wrapped in a std::optional, therefore they need unwrapping to either own or default value. */

                    DenseI64ArrayAttr strides = convOp.getWindowStrides().has_value()
                                                    ? DenseI64ArrayAttr::get(context, convOp.getWindowStrides().value())
                                                    : DenseI64ArrayAttr::get(context, {1, 1}); // default

                    DenseIntElementsAttr padding = convOp.getPadding().value_or(
                        DenseIntElementsAttr::get(
                            RankedTensorType::get({2, 2}, mlir::IntegerType::get(context, 64)),
                            {0, 0,
                             0, 0}));

                    DenseI64ArrayAttr lhsDilation = convOp.getLhsDilation().has_value()
                                                        ? DenseI64ArrayAttr::get(context, convOp.getLhsDilation().value())
                                                        : DenseI64ArrayAttr::get(context, {1, 1}); // default

                    DenseI64ArrayAttr rhsDilation = convOp.getRhsDilation().has_value()
                                                        ? DenseI64ArrayAttr::get(context, convOp.getRhsDilation().value())
                                                        : DenseI64ArrayAttr::get(context, {1, 1}); // default

                    DenseBoolArrayAttr windowReversal = convOp.getWindowReversal().has_value()
                                                            ? DenseBoolArrayAttr::get(context, convOp.getWindowReversal().value())
                                                            : DenseBoolArrayAttr::get(context, {false, false}); // default

                    ArrayAttr precisionConfig = convOp.getPrecisionConfig().has_value()
                                                    ? ArrayAttr::get(context, convOp.getPrecisionConfig().value())
                                                    : nullptr;

                    // Provide the new dimensions for the convolution.
                    stablehlo::ConvDimensionNumbersAttr dimNumbers = stablehlo::ConvDimensionNumbersAttr::get(
                        context,
                        /*input_batch_dimension=*/0,                           // N
                        /*input_feature_dimension=*/3,                         // IC
                        /*input_spatial_dimensions=*/ArrayRef<int64_t>{1, 2},  // H,W
                        /*kernel_input_feature_dimension=*/2,                  // IC
                        /*kernel_output_feature_dimension=*/3,                 // OC
                        /*kernel_spatial_dimensions=*/ArrayRef<int64_t>{0, 1}, // H,W
                        /*output_batch_dimension=*/0,                          // N
                        /*output_feature_dimension=*/3,                        // OC
                        /*output_spatial_dimensions=*/ArrayRef<int64_t>{1, 2}  // H,W
                    );

                    // Shape for the new input and output tensor, based on old conv dims
                    // Default:NCHW
                    // Wanted; NHWC
                    DenseI64ArrayAttr permutation = DenseI64ArrayAttr::get(
                        rewriter.getContext(),
                        llvm::ArrayRef<int64_t>({0, 2, 3, 1}));

                    // Shape for the new kernel tensor based on the old dims
                    // Default:
                    // Wanted; K_H,K_W,IC,OC
                    DenseI64ArrayAttr permutationKernel = DenseI64ArrayAttr::get(
                        rewriter.getContext(),
                        llvm::ArrayRef<int64_t>({2, 3, 1, 0}) // kh,kw,ic,oc
                        //
                    );

                    SmallVector<int64_t> newShapeInput = {
                        inputType.getShape()[0],
                        inputType.getShape()[2],
                        inputType.getShape()[3],
                        inputType.getShape()[1],
                    };

                    SmallVector<int64_t> newShapeKernel = {
                        kernelType.getShape()[2],
                        kernelType.getShape()[3],
                        kernelType.getShape()[1],
                        kernelType.getShape()[0]};

                    SmallVector<int64_t> newShapeResult = {
                        resultType.getShape()[0],
                        resultType.getShape()[2],
                        resultType.getShape()[3],
                        resultType.getShape()[1],
                    };

                    /* ------------------------------------------------
                     * --------- Creating the new operations ----------
                     * ------------------------------------------------*/

                    // Transpose op for the input tensor
                    Value inputNHWC = rewriter.create<mlir::stablehlo::TransposeOp>(
                        loc,
                        RankedTensorType::get(newShapeInput, inputType.getElementType()),
                        inputTensor, // original input
                        permutation);

                    inputNHWC.dump();

                    // Transpose op for the kernel
                    Value kernelOHWI = rewriter.create<mlir::stablehlo::TransposeOp>(
                        loc,
                        RankedTensorType::get(newShapeKernel, kernelType.getElementType()),
                        kernelTensor, // original input
                        permutationKernel);

                    kernelOHWI.dump();

                    // Creating the new convolution op, and replacing the old.
                    auto newConv = rewriter.create<mlir::stablehlo::ConvolutionOp>(
                        loc,
                        RankedTensorType::get(newShapeResult, resultType.getElementType()),
                        inputNHWC,
                        kernelOHWI,
                        strides,
                        padding,
                        lhsDilation,
                        rhsDilation,
                        windowReversal,
                        dimNumbers,
                        featureGroupCount,
                        batchCount,
                        precisionConfig);

                    // For the base case.
                    newConv->setAttr("data_layout", rewriter.getBoolAttr(true));

                    /*******************************************************************
                     * Transforming result back to NHWC to allow old testing setup still (Not the most optimal.).
                     * REMOVE BEFORE ANY FINAL TESTING, ADDS UNFAIR OVERHEAD.
                     *******************************************************************/

                    // Default: NHWC
                    // Wanted; NCHW
                    DenseI64ArrayAttr permBackToDefault = DenseI64ArrayAttr::get(
                        rewriter.getContext(),
                        llvm::ArrayRef<int64_t>({0, 3, 1, 2})

                    );

                    // // Necessary to transpose the result back to NCHW, in order to check mathematical correctness with current testing setup.

                    /**************************************************************************************************
                     * PLEASE NOTE:ADDING A FINAL TRANSPOSE MAKES OUR PERFORMANCE *BETTER*, THAN NOT TRANPOSING IT BACK!
                     * This was discovered by accident.
                     * The reasoning seems to be, that MLIR is clever enough to optimize it away and use metadata instead. 
                     * Therefore I suggest keeping this. 
                     **************************************************************************************************/
                    Value backToDefault = rewriter.create<mlir::stablehlo::TransposeOp>(
                        loc,
                        resultType,          // original type
                        newConv.getResult(), // new result
                        permBackToDefault);

                    // llvm::errs() << "hej";

                    rewriter.replaceOp(convOp, backToDefault);

                    return success();
                }
            };
        } // anonymous namespace

        // Pass construction must be public (outside anonymous namespace)
        struct ConvertToNHWCPass
            : public PassWrapper<ConvertToNHWCPass, OperationPass<func::FuncOp>>
        {

            void runOnOperation() override
            {

                func::FuncOp funcOp = getOperation();                 // Inside a func.func region.
                RewritePatternSet patterns(funcOp.getContext());      // Create pattern set for our convolution.
                patterns.add<ConvertConv2DNchw>(funcOp.getContext()); // Provide the pattern we defines above.

                // Apply the pattern greedily.
                if (failed(applyPatternsGreedily(
                        funcOp, std::move(patterns))))
                    signalPassFailure();

                /*------------------------------------------------------------------ *
                  Pass transition: UPDATE THE RETURN TYPE, SO THE NEXT PASS KNOWS.
                  ------------------------------------------------------------------ */

                Operation *returnOp = funcOp.front().getTerminator(); // GetTeminator: Get the terminator (return) op (func.return), front(): first block of function (three is often only one)).

                Type newReturnType = returnOp->getOperand(0).getType(); // Get the output operand of the new convolution

                FunctionType oldType = funcOp.getFunctionType(); // keep arg0 as old input type (before transpose)

                FunctionType newFuncType = FunctionType::get(
                    funcOp.getContext(),
                    oldType.getInputs(),
                    {newReturnType}); // new return type replaces the old NCHW.

                funcOp.setType(newFuncType); // Update the return type of the func.func.
            }

            StringRef getName() const override
            {
                return "convert-nchw-to-nhwc";
            }
        };

        // Everyting implementation related goes to the anonymous namespace, while the factory function stays public.
        /**Register the pass for passmanager. */
        std::unique_ptr<Pass> createConvertToNHWCPass()
        {
            return std::make_unique<ConvertToNHWCPass>();
        }

    } // namespace transforms
} // namespace LasseThesis
