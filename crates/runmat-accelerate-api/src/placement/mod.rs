//! Side-effect-free provider discovery and feasibility contracts.
//!
//! These types describe what a provider can execute; they do not select where
//! an operation should execute. Selection and profitability remain placement
//! concerns in `runmat-accelerate`, while command scheduling remains owned by
//! each provider implementation.

mod capability;
pub(crate) mod compound_execution;
pub use compound_execution::ProviderCompoundCancelled;
mod convolution;
pub use convolution::{
    ProviderConvolutionForm, ProviderConvolutionGeometry, ProviderConvolutionInputBackings,
    ProviderConvolutionWorkload,
};
mod cost;
mod execution_inventory;
mod extrema;
mod feasibility;
pub use extrema::{
    ProviderCheckedGlobalExtrema, ProviderExecutedGlobalExtrema, ProviderExtremaDemand,
    ProviderExtremaGeometry, ProviderExtremaInputBacking, ProviderExtremaKind,
    ProviderExtremaWorkload, ProviderGlobalExtremaObserved, ProviderGlobalExtremaOutputRoute,
    ProviderGlobalExtremaPhysicalProposal, ProviderGlobalExtremaRecipe,
    ProviderGlobalExtremaRequest, ProviderGlobalExtremaScalar,
};
mod image_stages;
mod reduction_nd;
pub use image_stages::{
    moment_precision, ProviderImageMeanCall, ProviderImageMeanClass, ProviderImageMeanHook,
    ProviderImageMomentConversion, ProviderImageMomentProgram, ProviderImageMomentPublication,
    ProviderImageOutputCall, ProviderImageOutputHook, ProviderImageOutputOperand,
    ProviderImageOutputProgram, ProviderImagePowerCall, ProviderImagePowerProgram,
    ProviderImageStagePhysicalWork, ProviderImageStageWorkload,
};
pub use reduction_nd::{
    ProviderColdNdMean, ProviderExecutedColdNdMean, ProviderExecutedOrderedRealMean,
    ProviderOrderedRealMean, ProviderOrderedRealMeanControl, ProviderOrderedRealMeanLimits,
    ProviderOrderedRealMeanRecipe, ProviderOrderedRealMeanResources, ProviderOrderedRealMeanSource,
    ProviderOrderedRealMeanStep, ProviderPinnedNdMean, ProviderRealMeanNaNMode,
};
mod fspecial;
pub use fspecial::ProviderFspecialFilter;
mod imfilter;
pub use imfilter::{
    ProviderImfilterGeometry, ProviderImfilterInputBackings, ProviderImfilterPhysicalEvidence,
    ProviderImfilterPhysicalRoute, ProviderImfilterPreparedKernel, ProviderImfilterWorkload,
};
mod iir;
pub use iir::{
    ProviderIirRoute, ProviderIirWorkload, PROVIDER_IIR_CAUSAL_PASS_OPERATION,
    PROVIDER_IIR_PREPARED_PASS_OPERATION, PROVIDER_IIR_ZERO_PHASE_OPERATION,
};
mod matmul;
pub use matmul::{MatmulScaleWork, MatmulWorkload, ProviderMatmulGeometry};
mod representation;
mod real_elementwise;
mod real_unary;
pub use real_elementwise::{
    ProviderExecutedMixedRealAdd, ProviderExecutedRealElementwise, ProviderMixedRealAddObserved,
    ProviderMixedRealAddRecipe, ProviderMixedRealAddResources,
    ProviderMixedRealAddStepObservations, ProviderRealElementwiseObserved,
    ProviderRealElementwiseOperation, ProviderRealElementwiseOutputRoute,
    ProviderRealElementwisePhysicalProposal, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest,
};
pub use real_unary::{
    ProviderRealUnaryCompound, ProviderRealUnaryContinuation, ProviderRealUnaryContinuationKind,
    ProviderRealUnaryContinuations, ProviderRealUnaryDomain, ProviderRealUnaryExecuted,
    ProviderRealUnaryExecution, ProviderRealUnaryExecutionEvidence, ProviderRealUnaryPublication,
};
mod real_binary;
pub use real_binary::{
    ProviderExecutedRealBinary, ProviderRealBinaryInput, ProviderRealBinaryObserved,
    ProviderRealBinaryOperation, ProviderRealBinaryPhysicalProposal, ProviderRealBinaryRecipe,
    ProviderRealBinaryRequest, ProviderRealBinaryRoute,
};
mod sum;
pub use sum::{ProviderSumGeometry, ProviderSumWorkload};
mod typed_reduction;
pub use typed_reduction::ProviderTypedReductionWorkload;
mod preparation;
mod query_metadata;
pub use preparation::{
    ProviderClassPreservingUpload, ProviderElementwiseGeometry, ProviderPreparationProgram,
    ProviderPreparationRoute, ProviderPreparationSnapshot,
};
pub use query_metadata::ProviderQueryMetadataFootprint;

pub use capability::{
    ProviderCapabilityOperation, ProviderCapabilitySnapshot, ProviderConcurrencyCapabilities,
    ProviderOperationFamily, PROVIDER_CAPABILITY_SCHEMA_VERSION,
};
pub use cost::{ProviderCostContext, ProviderCostEstimate, ProviderCostQuery};
pub use execution_inventory::{
    execution_accelerator_device, execution_provider_contract, ExecutionProviderContract,
    EXECUTION_PROVIDER_ABI_SCHEMA_VERSION, RUNMAT_BUILTIN_PROVIDER_VERSION,
    RUNMAT_CUDA_PROVIDER_ID, RUNMAT_WGPU_PROVIDER_ID,
};
pub use feasibility::{
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderFloatingCastPhysicalEvidence,
    ProviderNumericUploadPhysicalEvidence, ProviderOperationIdentity,
    ProviderProducedNumericReadbackEvidence, ProviderRejection, ProviderRejectionCode,
    ProviderResourceEstimate, ProviderTransferOutputBound, ProviderUploadPhysicalEvidence,
    ProviderWorkload,
};
pub use representation::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
