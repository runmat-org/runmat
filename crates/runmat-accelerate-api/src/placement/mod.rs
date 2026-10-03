//! Side-effect-free provider discovery and feasibility contracts.
//!
//! These types describe what a provider can execute; they do not select where
//! an operation should execute. Selection and profitability remain placement
//! concerns in `runmat-accelerate`, while command scheduling remains owned by
//! each provider implementation.

mod capability;
mod convolution;
pub use convolution::{
    ProviderConvolutionForm, ProviderConvolutionGeometry, ProviderConvolutionInputBackings,
    ProviderConvolutionWorkload,
};
mod cost;
mod execution_inventory;
mod feasibility;
mod fspecial;
pub use fspecial::ProviderFspecialFilter;
mod imfilter;
pub use imfilter::{ProviderImfilterInputBackings, ProviderImfilterWorkload};
mod iir;
pub use iir::{
    ProviderIirRoute, ProviderIirWorkload, PROVIDER_IIR_CAUSAL_PASS_OPERATION,
    PROVIDER_IIR_PREPARED_PASS_OPERATION, PROVIDER_IIR_ZERO_PHASE_OPERATION,
};
mod matmul;
pub use matmul::{MatmulScaleWork, MatmulWorkload, ProviderMatmulGeometry};
mod representation;
mod sum;
pub use sum::{ProviderSumGeometry, ProviderSumWorkload};
mod preparation;
pub use preparation::{
    ProviderElementwiseGeometry, ProviderPreparationProgram, ProviderPreparationRoute,
    ProviderPreparationSnapshot,
};

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
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderOperationIdentity, ProviderRejection,
    ProviderRejectionCode, ProviderResourceEstimate, ProviderTransferOutputBound, ProviderWorkload,
};
pub use representation::{
    ProviderElementType, ProviderLayout, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
