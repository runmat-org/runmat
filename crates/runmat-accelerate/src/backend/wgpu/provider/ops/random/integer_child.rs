//! Request-bound native integer child-stream candidates: fixed unbiased masks
//! and bounded rejection batches share one provider feasibility boundary.

use super::super::{
    backend_types::NumericBufferRegistration, core::map_completion_host_upper_bytes,
    NumericPrecision, WgpuProvider,
};
use crate::backend::wgpu::residency::BufferUsageClass;
use anyhow::{anyhow, ensure, Result};
use runmat_accelerate_api::random::{
    GpuIntegerBatchPhysicalProposal, GpuIntegerBatchProgress, GpuIntegerBatchRequest,
    GpuIntegerPending, GpuIntegerRequest, GpuRandomPhysicalProposal,
    PROVIDER_GPU_INTEGER_BATCH_OPERATION, PROVIDER_GPU_INTEGER_OPERATION,
};
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderElementType,
    ProviderExecutionError, ProviderExecutionResult, ProviderFeasibility, ProviderFeasibilityQuery,
    ProviderLayout, ProviderOperationFamily, ProviderOperationIdentity, ProviderRejectionCode,
    ProviderResidency, ProviderResourceEstimate, ProviderStorage,
};
use runmat_execution::resource::ProviderHostMetadataAdmission;

#[path = "integer_child/batch.rs"]
mod batch;
#[path = "integer_child/feasibility.rs"]
mod feasibility;
#[path = "integer_child/fixed.rs"]
mod fixed;
