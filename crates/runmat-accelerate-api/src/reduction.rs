use runmat_execution::{identity::AttemptId, Digest};
use serde::{Deserialize, Serialize};

use crate::{
    GpuTensorHandle, NumericElementType, ProviderCancellationToken, ProviderCapabilitySnapshot,
    ProviderOperationFamily, ProviderOperationIdentity, PROVIDER_CAPABILITY_SCHEMA_VERSION,
};

pub const DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION: u16 = 1;
pub const GOVERNED_DENSE_REDUCTION_OPERATION: &str = "dense.reduction.governed";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
pub enum DenseReductionOperation {
    Sum { dimension: u32 },
    Minimum { dimension: u32 },
    Maximum { dimension: u32 },
}

impl DenseReductionOperation {
    pub fn dimension(self) -> usize {
        match self {
            Self::Sum { dimension } | Self::Minimum { dimension } | Self::Maximum { dimension } => {
                dimension as usize
            }
        }
    }

    pub fn returns_indices(self) -> bool {
        matches!(self, Self::Minimum { .. } | Self::Maximum { .. })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DenseReductionRequest<'a> {
    pub input: &'a GpuTensorHandle,
    pub operation: DenseReductionOperation,
    pub attempt_id: AttemptId,
    pub input_identity: Digest,
    pub provider_capability_schema_version: u16,
    pub provider_capability_revision: u64,
    pub cancellation: &'a ProviderCancellationToken,
}

impl DenseReductionRequest<'_> {
    pub fn validate(self, provider_device_id: u32) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.input.device_id == provider_device_id,
            "dense reduction input belongs to another provider"
        );
        anyhow::ensure!(
            !self.cancellation.is_cancelled(),
            "dense reduction was cancelled before dispatch"
        );
        anyhow::ensure!(
            self.input.descriptor.element_type == Some(NumericElementType::F64)
                && self.input.descriptor.storage == Some(crate::GpuTensorStorage::Real),
            "governed dense reduction requires real f64 input"
        );
        anyhow::ensure!(
            !self.input.shape.is_empty()
                && self.input.shape.iter().all(|extent| *extent > 0)
                && self.operation.dimension() < self.input.shape.len(),
            "dense reduction dimension or input shape is invalid"
        );
        anyhow::ensure!(
            self.provider_capability_schema_version == PROVIDER_CAPABILITY_SCHEMA_VERSION
                && self.provider_capability_revision > 0,
            "dense reduction capability binding is invalid"
        );
        Ok(())
    }

    pub fn expected_output_shape(self) -> Vec<usize> {
        let mut shape = self.input.shape.clone();
        shape[self.operation.dimension()] = 1;
        shape
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseReductionExecutionEvidence {
    pub schema_version: u16,
    pub attempt_id: AttemptId,
    pub input_identity: Digest,
    pub provider_device_id: u32,
    pub provider_capability_schema_version: u16,
    pub provider_capability_revision: u64,
    pub operation: DenseReductionOperation,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub dispatches: u32,
    pub host_upload_bytes: u64,
    pub host_download_bytes: u64,
    pub host_synchronizations: u32,
    pub execution_ns: u64,
    pub used_host_fallback: bool,
    pub cancellation_checks: u32,
    pub cancellation_observed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseReductionResult {
    pub values: GpuTensorHandle,
    pub indices: Option<GpuTensorHandle>,
    pub evidence: DenseReductionExecutionEvidence,
}

impl DenseReductionResult {
    pub fn validate(
        &self,
        request: DenseReductionRequest<'_>,
        capability: &ProviderCapabilitySnapshot,
    ) -> anyhow::Result<()> {
        request.validate(capability.device.device_id)?;
        anyhow::ensure!(
            capability.schema_version == request.provider_capability_schema_version
                && capability.revision == request.provider_capability_revision
                && capability.supports_operation(
                    &ProviderOperationIdentity::new(GOVERNED_DENSE_REDUCTION_OPERATION),
                    ProviderOperationFamily::Reduction,
                ),
            "provider capability does not admit governed dense reduction"
        );
        let expected_shape = request.expected_output_shape();
        let valid_output = |handle: &GpuTensorHandle| {
            handle.device_id == capability.device.device_id
                && handle.shape == expected_shape
                && handle.descriptor.element_type == Some(NumericElementType::F64)
                && handle.descriptor.storage == Some(crate::GpuTensorStorage::Real)
        };
        anyhow::ensure!(
            valid_output(&self.values),
            "dense reduction value output is invalid"
        );
        anyhow::ensure!(
            self.indices.as_ref().is_some_and(valid_output) == request.operation.returns_indices(),
            "dense reduction index output does not match its operation"
        );
        let input_bytes = checked_f64_bytes(&request.input.shape)?;
        let output_count = if request.operation.returns_indices() {
            2
        } else {
            1
        };
        let output_bytes = checked_f64_bytes(&expected_shape)?
            .checked_mul(output_count)
            .ok_or_else(|| anyhow::anyhow!("dense reduction output bytes overflow"))?;
        let evidence = &self.evidence;
        anyhow::ensure!(
            evidence.schema_version == DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION
                && evidence.attempt_id == request.attempt_id
                && evidence.input_identity == request.input_identity
                && evidence.provider_device_id == capability.device.device_id
                && evidence.provider_capability_schema_version == capability.schema_version
                && evidence.provider_capability_revision == capability.revision
                && evidence.operation == request.operation
                && evidence.input_bytes == input_bytes
                && evidence.output_bytes == output_bytes
                && evidence.dispatches > 0
                && evidence.cancellation_checks > 0
                && !evidence.cancellation_observed
                && (evidence.used_host_fallback
                    || (evidence.host_upload_bytes == 0
                        && evidence.host_download_bytes == 0
                        && evidence.host_synchronizations == 0)),
            "dense reduction evidence is invalid or belongs to another attempt"
        );
        Ok(())
    }
}

pub fn checked_f64_bytes(shape: &[usize]) -> anyhow::Result<u64> {
    shape
        .iter()
        .try_fold(1_u64, |count, extent| {
            count.checked_mul(u64::try_from(*extent).ok()?)
        })
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| anyhow::anyhow!("dense reduction byte size overflows"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuTensorStorage, ProviderCapabilityOperation};

    #[test]
    fn result_rejects_substituted_attempt_evidence() {
        let input = GpuTensorHandle::new(vec![3, 5], 7, 1)
            .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real);
        let cancellation = ProviderCancellationToken::new();
        let request = DenseReductionRequest {
            input: &input,
            operation: DenseReductionOperation::Maximum { dimension: 1 },
            attempt_id: AttemptId::derive(&[b"attempt"]),
            input_identity: Digest::sha256(b"input"),
            provider_capability_schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            provider_capability_revision: 4,
            cancellation: &cancellation,
        };
        let capability = ProviderCapabilitySnapshot {
            schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
            revision: 4,
            device: crate::ApiDeviceInfo {
                device_id: 7,
                name: "test".into(),
                vendor: "test".into(),
                memory_bytes: Some(1 << 20),
                backend: None,
            },
            operations: vec![ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(GOVERNED_DENSE_REDUCTION_OPERATION),
                family: ProviderOperationFamily::Reduction,
            }],
            element_types: vec![crate::ProviderElementType::F64],
            max_rank: Some(2),
            max_allocation_bytes: Some(1 << 20),
            sparse: None,
            concurrency: crate::ProviderConcurrencyCapabilities {
                spawn_handles: crate::SpawnHandleConcurrency::Reject,
                concurrent_dispatch: false,
                cancellation: false,
                transactional_results: true,
            },
        };
        let output = |id| {
            GpuTensorHandle::new(vec![3, 1], 7, id)
                .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real)
        };
        let mut result = DenseReductionResult {
            values: output(2),
            indices: Some(output(3)),
            evidence: DenseReductionExecutionEvidence {
                schema_version: DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION,
                attempt_id: request.attempt_id,
                input_identity: request.input_identity,
                provider_device_id: 7,
                provider_capability_schema_version: PROVIDER_CAPABILITY_SCHEMA_VERSION,
                provider_capability_revision: 4,
                operation: request.operation,
                input_bytes: 120,
                output_bytes: 48,
                dispatches: 1,
                host_upload_bytes: 0,
                host_download_bytes: 0,
                host_synchronizations: 0,
                execution_ns: 1,
                used_host_fallback: false,
                cancellation_checks: 1,
                cancellation_observed: false,
            },
        };
        result.validate(request, &capability).unwrap();
        result.evidence.attempt_id = AttemptId::derive(&[b"other"]);
        assert!(result.validate(request, &capability).is_err());
    }
}
