use runmat_accelerate_api::{SparseOperationEvidence, SPARSE_EVIDENCE_SCHEMA_VERSION};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

mod evidence_profile;

pub const ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "terminal")]
pub enum AcceleratorSolveEvidence {
    Completed {
        schema_version: u16,
        operator_identity: Digest,
        right_hand_side_identity: Digest,
        solution_identity: Digest,
        provider: SparseOperationEvidence,
        provider_reported_final_residual_norm: f64,
        terminal_solution_readbacks: u32,
        independently_computed_residual_norm: f64,
        independently_computed_relative_residual: f64,
        required_relative_tolerance: f64,
    },
    Cancelled {
        schema_version: u16,
        operator_identity: Digest,
        right_hand_side_identity: Digest,
        provider: SparseOperationEvidence,
    },
    ProviderLost {
        schema_version: u16,
        operator_identity: Digest,
        right_hand_side_identity: Digest,
        provider: SparseOperationEvidence,
    },
}

impl AcceleratorSolveEvidence {
    pub fn validate(&self) -> Result<(), AcceleratorAdapterError> {
        match self {
            Self::Completed {
                schema_version,
                provider_reported_final_residual_norm,
                terminal_solution_readbacks,
                independently_computed_residual_norm,
                independently_computed_relative_residual,
                required_relative_tolerance,
                provider,
                operator_identity,
                solution_identity: _,
                right_hand_side_identity: _,
            } => {
                if *schema_version != ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION
                    || !provider_reported_final_residual_norm.is_finite()
                    || *provider_reported_final_residual_norm < 0.0
                    || *terminal_solution_readbacks != 1
                    || !independently_computed_residual_norm.is_finite()
                    || *independently_computed_residual_norm < 0.0
                    || !independently_computed_relative_residual.is_finite()
                    || *independently_computed_relative_residual < 0.0
                    || !required_relative_tolerance.is_finite()
                    || *required_relative_tolerance <= 0.0
                    || provider.cancellation_observed
                    || provider.provider_loss.is_some()
                {
                    return Err(AcceleratorAdapterError::acceptance(
                        "completed accelerator solve evidence is invalid",
                    ));
                }
                validate_provider_evidence(provider, *operator_identity, false, false)?;
            }
            Self::Cancelled {
                schema_version,
                provider,
                operator_identity,
                right_hand_side_identity: _,
            } => {
                if *schema_version != ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION
                    || !provider.cancellation_observed
                    || provider.resident_solution_bytes != 0
                {
                    return Err(AcceleratorAdapterError::acceptance(
                        "cancelled accelerator solve evidence is invalid",
                    ));
                }
                validate_provider_evidence(provider, *operator_identity, true, false)?;
            }
            Self::ProviderLost {
                schema_version,
                provider,
                operator_identity,
                right_hand_side_identity: _,
            } => {
                if *schema_version != ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION
                    || provider.provider_loss.is_none()
                    || provider.cancellation_observed
                    || provider.resident_solution_bytes != 0
                {
                    return Err(AcceleratorAdapterError::acceptance(
                        "provider-lost accelerator solve evidence is invalid",
                    ));
                }
                validate_provider_evidence(provider, *operator_identity, false, true)?;
            }
        }
        Ok(())
    }

    pub fn satisfies_tolerance(&self) -> bool {
        matches!(
            self,
            Self::Completed {
                independently_computed_relative_residual,
                required_relative_tolerance,
                ..
            } if independently_computed_relative_residual <= required_relative_tolerance
        )
    }

    pub fn provider(&self) -> &SparseOperationEvidence {
        match self {
            Self::Completed { provider, .. }
            | Self::Cancelled { provider, .. }
            | Self::ProviderLost { provider, .. } => provider,
        }
    }

    pub fn independent_residual_norm(&self) -> Option<f64> {
        match self {
            Self::Completed {
                independently_computed_residual_norm,
                ..
            } => Some(*independently_computed_residual_norm),
            Self::Cancelled { .. } | Self::ProviderLost { .. } => None,
        }
    }

    pub fn independent_relative_residual(&self) -> Option<f64> {
        match self {
            Self::Completed {
                independently_computed_relative_residual,
                ..
            } => Some(*independently_computed_relative_residual),
            Self::Cancelled { .. } | Self::ProviderLost { .. } => None,
        }
    }

    pub fn terminal_solution_readbacks(&self) -> u32 {
        match self {
            Self::Completed {
                terminal_solution_readbacks,
                ..
            } => *terminal_solution_readbacks,
            Self::Cancelled { .. } | Self::ProviderLost { .. } => 0,
        }
    }
}

fn validate_provider_evidence(
    provider: &SparseOperationEvidence,
    operator_identity: Digest,
    cancelled: bool,
    provider_lost: bool,
) -> Result<(), AcceleratorAdapterError> {
    let binding = &provider.binding;
    let required_resident_bytes = provider
        .resident_pattern_bytes
        .checked_add(provider.resident_matrix_bytes)
        .and_then(|bytes| bytes.checked_add(provider.resident_preconditioner_bytes))
        .and_then(|bytes| bytes.checked_add(provider.resident_solution_bytes))
        .and_then(|bytes| bytes.checked_add(provider.scratch_bytes))
        .ok_or_else(|| {
            AcceleratorAdapterError::acceptance(
                "accelerator provider evidence resident storage overflows",
            )
        })?;
    if provider.schema_version != SPARSE_EVIDENCE_SCHEMA_VERSION
        || provider.used_host_fallback
        || binding.operator_identity != operator_identity
        || binding.provider_capability_schema_version == 0
        || binding.provider_capability_revision == 0
        || provider.peak_resident_bytes < required_resident_bytes
        || provider.cancellation_observed != cancelled
        || provider.provider_loss.is_some() != provider_lost
        || ((cancelled || provider_lost) && provider.resident_solution_bytes != 0)
        || (!cancelled && !provider_lost && provider.resident_solution_bytes == 0)
    {
        return Err(AcceleratorAdapterError::acceptance(
            "accelerator provider evidence is invalid or bound to another execution",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcceleratorAdapterPhase {
    Admission,
    Preparation,
    Execution,
    Cancellation,
    ProviderLoss,
    Acceptance,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AcceleratorAdapterError {
    pub phase: AcceleratorAdapterPhase,
    pub message: String,
    pub evidence: Option<Box<AcceleratorSolveEvidence>>,
}

impl AcceleratorAdapterError {
    pub(super) fn admission(message: impl Into<String>) -> Self {
        Self::new(AcceleratorAdapterPhase::Admission, message)
    }

    pub(super) fn preparation(message: impl Into<String>) -> Self {
        Self::new(AcceleratorAdapterPhase::Preparation, message)
    }

    pub(super) fn execution(message: impl Into<String>) -> Self {
        Self::new(AcceleratorAdapterPhase::Execution, message)
    }

    pub(super) fn cancellation(
        message: impl Into<String>,
        evidence: AcceleratorSolveEvidence,
    ) -> Self {
        Self {
            phase: AcceleratorAdapterPhase::Cancellation,
            message: message.into(),
            evidence: Some(Box::new(evidence)),
        }
    }

    pub(super) fn provider_loss(
        message: impl Into<String>,
        evidence: AcceleratorSolveEvidence,
    ) -> Self {
        Self {
            phase: AcceleratorAdapterPhase::ProviderLoss,
            message: message.into(),
            evidence: Some(Box::new(evidence)),
        }
    }

    pub(super) fn acceptance(message: impl Into<String>) -> Self {
        Self::new(AcceleratorAdapterPhase::Acceptance, message)
    }

    fn new(phase: AcceleratorAdapterPhase, message: impl Into<String>) -> Self {
        Self {
            phase,
            message: message.into(),
            evidence: None,
        }
    }

    pub fn permits_cpu_fallback(&self) -> bool {
        self.phase == AcceleratorAdapterPhase::Admission
    }
}

impl std::fmt::Display for AcceleratorAdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "accelerator {:?} failed: {}",
            self.phase, self.message
        )
    }
}

impl std::error::Error for AcceleratorAdapterError {}

#[cfg(test)]
mod tests {
    use runmat_accelerate_api::{
        SparseExecutionBinding, SparseOperationTimings, SparseProviderLossReason,
    };

    use super::*;

    fn provider_loss_evidence(operator_identity: Digest) -> SparseOperationEvidence {
        SparseOperationEvidence {
            schema_version: SPARSE_EVIDENCE_SCHEMA_VERSION,
            binding: SparseExecutionBinding {
                attempt_id: runmat_execution::identity::AttemptId::derive(&[
                    b"fea-provider-loss-test",
                ]),
                operator_identity,
                provider_device_id: 7,
                provider_capability_schema_version: 4,
                provider_capability_revision: 2,
                pattern_id: 3,
                matrix_id: 5,
                preconditioner_id: Some(11),
            },
            timings: SparseOperationTimings::default(),
            resident_pattern_bytes: 0,
            resident_matrix_bytes: 0,
            resident_preconditioner_bytes: 0,
            resident_solution_bytes: 0,
            host_upload_bytes: 0,
            host_download_bytes: 16,
            setup_dispatches: 1,
            spmv_dispatches: 1,
            preconditioner_dispatches: 1,
            vector_dispatches: 4,
            host_synchronizations: 1,
            device_readbacks: 1,
            peak_resident_bytes: 0,
            scratch_bytes: 0,
            cancellation_checks: 2,
            cancellation_observed: false,
            provider_loss: Some(SparseProviderLossReason::Destroyed),
            used_host_fallback: false,
        }
    }

    #[test]
    fn provider_loss_evidence_is_typed_and_cannot_impersonate_cancellation() {
        let operator_identity = Digest::sha256(b"fea-provider-loss-operator");
        let right_hand_side_identity = Digest::sha256(b"fea-provider-loss-rhs");
        let valid = AcceleratorSolveEvidence::ProviderLost {
            schema_version: ACCELERATOR_SOLVE_EVIDENCE_SCHEMA_VERSION,
            operator_identity,
            right_hand_side_identity,
            provider: provider_loss_evidence(operator_identity),
        };
        valid.validate().expect("valid typed provider loss");

        let mut forged_cancellation = valid.clone();
        let AcceleratorSolveEvidence::ProviderLost { provider, .. } = &mut forged_cancellation
        else {
            unreachable!()
        };
        provider.cancellation_observed = true;
        assert!(forged_cancellation.validate().is_err());

        let mut missing_reason = valid;
        let AcceleratorSolveEvidence::ProviderLost { provider, .. } = &mut missing_reason else {
            unreachable!()
        };
        provider.provider_loss = None;
        assert!(missing_reason.validate().is_err());
    }
}
