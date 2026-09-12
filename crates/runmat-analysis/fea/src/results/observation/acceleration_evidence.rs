use runmat_accelerate_api::{
    DenseReductionExecutionEvidence, DenseReductionOperation,
    DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION,
};
use runmat_canonical_codec::encode_contract;
use runmat_execution::{
    resource::{AcceleratorRequest, ResourceRequest, ResourceUsage},
    Digest,
};
use serde::{Deserialize, Serialize};

use super::{contract::OBSERVATION_LIMITS, ObservationError};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationAccelerationEvidenceIdentity(Digest);

impl ObservationAccelerationEvidenceIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationAccelerationSummaryIdentity(Digest);

impl ObservationAccelerationSummaryIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationAccelerationEvidenceWire")]
pub struct ObservationAccelerationEvidence {
    pub schema_version: u16,
    pub identity: ObservationAccelerationEvidenceIdentity,
    pub selected_values_identity: Digest,
    pub allocation: AcceleratorRequest,
    pub provider: DenseReductionExecutionEvidence,
    pub input_upload_bytes: u64,
    pub result_download_bytes: u64,
    pub terminal_synchronizations: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAccelerationEvidenceWire {
    schema_version: u16,
    identity: ObservationAccelerationEvidenceIdentity,
    selected_values_identity: Digest,
    allocation: AcceleratorRequest,
    provider: DenseReductionExecutionEvidence,
    input_upload_bytes: u64,
    result_download_bytes: u64,
    terminal_synchronizations: u32,
}

impl TryFrom<ObservationAccelerationEvidenceWire> for ObservationAccelerationEvidence {
    type Error = ObservationError;

    fn try_from(wire: ObservationAccelerationEvidenceWire) -> Result<Self, Self::Error> {
        let evidence = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            selected_values_identity: wire.selected_values_identity,
            allocation: wire.allocation,
            provider: wire.provider,
            input_upload_bytes: wire.input_upload_bytes,
            result_download_bytes: wire.result_download_bytes,
            terminal_synchronizations: wire.terminal_synchronizations,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl ObservationAccelerationEvidence {
    pub(super) fn new(
        selected_values_identity: Digest,
        allocation: AcceleratorRequest,
        provider: DenseReductionExecutionEvidence,
        input_upload_bytes: u64,
        result_download_bytes: u64,
        terminal_synchronizations: u32,
    ) -> Result<Self, ObservationError> {
        let mut evidence = Self {
            schema_version: 1,
            identity: ObservationAccelerationEvidenceIdentity(Digest::sha256([])),
            selected_values_identity,
            allocation,
            provider,
            input_upload_bytes,
            result_download_bytes,
            terminal_synchronizations,
        };
        evidence.identity = evidence.derive_identity()?;
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        self.allocation
            .validate()
            .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        let operation_is_extrema = matches!(
            self.provider.operation,
            DenseReductionOperation::Minimum { .. } | DenseReductionOperation::Maximum { .. }
        );
        if self.schema_version != 1
            || self.provider.schema_version != DENSE_REDUCTION_EVIDENCE_SCHEMA_VERSION
            || self.provider.input_identity != self.selected_values_identity
            || !operation_is_extrema
            || self.allocation.count != 1
            || self.allocation.memory_bytes_each == 0
            || self.provider.used_host_fallback
            || self.provider.cancellation_observed
            || self.provider.cancellation_checks == 0
            || self.provider.dispatches == 0
            || self.provider.execution_ns == 0
            || self.provider.host_upload_bytes != 0
            || self.provider.host_download_bytes != 0
            || self.provider.host_synchronizations != 0
            || self.input_upload_bytes != self.provider.input_bytes
            || self.result_download_bytes != self.provider.output_bytes
            || self.terminal_synchronizations != 2
            || self.identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation acceleration evidence is inconsistent".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_for_allocation(
        &self,
        allocation: &AcceleratorRequest,
    ) -> Result<(), ObservationError> {
        self.validate()?;
        allocation
            .validate()
            .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        let resident_bytes = self
            .input_upload_bytes
            .checked_add(self.result_download_bytes)
            .ok_or(ObservationError::Limit(
                "observation accelerator resident bytes",
            ))?;
        if self.allocation != *allocation || resident_bytes > allocation.memory_bytes_each {
            return Err(ObservationError::Identity(
                "observation accelerator evidence differs from its admitted allocation",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ObservationAccelerationEvidenceIdentity, ObservationError> {
        let bytes = encode_contract(
            super::contract::OBSERVATION_PREFIX,
            "observation-acceleration-evidence-identity",
            &(
                self.schema_version,
                self.selected_values_identity,
                &self.allocation,
                &self.provider,
                self.input_upload_bytes,
                self.result_download_bytes,
                self.terminal_synchronizations,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationAccelerationEvidenceIdentity(Digest::sha256(
            bytes,
        )))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationAccelerationSummaryWire")]
pub struct ObservationAccelerationSummary {
    pub schema_version: u16,
    pub identity: ObservationAccelerationSummaryIdentity,
    pub allocation: AcceleratorRequest,
    pub evidence: Vec<ObservationAccelerationEvidenceIdentity>,
    pub input_upload_bytes: u64,
    pub result_download_bytes: u64,
    pub dispatches: u64,
    pub execution_ns: u64,
    pub terminal_synchronizations: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAccelerationSummaryWire {
    schema_version: u16,
    identity: ObservationAccelerationSummaryIdentity,
    allocation: AcceleratorRequest,
    evidence: Vec<ObservationAccelerationEvidenceIdentity>,
    input_upload_bytes: u64,
    result_download_bytes: u64,
    dispatches: u64,
    execution_ns: u64,
    terminal_synchronizations: u64,
}

impl TryFrom<ObservationAccelerationSummaryWire> for ObservationAccelerationSummary {
    type Error = ObservationError;

    fn try_from(wire: ObservationAccelerationSummaryWire) -> Result<Self, Self::Error> {
        let summary = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            allocation: wire.allocation,
            evidence: wire.evidence,
            input_upload_bytes: wire.input_upload_bytes,
            result_download_bytes: wire.result_download_bytes,
            dispatches: wire.dispatches,
            execution_ns: wire.execution_ns,
            terminal_synchronizations: wire.terminal_synchronizations,
        };
        summary.validate()?;
        Ok(summary)
    }
}

impl ObservationAccelerationSummary {
    pub fn from_evidence(
        evidence: &[ObservationAccelerationEvidence],
    ) -> Result<Option<Self>, ObservationError> {
        let Some(first) = evidence.first() else {
            return Ok(None);
        };
        for operation in evidence {
            operation.validate_for_allocation(&first.allocation)?;
        }
        let mut summary = Self {
            schema_version: 1,
            identity: ObservationAccelerationSummaryIdentity(Digest::sha256([])),
            allocation: first.allocation.clone(),
            evidence: evidence.iter().map(|item| item.identity).collect(),
            input_upload_bytes: checked_sum(
                evidence.iter().map(|item| item.input_upload_bytes),
                "observation acceleration upload bytes",
            )?,
            result_download_bytes: checked_sum(
                evidence.iter().map(|item| item.result_download_bytes),
                "observation acceleration download bytes",
            )?,
            dispatches: checked_sum(
                evidence
                    .iter()
                    .map(|item| u64::from(item.provider.dispatches)),
                "observation acceleration dispatches",
            )?,
            execution_ns: checked_sum(
                evidence.iter().map(|item| item.provider.execution_ns),
                "observation acceleration execution time",
            )?,
            terminal_synchronizations: checked_sum(
                evidence
                    .iter()
                    .map(|item| u64::from(item.terminal_synchronizations)),
                "observation acceleration synchronizations",
            )?,
        };
        summary.identity = summary.derive_identity()?;
        summary.validate()?;
        Ok(Some(summary))
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        self.allocation
            .validate()
            .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        let evidence_is_unique = self
            .evidence
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == self.evidence.len();
        if self.schema_version != 1
            || self.allocation.count != 1
            || self.allocation.memory_bytes_each == 0
            || self.evidence.is_empty()
            || !evidence_is_unique
            || self.input_upload_bytes == 0
            || self.result_download_bytes == 0
            || self.dispatches == 0
            || self.execution_ns == 0
            || self.terminal_synchronizations == 0
            || self.identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation acceleration summary is inconsistent".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ObservationAccelerationSummaryIdentity, ObservationError> {
        let bytes = encode_contract(
            super::contract::OBSERVATION_PREFIX,
            "observation-acceleration-summary-identity",
            &(
                self.schema_version,
                &self.allocation,
                &self.evidence,
                self.input_upload_bytes,
                self.result_download_bytes,
                self.dispatches,
                self.execution_ns,
                self.terminal_synchronizations,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationAccelerationSummaryIdentity(Digest::sha256(
            bytes,
        )))
    }
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    field: &'static str,
) -> Result<u64, ObservationError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or(ObservationError::Limit(field))
    })
}

pub(super) fn validate_acceleration_publication(
    summaries: &[&ObservationAccelerationSummary],
    resource_plan: &ResourceRequest,
    resource_usage: &ResourceUsage,
) -> Result<(), ObservationError> {
    resource_plan
        .validate()
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    if summaries.is_empty() {
        return Ok(());
    }
    let [allocation] = resource_plan.accelerators.as_slice() else {
        return Err(ObservationError::Identity(
            "observation acceleration has no unique resource-plan allocation",
        ));
    };
    if summaries
        .iter()
        .any(|summary| summary.allocation != *allocation)
    {
        return Err(ObservationError::Identity(
            "observation acceleration differs from its resource-plan allocation",
        ));
    }
    let matching_usage = resource_usage
        .accelerator_millis
        .iter()
        .filter(|usage| usage.class == allocation.class)
        .collect::<Vec<_>>();
    let [matching_usage] = matching_usage.as_slice() else {
        return Err(ObservationError::Identity(
            "observation acceleration has no unique scheduler usage record",
        ));
    };
    let measured_execution_ns = checked_sum(
        summaries.iter().map(|summary| summary.execution_ns),
        "observation acceleration execution time",
    )?;
    let reserved_execution_ns =
        matching_usage
            .device_millis
            .checked_mul(1_000_000)
            .ok_or(ObservationError::Limit(
                "observation accelerator reserved execution time",
            ))?;
    if measured_execution_ns > reserved_execution_ns {
        return Err(ObservationError::Invalid(
            "observation accelerator execution exceeds scheduler accounting".into(),
        ));
    }
    Ok(())
}
