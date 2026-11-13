use runmat_canonical_codec::encode_contract;
use runmat_execution::{identity::AttemptId, resource::AcceleratorRequest, Digest};
use serde::{Deserialize, Serialize};

use super::{contract::OBSERVATION_LIMITS, ObservationError};
use crate::planning::resource::FeaResourcePlanIdentity;

pub const OBSERVATION_PROVIDER_REDUCTION_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationExtremum {
    Minimum,
    Maximum,
}

/// One resident provider extremum over the selected values, as observed by the
/// FEA caller: device and capability binding, exact transfer sizes, and time.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationProviderReduction {
    pub schema_version: u16,
    pub attempt_id: AttemptId,
    pub input_identity: Digest,
    pub provider_device_id: u32,
    pub provider_capability_schema_version: u16,
    pub provider_capability_revision: u64,
    pub operation: ObservationExtremum,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub dispatches: u32,
    pub execution_ns: u64,
}

/// Canonical encoded-size ceiling for one governed dense-reduction evidence record. The record
/// contains only fixed-width fields plus one accelerator class bounded by `AcceleratorRequest`.
pub const OBSERVATION_ACCELERATION_EVIDENCE_MAX_ENCODED_BYTES: u64 = 64 * 1024;

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
    pub resource_plan: FeaResourcePlanIdentity,
    pub selected_values_identity: Digest,
    pub allocation: AcceleratorRequest,
    pub provider: ObservationProviderReduction,
    pub input_upload_bytes: u64,
    pub result_download_bytes: u64,
    pub terminal_synchronizations: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAccelerationEvidenceWire {
    schema_version: u16,
    identity: ObservationAccelerationEvidenceIdentity,
    resource_plan: FeaResourcePlanIdentity,
    selected_values_identity: Digest,
    allocation: AcceleratorRequest,
    provider: ObservationProviderReduction,
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
            resource_plan: wire.resource_plan,
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
        resource_plan: FeaResourcePlanIdentity,
        selected_values_identity: Digest,
        allocation: AcceleratorRequest,
        provider: ObservationProviderReduction,
        input_upload_bytes: u64,
        result_download_bytes: u64,
        terminal_synchronizations: u32,
    ) -> Result<Self, ObservationError> {
        let mut evidence = Self {
            schema_version: OBSERVATION_PROVIDER_REDUCTION_SCHEMA_VERSION,
            identity: ObservationAccelerationEvidenceIdentity(Digest::sha256([])),
            resource_plan,
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
        if self.schema_version != OBSERVATION_PROVIDER_REDUCTION_SCHEMA_VERSION
            || self.provider.schema_version != OBSERVATION_PROVIDER_REDUCTION_SCHEMA_VERSION
            || self.provider.input_identity != self.selected_values_identity
            || self.provider.provider_capability_revision == 0
            || self.allocation.count != 1
            || self.allocation.minimum_allocation_bytes == 0
            || self.provider.dispatches == 0
            || self.provider.execution_ns == 0
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
        if self.allocation != *allocation || resident_bytes > allocation.minimum_allocation_bytes {
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
                self.resource_plan,
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

/// Compact publication record for one completed provider operation and its exact admitted plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationAccelerationOperationSummary {
    pub evidence: ObservationAccelerationEvidenceIdentity,
    pub resource_plan: FeaResourcePlanIdentity,
    pub input_upload_bytes: u64,
    pub result_download_bytes: u64,
    pub dispatches: u64,
    pub execution_ns: u64,
    pub terminal_synchronizations: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationAccelerationSummaryWire")]
pub struct ObservationAccelerationSummary {
    pub schema_version: u16,
    pub identity: ObservationAccelerationSummaryIdentity,
    pub allocation: AcceleratorRequest,
    pub operations: Vec<ObservationAccelerationOperationSummary>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAccelerationSummaryWire {
    schema_version: u16,
    identity: ObservationAccelerationSummaryIdentity,
    allocation: AcceleratorRequest,
    operations: Vec<ObservationAccelerationOperationSummary>,
}

impl TryFrom<ObservationAccelerationSummaryWire> for ObservationAccelerationSummary {
    type Error = ObservationError;

    fn try_from(wire: ObservationAccelerationSummaryWire) -> Result<Self, Self::Error> {
        let summary = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            allocation: wire.allocation,
            operations: wire.operations,
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
            operations: evidence
                .iter()
                .map(|item| ObservationAccelerationOperationSummary {
                    evidence: item.identity,
                    resource_plan: item.resource_plan,
                    input_upload_bytes: item.input_upload_bytes,
                    result_download_bytes: item.result_download_bytes,
                    dispatches: u64::from(item.provider.dispatches),
                    execution_ns: item.provider.execution_ns,
                    terminal_synchronizations: u64::from(item.terminal_synchronizations),
                })
                .collect(),
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
            .operations
            .iter()
            .map(|operation| operation.evidence)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == self.operations.len();
        let plans_are_unique = self
            .operations
            .iter()
            .map(|operation| operation.resource_plan)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == self.operations.len();
        let operations_are_valid = self.operations.iter().all(|operation| {
            operation.input_upload_bytes > 0
                && operation.result_download_bytes > 0
                && operation.dispatches > 0
                && operation.execution_ns > 0
                && operation.terminal_synchronizations > 0
        });
        let totals_fit = checked_sum(
            self.operations
                .iter()
                .map(|operation| operation.execution_ns),
            "observation acceleration execution time",
        )
        .is_ok();
        if self.schema_version != 1
            || self.allocation.count != 1
            || self.allocation.minimum_allocation_bytes == 0
            || self.operations.is_empty()
            || !evidence_is_unique
            || !plans_are_unique
            || !operations_are_valid
            || !totals_fit
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
            &(self.schema_version, &self.allocation, &self.operations),
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
