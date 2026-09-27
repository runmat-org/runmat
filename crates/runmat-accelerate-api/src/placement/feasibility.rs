use serde::{Deserialize, Serialize};

use crate::ProviderPrecision;

#[cfg(test)]
#[path = "workload_tests.rs"]
mod workload_tests;

use super::{
    ProviderElementType, ProviderLayout, ProviderOperationFamily, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderOperationIdentity(pub String);

impl ProviderOperationIdentity {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWorkload {
    /// Exact admitted matrix/epilogue route. None is unknown, not a bare GEMM.
    #[serde(default)]
    pub matmul: Option<super::MatmulWorkload>,
    pub elements: Option<u64>,
    pub flops: Option<u64>,
    /// Total operation-owned scalar work prior, not a FLOP count or a timing.
    /// Missing means no resolved estimate; it must not be interpreted as zero.
    #[serde(default)]
    pub scalar_work_units: Option<u64>,
    pub batch: Option<u64>,
    /// Language-requested outputs; None is the operation's implicit default,
    /// not zero. This distinguishes resource work and placement feedback keys.
    #[serde(default)]
    pub requested_outputs: Option<u32>,
    #[serde(default)]
    pub numerical_domain: crate::NumericalDomainEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderFeasibilityQuery {
    pub operation: ProviderOperationIdentity,
    pub family: ProviderOperationFamily,
    pub inputs: Vec<ProviderRepresentation>,
    pub outputs: Vec<ProviderRepresentation>,
    pub workload: ProviderWorkload,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResourceEstimate {
    pub transient_bytes: Option<u64>,
    pub output_bytes: Option<u64>,
    pub dispatches: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRejectionCode {
    UnsupportedOperation,
    UnsupportedElementType,
    UnsupportedStorage,
    UnsupportedLayout,
    UnsupportedRank,
    InvalidShape,
    ResourceLimit,
    ProviderUnavailable,
    /// Execution is supported, but automatic placement lacks a required
    /// physical resource estimate. This does not assert an exceeded limit.
    UnknownResourceEstimate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRejection {
    pub code: ProviderRejectionCode,
    /// Stable provider-owned diagnostic token, never arbitrary error text.
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum ProviderFeasibility {
    Supported { estimate: ProviderResourceEstimate },
    Rejected { rejection: ProviderRejection },
}

impl ProviderFeasibility {
    pub fn supported(estimate: ProviderResourceEstimate) -> Self {
        Self::Supported { estimate }
    }

    pub fn rejected(code: ProviderRejectionCode, detail: &'static str) -> Self {
        Self::Rejected {
            rejection: ProviderRejection {
                code,
                detail: Some(detail.to_string()),
            },
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

impl ProviderFeasibilityQuery {
    pub(crate) fn conservative_transfer_feasibility(
        &self,
        precision: ProviderPrecision,
    ) -> ProviderFeasibility {
        let expected_identity = match self.family {
            ProviderOperationFamily::Upload => "transfer.upload",
            ProviderOperationFamily::Download => "transfer.download",
            _ => {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedOperation,
                    "provider.operation.unsupported",
                )
            }
        };
        if self.operation.0 != expected_identity {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedOperation,
                "provider.operation.unsupported",
            );
        }
        let expected = ProviderElementType::from(precision);
        for representation in self.inputs.iter().chain(&self.outputs) {
            if representation.checked_element_count().is_none() {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::InvalidShape,
                    "provider.shape.overflow",
                );
            }
            if representation.element_type != expected {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedElementType,
                    "provider.element_type.unsupported",
                );
            }
            if representation.storage != ProviderStorage::DenseReal {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedStorage,
                    "provider.storage.unsupported",
                );
            }
            if representation.layout != ProviderLayout::ColumnMajorContiguous {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnsupportedLayout,
                    "provider.layout.unsupported",
                );
            }
        }
        let valid_direction = match self.family {
            ProviderOperationFamily::Upload => self.inputs.iter().all(|representation| {
                matches!(
                    representation.residency,
                    ProviderResidency::Host | ProviderResidency::Mirrored
                )
            }),
            ProviderOperationFamily::Download => self.inputs.iter().all(|representation| {
                matches!(
                    representation.residency,
                    ProviderResidency::Device | ProviderResidency::Mirrored
                )
            }),
            _ => false,
        };
        if !valid_direction {
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnsupportedLayout,
                "provider.residency.direction_mismatch",
            );
        }
        let output_bytes = self.outputs.iter().try_fold(0_u64, |total, output| {
            total.checked_add(output.checked_byte_len()?)
        });
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes,
            dispatches: Some(0),
        })
    }
}
