use serde::{Deserialize, Serialize};

use crate::ProviderPrecision;

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
    pub elements: Option<u64>,
    pub flops: Option<u64>,
    pub batch: Option<u64>,
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
