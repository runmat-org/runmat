use serde::{Deserialize, Serialize};

use crate::{GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderPrecision};

#[cfg(test)]
#[path = "workload_tests.rs"]
mod workload_tests;

use super::{
    ProviderElementType, ProviderLayout, ProviderOperationFamily, ProviderRepresentation,
    ProviderResidency, ProviderStorage,
};

/// Admission bound for a transfer whose exact output shape depends on values
/// that have not yet been read. This never masquerades as an exact tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTransferOutputBound {
    pub element_type: ProviderElementType,
    pub storage: ProviderStorage,
    pub layout: ProviderLayout,
    pub residency: ProviderResidency,
    pub max_elements: u64,
}

impl ProviderTransferOutputBound {
    pub fn max_bytes(&self) -> Option<u64> {
        // Packed logical/short integer uploads vary by backend. Only native
        // floating payloads have a provider-independent physical width.
        let width = match self.element_type {
            ProviderElementType::F32 => 4,
            ProviderElementType::F64 => 8,
            _ => return None,
        };
        self.max_elements.max(1).checked_mul(width)
    }

    pub fn admits(&self, actual: &ProviderRepresentation) -> bool {
        self.element_type == actual.element_type
            && self.storage == actual.storage
            && self.layout == actual.layout
            && self.residency == actual.residency
            && actual
                .checked_element_count()
                .is_some_and(|count| count <= self.max_elements)
    }
}

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
    /// One session-issued GPU child-stream range for a typed random output.
    /// Ordinary host RNG calls never set this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_random: Option<crate::random::GpuRandomWorkload>,
    /// Native integer child-stream generation; absence never authorizes
    /// replacing an exact host rejection-sampling route with device modulo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_integer: Option<crate::random::GpuIntegerWorkload>,
    /// One acknowledged rejection batch, distinct from a complete fixed-work
    /// integer operation. Each batch has two authenticated API outputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_integer_batch: Option<crate::random::GpuIntegerBatchWorkload>,
    /// One selected caller-supplied-noise stochastic evolution. The workload
    /// carries semantic arithmetic identity, never an ambient RNG route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stochastic_evolution:
        Option<crate::stochastic_evolution::ProviderStochasticEvolutionWorkload>,
    /// Explicit physical ABI for the existing ordinary typed fusion owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_elementwise: Option<crate::ProviderFusedElementwiseWorkload>,
    /// One exact selected checked fused-reduction route. Absence leaves its
    /// physical execution inventory unknown, never a zero-cost default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_reduction: Option<super::ProviderTypedReductionWorkload>,
    /// Explicit real-power arithmetic mode. None retains the native legacy
    /// route; a mixed route is never inferred from the final result class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_power_mode: Option<crate::RealPowerArithmeticMode>,
    /// Original typed scalar exponent for a separately certified POWER route.
    /// Absence never authorizes specialization of a general mixed operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_power_scalar: Option<crate::PowerScalarWorkload>,
    /// Source-qualified staged image program. Its presence alone cannot
    /// establish physical provider work or numerical equivalence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_stages: Option<super::ProviderImageStageWorkload>,
    /// Exact admitted physical route; None leaves adaptation work unresolved.
    #[serde(default)]
    pub preparation_route: Option<super::ProviderPreparationRoute>,
    /// Exact admitted matrix/epilogue route. None is unknown, not a bare GEMM.
    #[serde(default)]
    pub matmul: Option<super::MatmulWorkload>,
    /// Resolved ordinary-sum route; absent for every other reduction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sum: Option<super::ProviderSumWorkload>,
    /// Resolved ordinary dimensional min/max route. The physical provider
    /// still produces values and indices for a values-only language call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extrema: Option<super::ProviderExtremaWorkload>,
    /// Physical convolution hook and public shape mode admitted by its owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convolution: Option<super::ProviderConvolutionWorkload>,
    /// Exact `fspecial` constructor and scalar controls; no value-bearing
    /// host payload is inferred from an output shape alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fspecial: Option<super::ProviderFspecialFilter>,
    /// Typed two-input image filtering hook controls and live backing facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imfilter: Option<super::ProviderImfilterWorkload>,
    /// Typed IIR execution route. Its absence cannot establish an IIR hook.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iir: Option<super::ProviderIirWorkload>,
    /// Maximum per-output transfer payloads when exact result dimensions are
    /// data-dependent. Only upload queries with empty exact inputs/outputs may
    /// use this; publication must later validate an exact upload query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_output_bounds: Option<Vec<ProviderTransferOutputBound>>,
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
    /// Peak simultaneously requested live bytes for the modeled route, not
    /// physical pool capacity or total device use. None is an unproven bound.
    pub transient_bytes: Option<u64>,
    /// Requested bytes retained by the published result, when its physical
    /// storage and ownership are proven by the provider's route.
    pub output_bytes: Option<u64>,
    /// Number of actual compute submissions for the complete modeled route;
    /// none means the physical dispatch inventory is unresolved.
    pub dispatches: Option<u32>,
}

/// Provider-owned physical evidence for one exact numeric host-to-device
/// upload used inside a selected operation's conditional continuation.
/// `provider_scratch_bytes` excludes the newly retained output but includes
/// completion instrumentation that coexists with it. Unknown implementations
/// return no evidence instead of fabricating zero staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderNumericUploadPhysicalEvidence {
    pub output_requested_bytes: u64,
    pub provider_scratch_bytes: runmat_execution::ResourceByteBound,
    pub completion_queue_slots: u32,
}

/// Exact native floating-cast stage for an already resident contiguous source.
/// A same-class alias has no fresh output; scalar POWER only requests genuine
/// F32/F64 changes. The eventual live source must still be authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderFloatingCastPhysicalEvidence {
    pub output_requested_bytes: u64,
    pub uniform_buffer_bytes: u64,
    pub dispatches: u32,
    pub queue_slots: u32,
}

/// A native numeric result produced by a selected provider operation may be
/// read back only after it exists. This is side-effect-free physical evidence
/// for the *future* contiguous output representation, not proof that a later
/// handle belongs to this provider. The caller must authenticate that handle
/// against the same representation before performing the transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProducedNumericReadbackEvidence {
    pub source_requested_bytes: u64,
    pub transfer_bytes: u64,
    pub host_workspace_bytes: runmat_execution::ResourceByteBound,
    pub provider_scratch_bytes: runmat_execution::ResourceByteBound,
    pub queue_slots: u32,
}

/// Physical one-result class-preserving host publication, including native
/// integer and logical storage. Logical payload bytes and physical provider
/// retention are distinct. Providers without an audited route return None.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderUploadPhysicalEvidence {
    pub output_requested_bytes: u64,
    /// Provider-created native host conversion/word-packing workspace beyond
    /// the caller's retained payload and operation-owned temporaries.
    pub host_workspace_bytes: runmat_execution::ResourceByteBound,
    pub provider_scratch_bytes: runmat_execution::ResourceByteBound,
    pub completion_queue_slots: u32,
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

impl std::fmt::Display for ProviderRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {:?}", self.code, self.detail)
    }
}
impl std::error::Error for ProviderRejection {}

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

fn numeric_download_element_type(element: NumericElementType) -> ProviderElementType {
    match element {
        NumericElementType::F32 => ProviderElementType::F32,
        NumericElementType::F64 => ProviderElementType::F64,
        NumericElementType::I8 => ProviderElementType::I8,
        NumericElementType::I16 => ProviderElementType::I16,
        NumericElementType::I32 => ProviderElementType::I32,
        NumericElementType::I64 => ProviderElementType::I64,
        NumericElementType::U8 => ProviderElementType::U8,
        NumericElementType::U16 => ProviderElementType::U16,
        NumericElementType::U32 => ProviderElementType::U32,
        NumericElementType::U64 => ProviderElementType::U64,
    }
}

fn numeric_download_role_matches(
    role: &ProviderRepresentation,
    source: &GpuTensorHandle,
    expected_element_type: ProviderElementType,
    expected_layout: ProviderLayout,
    expected_residency: ProviderResidency,
) -> bool {
    let ProviderRepresentation {
        element_type,
        storage,
        layout,
        shape,
        residency,
    } = role;
    *element_type == expected_element_type
        && *storage == ProviderStorage::DenseReal
        && *layout == expected_layout
        && *residency == expected_residency
        && shape.len() == source.shape().len()
        && shape
            .iter()
            .zip(source.shape())
            .all(|(&actual, &expected)| u64::try_from(expected).ok() == Some(actual))
}

impl ProviderFeasibilityQuery {
    const NUMERIC_DOWNLOAD_OPERATION: &'static str = "transfer.download";

    /// Construct the exact native real-numeric readback roles for one live
    /// source. Lazy views retain their strided input representation; the host
    /// payload is contiguous in logical shape. Complex/unsupported physical
    /// storage has no silently fabricated real transfer query.
    pub fn exact_numeric_download(source: &GpuTensorHandle) -> Option<Self> {
        if source.physical().storage() != GpuTensorStorage::Real {
            return None;
        }
        let element_type = numeric_download_element_type(source.physical().element_type());
        // This exact query is also used by finite host-readback admission.
        // Try-collecting through `Option` erases the iterator's lower size
        // hint and may request a minimum Vec capacity larger than the rank.
        // Reserve the authenticated rank before the first allocation.
        let mut shape = Vec::new();
        shape.try_reserve_exact(source.shape().len()).ok()?;
        for &extent in source.shape() {
            shape.push(u64::try_from(extent).ok()?);
        }
        let input = ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: if crate::handle_is_transposed(source) {
                ProviderLayout::Strided
            } else {
                ProviderLayout::ColumnMajorContiguous
            },
            shape: shape.clone(),
            residency: ProviderResidency::Device,
        };
        let output = ProviderRepresentation {
            element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape,
            residency: ProviderResidency::Host,
        };
        Some(Self {
            operation: ProviderOperationIdentity::new(Self::NUMERIC_DOWNLOAD_OPERATION),
            family: ProviderOperationFamily::Download,
            inputs: vec![input],
            outputs: vec![output],
            workload: ProviderWorkload {
                requested_outputs: Some(1),
                ..ProviderWorkload::default()
            },
        })
    }

    /// Canonical exact native readback roles. A provider may support a lazy
    /// source, but it may not reinterpret class, storage, shape, count, or
    /// host-output layout to turn another transfer query into that route.
    pub fn matches_numeric_download(&self, source: &GpuTensorHandle) -> bool {
        // Validation must not allocate a second query inside the caller's
        // already admitted query footprint. Compare borrowed roles instead.
        let Self {
            operation,
            family,
            inputs,
            outputs,
            workload,
        } = self;
        let ([input], [output]) = (inputs.as_slice(), outputs.as_slice()) else {
            return false;
        };
        if source.physical().storage() != GpuTensorStorage::Real
            || operation.0 != Self::NUMERIC_DOWNLOAD_OPERATION
            || *family != ProviderOperationFamily::Download
            || *workload
                != (ProviderWorkload {
                    requested_outputs: Some(1),
                    ..ProviderWorkload::default()
                })
        {
            return false;
        }
        let element_type = numeric_download_element_type(source.physical().element_type());
        numeric_download_role_matches(
            input,
            source,
            element_type,
            if crate::handle_is_transposed(source) {
                ProviderLayout::Strided
            } else {
                ProviderLayout::ColumnMajorContiguous
            },
            ProviderResidency::Device,
        ) && numeric_download_role_matches(
            output,
            source,
            element_type,
            ProviderLayout::ColumnMajorContiguous,
            ProviderResidency::Host,
        )
    }

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
        if let Some(bounds) = &self.workload.transfer_output_bounds {
            if self.family != ProviderOperationFamily::Upload
                || !self.inputs.is_empty()
                || !self.outputs.is_empty()
                || bounds.is_empty()
                || self.workload.requested_outputs != u32::try_from(bounds.len()).ok()
            {
                return ProviderFeasibility::rejected(
                    ProviderRejectionCode::InvalidShape,
                    "provider.transfer.bound_contract",
                );
            }
            let expected = ProviderElementType::from(precision);
            for bound in bounds {
                if bound.element_type != expected
                    || bound.storage != ProviderStorage::DenseReal
                    || bound.layout != ProviderLayout::ColumnMajorContiguous
                    || bound.residency != ProviderResidency::Device
                {
                    return ProviderFeasibility::rejected(
                        ProviderRejectionCode::UnsupportedElementType,
                        "provider.transfer.bound_class",
                    );
                }
                if bound.max_bytes().is_none() {
                    return ProviderFeasibility::rejected(
                        ProviderRejectionCode::ResourceLimit,
                        "provider.transfer.bound_overflow",
                    );
                }
            }
            // A provider may choose a wider physical upload representation.
            // Only an audited provider override can turn a logical bound into
            // a truthful physical allocation estimate.
            return ProviderFeasibility::rejected(
                ProviderRejectionCode::UnknownResourceEstimate,
                "provider.transfer.bound_physical_unknown",
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
