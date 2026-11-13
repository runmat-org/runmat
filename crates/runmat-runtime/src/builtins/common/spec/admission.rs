use super::{FusionError, ScalarType};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_types::{ValueFact, ValueKindFact};
use runmat_value::Value;
use std::{future::Future, pin::Pin};

mod sign;
pub use sign::{scan_represented_sign, FusionSignEvidence};

#[path = "admission/arithmetic.rs"]
mod arithmetic;
pub use arithmetic::{
    FusionArithmeticEvidence, FusionArithmeticKind, FusionPowerArithmeticReceipt,
    FusionScalarControl,
};

/// Exact values originate from graph inputs/constants or a request-local,
/// selected pure scalar producer. An unevaluated intermediate never borrows
/// the value of an ancestor as a semantic witness.
pub struct FusionAdmissionContext<'a> {
    pub operands: &'a [Option<&'a Value>],
    pub facts: &'a [ValueFact],
    operand_ids: &'a [u32],
    /// Request-local evidence aligned to the exact source operand ValueIds.
    operand_evidence: &'a [Option<FusionOperandDomainEvidence>],
    pub output_fact: Option<&'a ValueFact>,
    /// Record before executing work so cancellation and failure are observable.
    pub work: &'a mut FusionAdmissionWork,
}

impl<'a> FusionAdmissionContext<'a> {
    /// Bind every evidence slot to the selected operation's actual operand
    /// ValueId before invoking an operation owner. A correctly positioned but
    /// differently owned receipt is invalid, not evidence for that slot.
    pub fn new(
        operand_ids: &'a [u32],
        operands: &'a [Option<&'a Value>],
        facts: &'a [ValueFact],
        operand_evidence: &'a [Option<FusionOperandDomainEvidence>],
        work: &'a mut FusionAdmissionWork,
    ) -> Option<Self> {
        if operand_ids.len() != operands.len()
            || operand_ids.len() != facts.len()
            || operand_ids.len() != operand_evidence.len()
            || operand_evidence
                .iter()
                .zip(operand_ids.iter().zip(facts))
                .any(|(evidence, (id, fact))| {
                    evidence.as_ref().is_some_and(|evidence| {
                        evidence.value_id != *id
                            || evidence.sign.is_some_and(|sign| {
                                fact.numeric().is_none_or(|numeric| {
                                    numeric.domain != runmat_types::NumericDomain::Real
                                        || numeric.class != sign.class()
                                })
                            })
                    })
                })
        {
            return None;
        }
        Some(Self {
            operands,
            facts,
            operand_ids,
            operand_evidence,
            output_fact: None,
            work,
        })
    }

    pub fn with_output_fact(mut self, output_fact: &'a ValueFact) -> Self {
        self.output_fact = Some(output_fact);
        self
    }

    pub fn evidence_at(&self, index: usize) -> Option<FusionOperandDomainEvidence> {
        self.operand_evidence
            .get(index)
            .and_then(|evidence| *evidence)
            .filter(|evidence| self.operand_ids.get(index) == Some(&evidence.value_id))
    }

    pub fn operand_id(&self, index: usize) -> Option<u32> {
        self.operand_ids.get(index).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionOperandDomainEvidence {
    pub value_id: u32,
    pub numerical_domain: runmat_accelerate_api::NumericalDomainEvidence,
    pub sign: Option<FusionSignEvidence>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub enum FusionAdmissionWork {
    #[default]
    None,
    Host {
        elements: u64,
    },
    ProviderProbe {
        binding: BuiltinBindingIdentity,
        /// Exact operand precision; not a promise about provider temporaries.
        precision: runmat_accelerate_api::ProviderPrecision,
        provider_precision: runmat_accelerate_api::ProviderPrecision,
        inputs: Vec<runmat_accelerate_api::ProviderRepresentation>,
    },
    /// One request-local proof may inspect several host and resident leaves.
    /// Stages are appended before work starts, including failed or abandoned
    /// provider hooks; one observation measures the complete proof wall time.
    DomainProof {
        binding: BuiltinBindingIdentity,
        provider_precision: runmat_accelerate_api::ProviderPrecision,
        stages: Vec<FusionProofStage>,
    },
    SignProof {
        operation: runmat_types::OperatorKind,
        host_elements: u64,
        propagated_operands: u64,
    },
}

impl FusionAdmissionWork {
    /// A selected provider hook or scalar readback was entered during proof.
    /// Later failure cannot certify pre-effect numerical replay.
    pub fn provider_effect_observed(&self) -> bool {
        match self {
            Self::ProviderProbe { .. } => true,
            Self::DomainProof { stages, .. } => stages.iter().any(|stage| {
                matches!(
                    stage,
                    FusionProofStage::Provider { .. } | FusionProofStage::ScalarReadback { .. }
                )
            }),
            Self::None | Self::Host { .. } | Self::SignProof { .. } => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum FusionProofStage {
    GraphIndex {
        operations: u64,
        leaves: u64,
    },
    GraphWalk {
        examined: u64,
    },
    RangePropagation {
        operands: u64,
    },
    ControlMaterialization {
        values: u64,
        shape_dimensions: u64,
        string_bytes: u64,
        char_elements: u64,
    },
    HostScan {
        elements: u64,
    },
    Provider {
        operation: runmat_accelerate_api::ProviderOperationIdentity,
        input: runmat_accelerate_api::ProviderRepresentation,
    },
    ScalarReadback {
        elements: u64,
    },
}

pub struct FusionAdmissionEvidence {
    pub output_kind: ValueKindFact,
    /// Optional operation-owned ordinary publication carrier, independent of
    /// sign. Absence preserves conservative inferred storage.
    pub output_storage: Option<runmat_types::StorageFact>,
    pub numerical_domain: runmat_accelerate_api::NumericalDomainEvidence,
    pub arithmetic: Option<FusionArithmeticEvidence>,
    pub sign: Option<FusionSignEvidence>,
}

pub type FusionAdmissionFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<Option<FusionAdmissionEvidence>>> + 'a>>;
pub type FusionAdmissionFunction =
    for<'a> fn(FusionAdmissionContext<'a>) -> FusionAdmissionFuture<'a>;
#[path = "admission/preparation.rs"]
mod preparation;
pub use preparation::{FusionAdmission, FusionAdmissionPreparationQuery, FusionAdmissionResources};
pub type FusionStaticAdmission =
    for<'a> fn(
        &FusionAdmissionContext<'a>,
    ) -> anyhow::Result<Option<(FusionAdmissionEvidence, FusionAdmissionWork)>>;

/// A typed helper requirement, not an operation-name dispatch or independent
/// numerical implementation. Backends and fusion share the helper's source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum FusionMathHelper {
    RealPower,
    Round,
    FloatingRemainder,
}

impl FusionMathHelper {
    pub fn entrypoint(self, precision: ScalarType) -> Result<&'static str, FusionError> {
        match (self, precision) {
            (Self::RealPower, ScalarType::F32) => Ok("rm_power_f32"),
            (Self::RealPower, ScalarType::F64) => Ok("rm_power_f64"),
            (Self::Round, ScalarType::F32) => Ok("rm_round_f32"),
            (Self::Round, ScalarType::F64) => Ok("rm_round_f64"),
            (Self::FloatingRemainder, ScalarType::F32) => Ok("rm_remainder_f32"),
            (Self::FloatingRemainder, ScalarType::F64) => Ok("rm_remainder_f64"),
            (_, precision) => Err(FusionError::UnsupportedPrecision(precision)),
        }
    }
}

/// Optional explicit authority for data-dependent fusion semantics. The
/// revision must change when admission behavior changes independently of its
/// emitted expression. Older declarations remain provenance-resolved and
/// cannot claim this stronger, operation-owned admission protocol.
pub struct FusionSemanticDeclaration {
    pub binding: BuiltinBindingIdentity,
    pub revision: u32,
    pub admit: Option<FusionAdmission>,
    /// Pure, non-suspending subset of the same operation-owned admission.
    pub admit_static: Option<FusionStaticAdmission>,
    /// A selected operation-owned real-domain requirement and its catalog
    /// diagnostic. The resolver proves this exact predicate before shader
    /// generation; probe failures use the declared descriptor identifier.
    pub required_real_unary: Option<(
        runmat_types::RealUnaryPredicate,
        &'static runmat_builtins::BuiltinErrorDescriptor,
    )>,
    pub required_arithmetic: Option<FusionArithmeticKind>,
    pub helpers: &'static [FusionMathHelper],
    /// Optional closed scalar operation selected by this exact binding.
    /// WGSL-only declarations remain on their established route until their
    /// numerical and domain policies have an independently checked lowering.
    pub scalar_operation: Option<runmat_types::ScalarOperation>,
}

/// Source-owned shape of the first finite-preparation accounting cohort.
/// This is a typed declaration property, not a dispatch by builtin spelling.
/// Admissions that can scan, suspend, or retain control work require their
/// own proven bound and must not inherit this pure scalar class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusionPreparationClass {
    PureRealUnary,
    PureRealBinary,
}

impl FusionPreparationClass {
    pub const fn arity(self) -> usize {
        match self {
            Self::PureRealUnary => 1,
            Self::PureRealBinary => 2,
        }
    }
}

impl FusionSemanticDeclaration {
    pub fn finite_preparation_class(&self) -> Option<FusionPreparationClass> {
        if self.admit.is_some()
            || self.admit_static.is_some()
            || self.required_real_unary.is_some()
            || self.required_arithmetic.is_some()
        {
            return None;
        }
        match self.scalar_operation?.arity()? {
            1 => Some(FusionPreparationClass::PureRealUnary),
            2 => Some(FusionPreparationClass::PureRealBinary),
            _ => None,
        }
    }

    /// Only total real-domain, class-preserving unary instructions use this
    /// convenience constructor. Promoting operations retain explicit domain
    /// admission declarations instead of inheriting this absence of checks.
    pub const fn total_real_unary_math(
        binding: BuiltinBindingIdentity,
        operation: runmat_types::ScalarUnaryMath,
    ) -> Self {
        Self {
            binding,
            revision: 1,
            admit: None,
            admit_static: None,
            required_real_unary: None,
            required_arithmetic: None,
            helpers: &[],
            scalar_operation: Some(runmat_types::ScalarOperation::UnaryMath(operation)),
        }
    }
}
