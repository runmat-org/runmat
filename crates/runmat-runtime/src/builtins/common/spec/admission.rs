use super::{FusionError, ScalarType};
use runmat_builtins::BuiltinBindingIdentity;
use runmat_types::{ValueFact, ValueKindFact};
use runmat_value::Value;
use std::{future::Future, pin::Pin};

/// Exact values are available only for graph inputs/constants. An unevaluated
/// intermediate never borrows the value of an ancestor as a semantic witness.
pub struct FusionAdmissionContext<'a> {
    pub operands: &'a [Option<&'a Value>],
    pub facts: &'a [ValueFact],
    /// Record before executing work so cancellation and failure are observable.
    pub work: &'a mut FusionAdmissionWork,
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
}

pub struct FusionAdmissionEvidence {
    pub output_kind: ValueKindFact,
    pub numerical_domain: runmat_accelerate_api::NumericalDomainEvidence,
}

pub type FusionAdmissionFuture<'a> =
    Pin<Box<dyn Future<Output = anyhow::Result<Option<FusionAdmissionEvidence>>> + 'a>>;
pub type FusionAdmission = for<'a> fn(FusionAdmissionContext<'a>) -> FusionAdmissionFuture<'a>;

/// A typed helper requirement, not an operation-name dispatch or independent
/// numerical implementation. Backends and fusion share the helper's source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum FusionMathHelper {
    RealPower,
    Round,
}

impl FusionMathHelper {
    pub fn entrypoint(self, precision: ScalarType) -> Result<&'static str, FusionError> {
        match (self, precision) {
            (Self::RealPower, ScalarType::F32) => Ok("rm_power_f32"),
            (Self::RealPower, ScalarType::F64) => Ok("rm_power_f64"),
            (Self::Round, ScalarType::F32) => Ok("rm_round_f32"),
            (Self::Round, ScalarType::F64) => Ok("rm_round_f64"),
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
    pub helpers: &'static [FusionMathHelper],
}
