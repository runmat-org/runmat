//! Roles for already-admitted calls, never another argument or option parser.
use serde::Serialize;

mod validation;
pub use validation::validate_builtin_preparation;
#[cfg(test)]
mod tests;

/// Stable within a canonical identity. IDs are not signature array offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct BuiltinPreparationFormId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinOperandPreparation {
    /// The implementation owns preparation, including any provider execution.
    /// This does NOT mean host-only, nor prove migration of an old auto route.
    RuntimeOwned,
    /// The domain selects a form only after its existing admission succeeds.
    Forms(&'static [BuiltinPreparationForm]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPreparationForm {
    pub id: BuiltinPreparationFormId,
    pub binding_variant: &'static str,
    /// Canonical descriptor signatures to which the role declaration applies.
    pub signatures: &'static [usize],
    /// Original argument positions, strictly increasing. Every other argument
    /// is an unchanged control retained by the admitted domain implementation.
    pub data_arguments: &'static [usize],
    pub operation: BuiltinPreparationOperation,
    pub provider_requirement: BuiltinPreparationProviderRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPreparationOperation {
    Unary,
    /// The admitted form retains its physical route in the runtime domain.
    /// The shared service still receives the declared data operand so a host
    /// port can observe execution, but native auto must not model or promote it.
    RuntimeOwned,
    /// Plain, shape-preserving floating cast; no output prototype adaptation.
    FloatingCast(runmat_types::NumericClass),
    /// A real unary route whose numerical domain is admitted by its owner.
    RealUnary(runmat_types::RealUnaryPredicate),
    /// Digit count and decimal/significant mode remain typed request controls.
    RoundDigits,
    FloatingRemainder(runmat_types::FloatingRemainderKind),
    ElementwiseBinary,
    /// An admitted broadcast operation consuming every declared data operand.
    ElementwiseNary {
        data_arity: usize,
    },
    /// Real floating operands whose result domain depends on paired values.
    RealPower,
    MatrixMultiply,
    Transpose,
    Reduction(BuiltinPreparationReduction),
    Sink(BuiltinSinkPreparation),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPreparationReduction {
    Sum,
    Mean,
    Min,
    Max,
    All,
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinSinkPreparation {
    /// Retain residency; terminate pending fusion use of these data operands.
    Barrier,
    /// Materialize declared data operands after admission, not option controls.
    Gather,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPreparationProviderRequirement {
    /// Existing transfer/class admission remains authoritative.
    InputSupported,
    /// The operation produces/requires double storage even for single inputs.
    F64,
}

impl BuiltinPreparationOperation {
    pub const fn data_arity(self) -> Option<usize> {
        match self {
            Self::Unary
            | Self::RuntimeOwned
            | Self::FloatingCast(_)
            | Self::RealUnary(_)
            | Self::RoundDigits
            | Self::Transpose
            | Self::Reduction(_) => Some(1),
            Self::ElementwiseBinary
            | Self::RealPower
            | Self::FloatingRemainder(_)
            | Self::MatrixMultiply => Some(2),
            Self::ElementwiseNary { data_arity } => Some(data_arity),
            Self::Sink(_) => None,
        }
    }
}
