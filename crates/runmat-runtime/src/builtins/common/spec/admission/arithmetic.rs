use runmat_builtins::BuiltinBindingIdentity;
use runmat_types::NumericClass;

pub use runmat_accelerate_api::PowerScalarControl as FusionScalarControl;

/// Operation-owned arithmetic authority, distinct from real-result domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionArithmeticEvidence {
    RealPower(FusionPowerArithmeticReceipt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionArithmeticKind {
    RealPower,
}

impl FusionArithmeticEvidence {
    pub fn kind(self) -> FusionArithmeticKind {
        match self {
            Self::RealPower(_) => FusionArithmeticKind::RealPower,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct FusionPowerArithmeticReceipt {
    pub binding: BuiltinBindingIdentity,
    pub revision: u32,
    pub base_id: u32,
    pub exponent_id: u32,
    pub base_class: NumericClass,
    pub exponent_class: NumericClass,
    pub exponent_scalar: Option<FusionScalarControl>,
    pub mode: runmat_accelerate_api::RealPowerArithmeticMode,
    pub output_class: NumericClass,
}
