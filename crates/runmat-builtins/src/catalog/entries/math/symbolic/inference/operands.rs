use super::super::scalar_contract::SymbolicScalarOperand;
use runmat_types::*;
pub(super) fn permits(value: &ValueFact, role: SymbolicScalarOperand) -> bool {
    use SymbolicScalarOperand::*;
    if matches!(
        value.residency,
        ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
    ) || value.storage == StorageFact::Sparse
    {
        return false;
    }
    let scalar = value.shape.element_count().is_none_or(|n| n == 1);
    match value.kind {
        ValueKindFact::Unknown => {
            scalar
                || matches!(
                    role,
                    DecimalInput | Variable | VariableOrPoint | Direction | Digits
                ) && value.shape.known_dims().is_none_or(|dims| {
                    dims.first().is_none_or(|d| d.is_none_or(|d| d == 1))
                        && dims.iter().skip(2).all(|d| d.is_none_or(|d| d == 1))
                })
        }
        ValueKindFact::Symbolic => {
            !matches!(role, Direction | Digits | Precision)
                && value.storage != StorageFact::Dense
                && scalar
        }
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) => !matches!(role, Variable | Direction) && scalar,
        ValueKindFact::Logical => {
            !matches!(role, Variable | Direction | Digits)
                && value.storage != StorageFact::Dense
                && scalar
        }
        ValueKindFact::String => {
            matches!(
                role,
                DecimalInput | Variable | VariableOrPoint | Direction | Digits
            ) && scalar
        }
        ValueKindFact::Character => {
            matches!(
                role,
                DecimalInput | Variable | VariableOrPoint | Direction | Digits
            ) && value.shape.known_dims().is_none_or(|dims| {
                dims.first().is_none_or(|d| d.is_none_or(|d| d == 1))
                    && dims.iter().skip(2).all(|d| d.is_none_or(|d| d == 1))
            })
        }
        _ => false,
    }
}
