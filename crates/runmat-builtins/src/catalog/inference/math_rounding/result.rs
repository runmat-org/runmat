use super::super::numeric_kind;
use super::super::support::facts::materialize_floating_result;
use runmat_types::{
    NumericClass, NumericDomain, ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn materialize_rounding_result(output: &mut ValueFact) {
    if output.residency != ResidencyFact::Host {
        // Placement facts do not distinguish explicit device intent from an
        // automatically resident value whose structured fallback stays host.
        output.residency = ResidencyFact::Unknown;
    }
    if output.numeric().is_some_and(|numeric| {
        numeric.domain == NumericDomain::Real
            && matches!(numeric.class, NumericClass::Double | NumericClass::Single)
    }) && output.storage != StorageFact::Scalar
    {
        // Eligible dense floating operands can be automatically prepared.
        // Static type/shape facts do not choose the executing provider.
        output.residency = ResidencyFact::Unknown;
    }
    let character = output.kind == ValueKindFact::Character;
    let complex_storage = output
        .numeric()
        .filter(|numeric| numeric.domain == NumericDomain::Complex)
        .map(|_| output.storage);
    if matches!(
        output.kind,
        ValueKindFact::Logical | ValueKindFact::Character
    ) {
        output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
    }
    materialize_floating_result(output, false);
    if output.residency == ResidencyFact::Host {
        if character {
            // Character conversion always constructs a Tensor, even for 1x1.
            output.storage = StorageFact::Dense;
        } else if let Some(storage) = complex_storage {
            // Unlike real double tensors, complex rounding does not collapse
            // a singleton tensor into the scalar Complex carrier.
            output.storage = match storage {
                StorageFact::Scalar => StorageFact::Scalar,
                StorageFact::Dense => StorageFact::Dense,
                _ => StorageFact::Unknown,
            };
        }
    }
}
