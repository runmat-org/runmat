use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueFact, ValueKindFact};

pub(super) fn numeric_input(input: &ValueFact) -> Option<NumericFact> {
    match input.kind {
        ValueKindFact::Numeric(numeric) => Some(numeric),
        ValueKindFact::Logical | ValueKindFact::Character => Some(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        _ => None,
    }
}

pub(super) const fn is_integer(class: NumericClass) -> bool {
    !matches!(class, NumericClass::Double | NumericClass::Single)
}

pub(super) fn argument_error(
    message: &'static str,
    argument: usize,
) -> runmat_types::InferenceDiagnostic {
    super::super::super::super::argument_error("RM-CATALOG-POW2-INPUT", message, argument)
}
