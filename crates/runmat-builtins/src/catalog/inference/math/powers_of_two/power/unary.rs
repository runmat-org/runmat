use runmat_types::{
    DynamicReason, NumericClass, NumericDomain, StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::super::{
    numeric_kind,
    support::facts::{materialize, preserve_shape_as_dynamic},
};
use super::input;

pub(super) fn infer(
    exponent: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    if matches!(exponent.storage, StorageFact::Sparse) {
        diagnostics.push(input::argument_error(
            "pow2 does not accept sparse input",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }
    let Some(numeric) = input::numeric_input(exponent) else {
        if matches!(exponent.kind, ValueKindFact::Unknown) {
            let mut output = exponent.clone();
            preserve_shape_as_dynamic(&mut output);
            return output;
        }
        diagnostics.push(input::argument_error(
            "pow2 requires numeric, logical, or character input",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    };
    if input::is_integer(numeric.class) && numeric.domain == NumericDomain::Complex {
        diagnostics.push(input::argument_error(
            "pow2 does not accept complex fixed-width integer input",
            0,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let class = if input::is_integer(numeric.class)
        || matches!(
            exponent.kind,
            ValueKindFact::Logical | ValueKindFact::Character
        ) {
        NumericClass::Double
    } else {
        numeric.class
    };
    let mut output = exponent.clone();
    output.kind = numeric_kind(class, numeric.domain);
    materialize(&mut output);
    output
}
