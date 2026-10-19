use runmat_types::{
    broadcast_shape, DynamicReason, NumericClass, NumericDomain, ResidencyFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

use super::super::super::super::{numeric_kind, support::facts::materialize};
use super::input;

pub(super) fn infer(
    significand: &ValueFact,
    exponent: &ValueFact,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ValueFact {
    for (index, value) in [significand, exponent].into_iter().enumerate() {
        if matches!(value.storage, StorageFact::Sparse) {
            diagnostics.push(input::argument_error(
                "pow2 does not accept sparse input",
                index,
            ));
            return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    let Some(left) = classify(significand, 0, diagnostics) else {
        return unknown_input(significand);
    };
    let Some(right) = classify(exponent, 1, diagnostics) else {
        return unknown_input(exponent);
    };
    if (input::is_integer(left.class) && left.domain == NumericDomain::Complex)
        || (input::is_integer(right.class) && right.domain == NumericDomain::Complex)
    {
        diagnostics.push(input::argument_error(
            "pow2 does not accept complex fixed-width integer input",
            usize::from(!(input::is_integer(left.class) && left.domain == NumericDomain::Complex)),
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let shape = match broadcast_shape(&significand.shape, &exponent.shape) {
        Ok(shape) => shape,
        Err(diagnostic) => {
            diagnostics.push(diagnostic);
            ShapeFact::Unknown
        }
    };
    let class = if left.class == NumericClass::Single || right.class == NumericClass::Single {
        NumericClass::Single
    } else {
        NumericClass::Double
    };
    let domain = if left.domain == NumericDomain::Complex || right.domain == NumericDomain::Complex
    {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = numeric_kind(class, domain);
    output.shape = shape;
    materialize(&mut output);
    output.residency = direct_residency(significand, exponent, left.class, right.class);
    output
}

fn unknown_input(input: &ValueFact) -> ValueFact {
    let reason = if matches!(input.kind, ValueKindFact::Unknown) {
        DynamicReason::RuntimeValue
    } else {
        DynamicReason::UnsupportedRepresentation
    };
    ValueFact::unknown(reason)
}

fn classify(
    input_fact: &ValueFact,
    argument: usize,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> Option<runmat_types::NumericFact> {
    let numeric = input::numeric_input(input_fact);
    if numeric.is_none() && !matches!(input_fact.kind, ValueKindFact::Unknown) {
        diagnostics.push(input::argument_error(
            "pow2 requires numeric, logical, or character input",
            argument,
        ));
    }
    numeric
}

fn direct_residency(
    left: &ValueFact,
    right: &ValueFact,
    left_class: NumericClass,
    right_class: NumericClass,
) -> ResidencyFact {
    match (&left.residency, &right.residency) {
        (ResidencyFact::Host, _) | (_, ResidencyFact::Host) => ResidencyFact::Host,
        (
            ResidencyFact::Device {
                provider: left_owner,
            },
            ResidencyFact::Device {
                provider: right_owner,
            },
        ) if left_owner == right_owner
            && left.shape == right.shape
            && left_class == right_class
            && matches!(left_class, NumericClass::Double | NumericClass::Single) =>
        {
            left.residency.clone()
        }
        (ResidencyFact::Device { .. }, ResidencyFact::Device { .. }) => ResidencyFact::Host,
        _ => ResidencyFact::Unknown,
    }
}
