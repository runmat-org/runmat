use super::{argument_error, finish_fixed, materialize, numeric_kind, preserve_shape_as_dynamic};
use crate::{BuiltinCatalogEntry, PiScaledTrigonometricFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(in crate::catalog::inference) fn infer_pi_scaled_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: PiScaledTrigonometricFunction,
) -> CallInference {
    let name = match function {
        PiScaledTrigonometricFunction::Sin => "sinpi",
        PiScaledTrigonometricFunction::Cos => "cospi",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-PI-TRIGONOMETRIC-ARITY",
            format!("{name} requires exactly one input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PI-TRIGONOMETRIC-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-PI-TRIGONOMETRIC-SPARSE",
            format!("{name} does not currently accept sparse input"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let output_residency = match function {
        PiScaledTrigonometricFunction::Sin => ResidencyFact::Host,
        PiScaledTrigonometricFunction::Cos => input.residency.clone(),
    };

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-PI-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            materialize(&mut output);
            output.residency = output_residency.clone();
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(&mut output);
            output.residency = output_residency.clone();
        }
        ValueKindFact::Unknown => {
            preserve_shape_as_dynamic(&mut output);
            output.residency = output_residency;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-PI-TRIGONOMETRIC-INPUT",
                format!("{name} requires numeric, logical, or character input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}
