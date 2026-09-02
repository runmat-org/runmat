use super::math_fact_transforms::{materialize_output, preserve_shape_on_dynamic_input};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, DegreeTrigonometricFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ResidencyFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_degree_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: DegreeTrigonometricFunction,
) -> CallInference {
    let (name, accepts_character, preserves_residency) = match function {
        DegreeTrigonometricFunction::Sin => ("sind", false, false),
        DegreeTrigonometricFunction::Cos => ("cosd", true, true),
        DegreeTrigonometricFunction::Tan => ("tand", true, true),
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-DEGREE-TRIGONOMETRIC-ARITY",
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
            "RM-CATALOG-DEGREE-TRIGONOMETRIC-ARITY",
            format!("{name} accepts exactly one input"),
            1,
        ));
    }
    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-DEGREE-TRIGONOMETRIC-SPARSE",
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

    let residency = if preserves_residency {
        input.residency.clone()
    } else {
        ResidencyFact::Host
    };
    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-DEGREE-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            materialize_output(&mut output);
            output.residency = residency.clone();
        }
        ValueKindFact::Logical => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = residency.clone();
        }
        ValueKindFact::Character if accepts_character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = residency.clone();
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
            output.residency = residency;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-DEGREE-TRIGONOMETRIC-INPUT",
                format!("{name} requires a supported numeric input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}
