use super::math_fact_transforms::{materialize_output, preserve_shape_on_dynamic_input};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, PiScaledTrigonometricFunction, TrigonometricFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: TrigonometricFunction,
) -> CallInference {
    let name = match function {
        TrigonometricFunction::Sin => "sin",
        TrigonometricFunction::Cos => "cos",
        TrigonometricFunction::Tan => "tan",
    };
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-TRIGONOMETRIC-ARITY",
            format!("{name} requires an input value"),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(numeric)
            if numeric.domain == NumericDomain::Complex
                && !matches!(numeric.class, NumericClass::Double | NumericClass::Single) =>
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-COMPLEX-INTEGER",
                format!("{name} does not accept complex fixed-width integer input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
        ValueKindFact::Numeric(numeric) => {
            if !matches!(numeric.class, NumericClass::Double | NumericClass::Single) {
                numeric.class = NumericClass::Double;
            }
            if matches!(output.storage, StorageFact::Sparse) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TRIGONOMETRIC-SPARSE",
                    format!("{name} does not currently accept sparse input"),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            } else {
                materialize_output(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => preserve_shape_on_dynamic_input(&mut output),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-INPUT",
                format!("{name} requires numeric, logical, character, or symbolic input"),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    match request.arguments.as_slice() {
        [_] => {
            if matches!(output.residency, ResidencyFact::Device { .. }) {
                output.residency = ResidencyFact::Unknown;
            }
        }
        [_, keyword, prototype] => {
            let keyword_literal =
                request
                    .literals
                    .literal_args
                    .get(1)
                    .and_then(|literal| match literal {
                        LiteralValue::String(value)
                        | LiteralValue::Character(value)
                        | LiteralValue::Keyword(value) => Some(value.as_str()),
                        _ => None,
                    });
            match keyword_literal {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    apply_trigonometric_prototype(&mut output, prototype, name, &mut diagnostics);
                }
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-TRIGONOMETRIC-LIKE",
                    format!("{name} accepts only the \"like\" option"),
                    1,
                )),
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    output.residency = ResidencyFact::Unknown;
                }
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-TRIGONOMETRIC-LIKE",
                    format!("{name} requires \"like\" as its second input"),
                    1,
                )),
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-ARITY",
                format!(
                    "{name} accepts one input or an input followed by \"like\" and a prototype"
                ),
                request.arguments.len().saturating_sub(1),
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn infer_pi_scaled_trigonometric(
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
            materialize_output(&mut output);
            output.residency = output_residency.clone();
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize_output(&mut output);
            output.residency = output_residency.clone();
        }
        ValueKindFact::Unknown => {
            preserve_shape_on_dynamic_input(&mut output);
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

fn apply_trigonometric_prototype(
    output: &mut ValueFact,
    prototype: &ValueFact,
    name: &str,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    match &prototype.kind {
        ValueKindFact::Numeric(prototype_numeric) => {
            if let ValueKindFact::Numeric(output_numeric) = &mut output.kind {
                if prototype_numeric.domain == NumericDomain::Complex {
                    output_numeric.domain = NumericDomain::Complex;
                }
            }
            output.residency = prototype.residency.clone();
        }
        ValueKindFact::Logical | ValueKindFact::Unknown => {
            output.residency = prototype.residency.clone();
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-TRIGONOMETRIC-PROTOTYPE",
                format!("{name} requires a numeric or logical prototype"),
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}
