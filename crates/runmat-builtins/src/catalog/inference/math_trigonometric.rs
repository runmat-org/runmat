use super::support::facts::{materialize, preserve_shape_as_dynamic};
use super::{argument_error, finish_fixed, numeric_kind};
use crate::{BuiltinCatalogEntry, TrigonometricFunction};
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericClass, NumericDomain,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

mod prototype;
mod pi_scaled;

pub(super) use pi_scaled::infer_pi_scaled_trigonometric;
use prototype::apply_trigonometric_prototype;

pub(super) fn infer_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: TrigonometricFunction,
) -> CallInference {
    let (output, diagnostics) = trigonometric_fact(request, function);
    finish_fixed(entry, request, output, diagnostics)
}

pub(in crate::catalog::inference) fn infer_fusion_numeric_fact(
    request: &CallRequest,
    function: TrigonometricFunction,
) -> Result<ValueFact, Vec<runmat_types::InferenceDiagnostic>> {
    let [input] = request.arguments.as_slice() else {
        return Err(Vec::new());
    };
    let admitted = match input.kind {
        ValueKindFact::Numeric(numeric) => {
            input.storage != StorageFact::Sparse
                && (numeric.domain != NumericDomain::Complex
                    || matches!(numeric.class, NumericClass::Double | NumericClass::Single))
        }
        ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::Symbolic
        | ValueKindFact::Unknown => true,
        _ => false,
    };
    if !admitted {
        return Err(Vec::new());
    }
    let (fact, diagnostics) = trigonometric_fact(request, function);
    if diagnostics.is_empty() {
        Ok(fact)
    } else {
        Err(diagnostics)
    }
}

fn trigonometric_fact(
    request: &CallRequest,
    function: TrigonometricFunction,
) -> (ValueFact, Vec<runmat_types::InferenceDiagnostic>) {
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
        return (ValueFact::unknown(DynamicReason::RuntimeValue), diagnostics);
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
                materialize(&mut output);
            }
        }
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materialize(&mut output);
        }
        ValueKindFact::Symbolic => {}
        ValueKindFact::Unknown => preserve_shape_as_dynamic(&mut output),
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
            if matches!(output.residency, ResidencyFact::Device { .. })
                || matches!(input.kind, ValueKindFact::Numeric(numeric)
                    if matches!(numeric.class, NumericClass::Single | NumericClass::Double)
                        && numeric.domain != NumericDomain::Complex)
            {
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
                    // On every successful call the dynamic text must be `like`.
                    apply_trigonometric_prototype(&mut output, prototype, name, &mut diagnostics);
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

    (output, diagnostics)
}
