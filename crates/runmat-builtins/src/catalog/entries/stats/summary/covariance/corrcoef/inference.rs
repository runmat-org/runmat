use super::{CorrcoefOptionKey, CorrcoefRowsOption};
use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn is_text(value: &ValueFact) -> bool {
    matches!(
        &value.kind,
        ValueKindFact::String | ValueKindFact::Character
    )
}

fn validate_options(
    request: &CallRequest,
    start: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let mut index = start;
    let mut rows_seen = false;
    let mut alpha_seen = false;
    while index < request.arguments.len() {
        let Some(key) = request.literals.literal_string_at(index) else {
            if !is_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CORRCOEF-OPTION",
                    "option name must be text",
                    index,
                ));
            }
            // A computed option key cannot be validated before execution.
            index = index.saturating_add(2);
            continue;
        };
        let Some(value) = request.arguments.get(index + 1) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-CORRCOEF-OPTION",
                "option requires a value",
                index,
            ));
            break;
        };
        match CorrcoefOptionKey::parse(&key) {
            Some(CorrcoefOptionKey::Rows) => {
                if rows_seen {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CORRCOEF-DUPLICATE",
                        "Rows cannot be repeated",
                        index,
                    ));
                }
                rows_seen = true;
                if let Some(rows) = request.literals.literal_string_at(index + 1) {
                    if CorrcoefRowsOption::parse(&rows).is_none() {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-CORRCOEF-ROWS",
                            "unsupported Rows value",
                            index + 1,
                        ));
                    }
                } else if !is_text(value) && !matches!(&value.kind, ValueKindFact::Unknown) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CORRCOEF-ROWS",
                        "Rows value must be text",
                        index + 1,
                    ));
                }
            }
            Some(CorrcoefOptionKey::Alpha) => {
                if alpha_seen {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CORRCOEF-DUPLICATE",
                        "Alpha cannot be repeated",
                        index,
                    ));
                }
                alpha_seen = true;
                if !matches!(
                    &value.kind,
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Single | NumericClass::Double,
                        domain: NumericDomain::Real,
                    }) | ValueKindFact::Unknown
                ) || value.shape.element_count().is_some_and(|count| count != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CORRCOEF-ALPHA",
                        "Alpha must be a real floating scalar",
                        index + 1,
                    ));
                }
                if let Some(alpha) = request.literals.numeric_at(index + 1) {
                    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) || alpha == 0.0 {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-CORRCOEF-ALPHA",
                            "Alpha must be strictly between zero and one",
                            index + 1,
                        ));
                    }
                }
            }
            None => diagnostics.push(argument_error(
                "RM-CATALOG-CORRCOEF-OPTION",
                "unsupported corrcoef option",
                index,
            )),
        }
        index += 2;
    }
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-CORRCOEF-ARITY",
            "corrcoef requires observations",
            0,
        ));
    }
    let first = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let first_geometry = super::super::geometry::observations(&first, 0, true, &mut diagnostics);
    let second = request.arguments.get(1).filter(|value| !is_text(value));
    let second_unknown = second.is_some_and(|value| matches!(&value.kind, ValueKindFact::Unknown));
    let paired = second.is_some() && !second_unknown;
    let width = if let Some(second) = second.filter(|_| paired) {
        let second_geometry =
            super::super::geometry::observations(second, 1, true, &mut diagnostics);
        if let (
            DimensionFact::Known(left_rows),
            DimensionFact::Known(left_columns),
            DimensionFact::Known(right_rows),
            DimensionFact::Known(right_columns),
        ) = (
            &first_geometry.rows,
            &first_geometry.columns,
            &second_geometry.rows,
            &second_geometry.columns,
        ) {
            if left_rows != right_rows || left_columns != right_columns {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CORRCOEF-SIZE",
                    "A and B must have the same size",
                    1,
                ));
            }
        }
        if first.shape.element_count() == Some(1) && second.shape.element_count() == Some(1) {
            DimensionFact::Known(1)
        } else {
            DimensionFact::Known(2)
        }
    } else if second_unknown {
        DimensionFact::Unknown
    } else {
        first_geometry.variables
    };
    if !second_unknown {
        validate_options(request, if paired { 2 } else { 1 }, &mut diagnostics);
    }
    let domain = if request
        .arguments
        .iter()
        .take(if paired { 2 } else { 1 })
        .any(|argument| {
            matches!(
                &argument.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Complex,
                    ..
                })
            )
        }) {
        NumericDomain::Complex
    } else {
        NumericDomain::Real
    };
    let class = if second_unknown {
        None
    } else if let Some(second) = second.filter(|_| paired) {
        super::super::geometry::output_class(&[&first, second])
    } else {
        super::super::geometry::output_class(&[&first])
    };
    let mut input = first;
    if second.is_some_and(|value| !matches!(&value.residency, ResidencyFact::Host)) {
        input.residency = ResidencyFact::Unknown;
    }
    if second_unknown {
        input.kind = ValueKindFact::Unknown;
    }
    let shape = super::super::geometry::square_shape(width);
    let outputs = (0..4)
        .map(|_| super::super::geometry::numeric_output(&input, shape.clone(), domain, class))
        .collect();
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if domain == NumericDomain::Complex
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|count| count > 1)
    {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-CORRCOEF-COMPLEX-OUTPUTS",
            "complex correlation supports only one output",
            0,
        ));
    }
    result.diagnostics.extend(diagnostics);
    result
}
