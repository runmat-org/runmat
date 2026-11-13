use crate::catalog::inference::{argument_error, literal_text};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
mod validation;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    validation::validate(request, &mut diagnostics);
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONFUSIONMAT-ARITY",
            "confusionmat expects true and predicted labels",
            0,
        ));
    }
    let mut order_source = request.arguments.first();
    let mut explicit = false;
    let mut ambiguous = false;
    if !request.arguments.len().saturating_sub(2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONFUSIONMAT-OPTIONS",
            "name-value options must be paired",
            2,
        ));
    }
    for index in (2..request.arguments.len().saturating_sub(1)).step_by(2) {
        match request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        {
            Some(name) if canonical(&name) == "order" => {
                order_source = request.arguments.get(index + 1);
                explicit = true;
            }
            Some(_) => diagnostics.push(argument_error(
                "RM-CATALOG-CONFUSIONMAT-OPTION",
                "unsupported confusionmat option",
                index,
            )),
            None => {
                ambiguous = true;
            }
        }
    }
    let known_logical = !explicit
        && !ambiguous
        && request
            .arguments
            .first()
            .is_some_and(|v| matches!(v.kind, ValueKindFact::Logical));
    let dimension = if known_logical {
        DimensionFact::Known(2)
    } else {
        DimensionFact::Unknown
    };
    let counts = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: vec![dimension.clone(), dimension.clone()],
        },
        StorageFact::Dense,
    );
    let order = if ambiguous {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        order_fact(order_source, dimension)
    };
    let requested = request.outputs.requested.known_count();
    if requested.is_some_and(|n| n > 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-CONFUSIONMAT-OUTPUTS",
            "confusionmat supports at most two outputs",
            0,
        ));
    }
    let outputs = match requested.unwrap_or(1) {
        0 => vec![],
        1 => vec![counts],
        _ => vec![counts, order],
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = requested.is_none();
    result.diagnostics.extend(diagnostics);
    result
}
fn order_fact(source: Option<&ValueFact>, dimension: DimensionFact) -> ValueFact {
    let kind = match source.map(|s| &s.kind) {
        Some(ValueKindFact::Numeric(n)) if n.domain == NumericDomain::Real => {
            ValueKindFact::Numeric(NumericFact {
                class: if n.class.integer_class().is_some() {
                    n.class
                } else {
                    NumericClass::Double
                },
                domain: NumericDomain::Real,
            })
        }
        Some(ValueKindFact::Logical) => ValueKindFact::Logical,
        Some(ValueKindFact::String) => ValueKindFact::String,
        Some(ValueKindFact::Character) => {
            return ValueFact::proven(
                ValueKindFact::Character,
                ShapeFact::Shaped {
                    dims: vec![dimension, DimensionFact::Unknown],
                },
                StorageFact::Dense,
            )
        }
        _ => return ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    ValueFact::proven(
        kind,
        ShapeFact::Shaped {
            dims: vec![dimension, DimensionFact::Known(1)],
        },
        StorageFact::Dense,
    )
}
fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}
