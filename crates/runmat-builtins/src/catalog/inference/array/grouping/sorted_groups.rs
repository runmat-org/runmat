use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DimensionFact, DynamicReason,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use super::super::super::{argument_error, support};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = validate(request);
    let first = request.arguments.first();
    let table_input = first.is_some_and(is_tabular);
    let mut outputs = vec![group_numbers(first, table_input)];
    let requested = request.outputs.requested.known_count();
    let identifier_count = requested.map(|count| count.saturating_sub(1)).unwrap_or(0);

    if table_input {
        if identifier_count > 0 {
            outputs.push(identifier(first));
        }
        if identifier_count > 1 {
            diagnostics.push(argument_error(
                "RM-CATALOG-FINDGROUPS-OUTPUTS",
                "table findgroups returns at most G and TID",
                2,
            ));
        }
    } else {
        let identifiers = expanded_identifiers(request);
        for identifier in identifiers.iter().take(identifier_count) {
            outputs.push(identifier.clone());
        }
        if identifiers.len() < identifier_count && expansion_is_complete(request) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FINDGROUPS-OUTPUTS",
                "findgroups returns one identifier output per grouping input",
                identifiers.len() + 1,
            ));
        } else {
            outputs.extend(
                (outputs.len()..=identifier_count)
                    .map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)),
            );
        }
    }

    let mut contract = if requested.is_none() {
        CallContract::dynamic(DynamicReason::RuntimeValue)
    } else {
        CallContract::fixed(outputs)
    };
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn validate(request: &CallRequest) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-FINDGROUPS-ARITY",
            "findgroups requires at least one grouping input",
            0,
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        if !supported(input) && !matches!(input.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FINDGROUPS-INPUT",
                "findgroups requires supported grouping vectors or a table",
                index,
            ));
        }
    }
    diagnostics
}

fn supported(input: &ValueFact) -> bool {
    match &input.kind {
        ValueKindFact::Numeric(value) => value.domain == NumericDomain::Real,
        ValueKindFact::Logical | ValueKindFact::String | ValueKindFact::Cell(_) => true,
        ValueKindFact::Object(value) => value.runtime_class.as_ref().is_none_or(|class| {
            class.is(runmat_types::standard::TABLE)
                || class.is(runmat_types::standard::TIMETABLE)
                || class.is(runmat_types::standard::CATEGORICAL)
                || class.is(runmat_types::standard::DATETIME)
                || class.is(runmat_types::standard::DURATION)
                || class.is(runmat_types::standard::CALENDAR_DURATION)
        }),
        _ => false,
    }
}

fn is_tabular(input: &ValueFact) -> bool {
    matches!(&input.kind, ValueKindFact::Object(value) if value.runtime_class.as_ref().is_some_and(|class| class.is(runmat_types::standard::TABLE) || class.is(runmat_types::standard::TIMETABLE)))
}

fn group_numbers(input: Option<&ValueFact>, table: bool) -> ValueFact {
    let shape = input.map_or(ShapeFact::Unknown, |input| {
        if table || super::roles::is_known_matrix(input) {
            ShapeFact::from(vec![first_dimension(&input.shape), Some(1)])
        } else {
            input.shape.clone()
        }
    });
    host(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
    )
}

fn identifier(input: Option<&ValueFact>) -> ValueFact {
    input
        .map(|input| host(input.kind.clone(), ShapeFact::from(vec![None, Some(1)])))
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
}

fn expanded_identifiers(request: &CallRequest) -> Vec<ValueFact> {
    request
        .arguments
        .iter()
        .flat_map(|input| {
            let count = super::roles::known_expanded_count(input).unwrap_or(1);
            std::iter::repeat_with(|| identifier(Some(input))).take(count)
        })
        .collect()
}

fn expansion_is_complete(request: &CallRequest) -> bool {
    request
        .arguments
        .iter()
        .all(|input| super::roles::known_expanded_count(input).is_some())
}

fn first_dimension(shape: &ShapeFact) -> Option<usize> {
    match shape {
        ShapeFact::Shaped { dims } => dims.first().and_then(|dimension| match dimension {
            DimensionFact::Known(value) => Some(*value),
            DimensionFact::Symbolic(_) | DimensionFact::Unknown => None,
        }),
        ShapeFact::Scalar => Some(1),
        ShapeFact::Ranked { .. } | ShapeFact::Unknown => None,
    }
}

fn host(kind: ValueKindFact, shape: ShapeFact) -> ValueFact {
    let mut fact = ValueFact::proven(kind, shape, StorageFact::Dense);
    support::facts::materialize(&mut fact);
    fact.residency = ResidencyFact::Host;
    fact
}
