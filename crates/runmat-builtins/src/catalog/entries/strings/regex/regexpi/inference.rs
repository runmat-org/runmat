use crate::{
    catalog::inference::{argument_error, numeric_kind},
    BuiltinCatalogEntry,
};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DynamicReason,
    NumericClass, NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-REGEXPI-ARITY",
            "regexpi requires subject and expression",
            request.arguments.len(),
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !matches!(
            &argument.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-REGEXPI-TEXT",
                "regexpi accepts host text inputs and option flags",
                index,
            ));
        }
    }
    let mut outputs = Vec::new();
    let requested = request.outputs.requested.known_count().unwrap_or(1).max(1);
    let unknown_option = (2..request.arguments.len())
        .any(|index| request.literals.literal_string_at(index).is_none());
    let explicit: Vec<_> = (2..request.arguments.len())
        .filter_map(|index| request.literals.literal_string_at(index))
        .filter(|name| {
            matches!(
                name.as_str(),
                "start" | "end" | "match" | "tokens" | "tokenextents" | "names" | "split"
            )
        })
        .collect();
    let force_cell = (2..request.arguments.len()).any(|index| {
        request.literals.literal_string_at(index).as_deref() == Some("forcecelloutput")
    });
    let once = (2..request.arguments.len())
        .any(|index| request.literals.literal_string_at(index).as_deref() == Some("once"));
    for index in 0..requested.min(6) {
        let output = match (unknown_option, explicit.get(index).map(String::as_str)) {
            (true, _) => ValueFact::unknown(DynamicReason::RuntimeValue),
            (false, Some("start" | "end")) => index_output(request.arguments.first(), force_cell),
            (false, Some("match")) if once => ValueFact::unknown(DynamicReason::RuntimeValue),
            (false, Some("match" | "tokens" | "split")) => cell_output(request.arguments.first()),
            (false, Some("tokenextents" | "names")) => {
                ValueFact::unknown(DynamicReason::RuntimeValue)
            }
            (false, None) if explicit.is_empty() && index < 2 => {
                index_output(request.arguments.first(), force_cell)
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        };
        outputs.push(output);
    }
    let mut contract = CallContract::fixed(outputs);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    diagnostics.append(&mut inferred.diagnostics);
    inferred.diagnostics = diagnostics;
    inferred
}

fn index_output(subject: Option<&ValueFact>, force_cell: bool) -> ValueFact {
    let Some(subject) = subject else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if matches!(&subject.kind, ValueKindFact::Unknown) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let scalar = match &subject.kind {
        ValueKindFact::String => subject.shape.element_count() == Some(1),
        ValueKindFact::Character => subject
            .shape
            .known_dims()
            .is_some_and(|dims| dims.first().copied().flatten() == Some(1)),
        _ => false,
    };
    if !scalar || force_cell {
        return cell_output(Some(subject));
    }
    let mut output = ValueFact::proven(
        numeric_kind(NumericClass::Double, NumericDomain::Real),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    output
}
fn cell_output(subject: Option<&ValueFact>) -> ValueFact {
    if subject.is_none_or(|value| matches!(&value.kind, ValueKindFact::Unknown)) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Cell(runmat_types::CellFact {
        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        elements: vec![],
        elements_complete: false,
    });
    if let Some(subject) = subject {
        output.shape = match &subject.kind {
            ValueKindFact::String | ValueKindFact::Cell(_) => subject.shape.clone(),
            ValueKindFact::Character => subject
                .shape
                .known_dims()
                .map_or(ShapeFact::Unknown, |dims| {
                    ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
                }),
            _ => ShapeFact::Unknown,
        };
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, OutputSelection};
    #[test]
    fn scalar_indices_are_double_with_dynamic_match_count() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::REGEXPI_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].kind,
            numeric_kind(NumericClass::Double, NumericDomain::Real)
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
    #[test]
    fn array_results_remain_containers_and_invalid_inputs_diagnose() {
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::REGEXPI_CATALOG_ENTRY);
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
        request.arguments[1] = ValueFact::scalar(ValueKindFact::Logical);
        assert!(infer(&request, &super::super::REGEXPI_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-REGEXPI-TEXT"));
    }
}
