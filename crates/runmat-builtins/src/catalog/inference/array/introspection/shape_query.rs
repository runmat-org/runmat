use super::super::super::{argument_error, finish_fixed};
use crate::{BuiltinCatalogEntry, ShapeQuery};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    query: ShapeQuery,
) -> CallInference {
    match query {
        ShapeQuery::Size => infer_size(request, entry),
        ShapeQuery::ElementCount => infer_numel(request, entry),
    }
}

fn infer_numel(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity_diagnostics(request, entry);
    finish_fixed(entry, request, double_scalar(), diagnostics)
}

fn infer_size(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = arity_diagnostics(request, entry);
    let requested = request.outputs.requested.known_count();
    if let Some(count) = requested.filter(|count| *count > 1) {
        if request.arguments.len() > 1 {
            if let Some(selector_count) = statically_known_selector_count(request) {
                if count != selector_count {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-SIZE-OUTPUT-COUNT",
                        format!(
                            "size requests {count} outputs for {selector_count} queried dimensions"
                        ),
                        1,
                    ));
                }
            }
        }
        return finish_contract(
            entry,
            request,
            CallContract::fixed((0..count).map(|_| double_scalar()).collect()),
            diagnostics,
        );
    }

    if requested.is_none() {
        let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
        contract.outputs = vec![size_single_output(request)];
        contract.variadic_output = Some(Box::new(double_scalar()));
        return finish_contract(entry, request, contract, diagnostics);
    }
    finish_fixed(entry, request, size_single_output(request), diagnostics)
}

fn size_single_output(request: &CallRequest) -> ValueFact {
    if request.arguments.len() == 1 {
        let rank = request
            .arguments
            .first()
            .map(|argument| reported_rank(&argument.shape))
            .unwrap_or(None);
        return double_row(rank);
    }
    match statically_known_selector_count(request) {
        Some(1) => double_scalar(),
        count => double_row(count),
    }
}

fn reported_rank(shape: &ShapeFact) -> Option<usize> {
    if let Some(mut dimensions) = shape.known_dims() {
        while dimensions.len() > 2 && dimensions.last() == Some(&Some(1)) {
            dimensions.pop();
        }
        return Some(dimensions.len().max(2));
    }
    shape.rank().map(|rank| rank.max(2))
}

fn statically_known_selector_count(request: &CallRequest) -> Option<usize> {
    if request.arguments.len() > 2 {
        return Some(request.arguments.len() - 1);
    }
    request
        .literals
        .numeric_vector_at(1)
        .map(|values| values.len())
        .or_else(|| {
            request
                .arguments
                .get(1)
                .and_then(|argument| argument.shape.element_count())
        })
}

fn double_scalar() -> ValueFact {
    ValueFact::scalar(double_kind())
}

fn double_row(length: Option<usize>) -> ValueFact {
    ValueFact::proven(
        double_kind(),
        ShapeFact::from(vec![Some(1), length]),
        StorageFact::Dense,
    )
}

fn double_kind() -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    })
}

fn arity_diagnostics(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-SHAPE-QUERY-ARITY",
            format!("{} requires at least one input", entry.identity.name),
            0,
        ));
    }
    diagnostics
}

fn finish_contract(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    mut contract: CallContract,
    mut diagnostics: Vec<runmat_types::InferenceDiagnostic>,
) -> CallInference {
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{OutputSelection, RequestedOutputCount};

    fn request(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> CallRequest {
        CallRequest {
            arguments,
            literals: Default::default(),
            outputs: OutputSelection::new(outputs),
        }
    }

    #[test]
    fn size_distinguishes_vector_and_multiple_output_facts() {
        let input = ValueFact::proven(
            double_kind(),
            ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
            StorageFact::Dense,
        );
        let single = infer_size(
            &request(vec![input.clone()], RequestedOutputCount::One),
            &crate::SIZE_CATALOG_ENTRY,
        );
        assert_eq!(
            single.outputs[0].shape,
            ShapeFact::from(vec![Some(1), Some(3)])
        );

        let multiple = infer_size(
            &request(vec![input], RequestedOutputCount::Exactly(2)),
            &crate::SIZE_CATALOG_ENTRY,
        );
        assert_eq!(multiple.outputs.len(), 2);
        assert!(multiple.outputs.iter().all(ValueFact::is_scalar));
    }

    #[test]
    fn numel_is_always_a_double_scalar() {
        let result = infer_numel(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                RequestedOutputCount::One,
            ),
            &crate::NUMEL_CATALOG_ENTRY,
        );
        assert!(result.outputs[0].is_scalar());
        assert_eq!(result.outputs[0].kind, double_kind());
    }
}
