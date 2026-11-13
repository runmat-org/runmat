use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DynamicReason, ResidencyFact,
    StructFact, ValueFact, ValueKindFact,
};

fn result_tree(input: &ValueFact) -> ValueFact {
    let unknown_leaf = || ValueFact::unknown(DynamicReason::RuntimeValue);
    let kind = match &input.kind {
        ValueKindFact::Cell(cell) => ValueKindFact::Cell(CellFact {
            element: Box::new(unknown_leaf()),
            elements: cell.elements.iter().map(|_| unknown_leaf()).collect(),
            elements_complete: cell.elements_complete,
        }),
        ValueKindFact::Struct(st) => ValueKindFact::Struct(StructFact {
            fields: st
                .fields
                .keys()
                .map(|name| (name.clone(), unknown_leaf()))
                .collect(),
            fields_complete: st.fields_complete,
            elements: st
                .elements
                .iter()
                .map(|fields| {
                    fields
                        .keys()
                        .map(|name| (name.clone(), unknown_leaf()))
                        .collect()
                })
                .collect(),
            elements_complete: st.elements_complete,
        }),
        // A callback may change a leaf's class, shape, and device. No value
        // fact for a generic callable proves what that callback returns.
        _ => return unknown_leaf(),
    };
    let mut result = ValueFact::proven(kind, input.shape.clone(), input.storage);
    result.residency = ResidencyFact::Host;
    result
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-DL-DLUPDATE-ARITY",
            "dlupdate requires a callback and at least one parameter tree",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if let Some(function) = request.arguments.first() {
        if !matches!(
            function.kind,
            ValueKindFact::Unknown | ValueKindFact::Callable(_)
        ) {
            diagnostics.push(argument_error(
                "RM-DL-DLUPDATE-CALLABLE",
                "the first input must be a function handle",
                0,
            ));
        }
    }
    let primary = request.arguments.get(1);
    if let Some(primary) = primary {
        for (index, other) in request.arguments.iter().enumerate().skip(2) {
            match (&primary.kind, &other.kind) {
                (ValueKindFact::Cell(_), ValueKindFact::Cell(_))
                | (ValueKindFact::Struct(_), ValueKindFact::Struct(_)) => {
                    if primary
                        .shape
                        .known_dims()
                        .zip(other.shape.known_dims())
                        .is_some_and(|(a, b)| {
                            a.iter().all(Option::is_some) && b.iter().all(Option::is_some) && a != b
                        })
                    {
                        diagnostics.push(argument_error(
                            "RM-DL-DLUPDATE-TREE",
                            "matching parameter containers must have the same shape",
                            index,
                        ));
                    }
                }
                (ValueKindFact::Cell(_), ValueKindFact::Unknown)
                | (ValueKindFact::Struct(_), ValueKindFact::Unknown)
                | (ValueKindFact::Unknown, _)
                | (_, ValueKindFact::Unknown) => {}
                (ValueKindFact::Cell(_) | ValueKindFact::Struct(_), _)
                | (_, ValueKindFact::Cell(_) | ValueKindFact::Struct(_)) => {
                    diagnostics.push(argument_error(
                        "RM-DL-DLUPDATE-TREE",
                        "parameter trees must have matching container structure",
                        index,
                    ))
                }
                _ => {}
            }
        }
    }
    let output = primary
        .map(result_tree)
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let mut contract = CallContract::fixed(vec![output.clone()]);
    contract.variadic_output = Some(Box::new(output));
    contract.maximum_outputs = None;
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    diagnostics.append(&mut result.diagnostics);
    result.diagnostics = diagnostics;
    result
}
