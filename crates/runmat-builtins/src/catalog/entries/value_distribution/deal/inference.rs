use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut contract = CallContract::fixed(request.arguments.clone());
    if let [single] = request.arguments.as_slice() {
        let mut output = single.clone();
        if request
            .outputs
            .requested
            .known_count()
            .is_none_or(|count| count > 1)
        {
            forget_replicated_uniqueness(&mut output);
        }
        contract.outputs = vec![output.clone()];
        contract.variadic_output = Some(Box::new(output));
        contract.maximum_outputs = None;
    }
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1
        && request
            .outputs
            .requested
            .known_count()
            .is_some_and(|n| n != request.arguments.len())
    {
        result.diagnostics.push(argument_error(
            "RM-DEAL-COUNT",
            "deal input and output counts must match unless there is one input",
            0,
        ));
    }
    result
}

/// Replication can share CoW buffers, resident handles and reference-bearing
/// payloads. Keep semantic facts, but do not promise exclusive ownership. Unknown
/// is deliberate: immediate scalars/deep aggregate shells may instead be copied.
fn forget_replicated_uniqueness(value: &mut ValueFact) {
    if value.alias == AliasFact::Unique {
        value.alias = AliasFact::Unknown;
    }
    match &mut value.kind {
        ValueKindFact::Cell(cell) => {
            forget_replicated_uniqueness(&mut cell.element);
            for element in &mut cell.elements {
                forget_replicated_uniqueness(element);
            }
        }
        ValueKindFact::Struct(structure) => {
            for field in structure.fields.values_mut() {
                forget_replicated_uniqueness(field);
            }
            for element in &mut structure.elements {
                for field in element.values_mut() {
                    forget_replicated_uniqueness(field);
                }
            }
        }
        ValueKindFact::Object(object) => {
            for property in object.properties.values_mut() {
                forget_replicated_uniqueness(property);
            }
        }
        ValueKindFact::Callable(callable) => {
            for capture in &mut callable.captures {
                forget_replicated_uniqueness(capture);
            }
            // Parameters/results describe future calls, not captured payloads.
        }
        ValueKindFact::OutputList(sequence) => {
            for output in &mut sequence.outputs {
                forget_replicated_uniqueness(output);
            }
        }
        ValueKindFact::Distributed(distributed) => {
            forget_replicated_uniqueness(&mut distributed.value)
        }
        ValueKindFact::Composite(composite) => forget_replicated_uniqueness(&mut composite.value),
        _ => {}
    }
}
