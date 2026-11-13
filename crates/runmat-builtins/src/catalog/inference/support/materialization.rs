//! Facts after successful `dispatcher::gather_if_needed_async` execution.
//!
//! This describes the value transform, not its effects, capability admission,
//! provider availability, or failure ordering. Callers retain those contracts.
//! In particular this is not distributed gather or generic densification.
//!
//! Applicable callback boundaries: cellfun gathers each cell argument and extra
//! argument before primary/error callbacks, and gathers uniform results only;
//! structfun gathers field inputs before primary/error callbacks and all callback
//! results before collection. Neither boundary gathers the callable itself.
//! Consumers must apply this helper only at those actual runtime gather points.

use runmat_types::{
    AliasFact, ContiguityFact, FactJoin, LayoutFact, MutationFact, ResidencyFact, StorageFact,
    ValueFact, ValueKindFact, ViewFact,
};

/// Preserve class, shape, storage, and aggregate facts without inventing host
/// residency for foreign/remote/unknown representations.
pub fn gathered_value_fact(input: &ValueFact) -> ValueFact {
    let mut output = input.clone();
    match &mut output.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical
            if matches!(input.residency, ResidencyFact::Device { .. }) =>
        {
            output.residency = ResidencyFact::Host;
            // Download owns a host payload, but does not change sparse storage
            // into dense storage or infer storage from the number of elements.
            output.alias = AliasFact::Unique;
            output.mutation = MutationFact::ValueSemantics;
            output.view = ViewFact::Materialized;
            if matches!(input.storage, StorageFact::Dense | StorageFact::Scalar) {
                output.layout = LayoutFact::ColumnMajor;
                output.contiguity = ContiguityFact::Contiguous;
            }
        }
        ValueKindFact::Cell(cell) => {
            cell.element = Box::new(gathered_value_fact(&cell.element));
            for element in &mut cell.elements {
                *element = gathered_value_fact(element);
            }
        }
        ValueKindFact::Struct(structure) => {
            for field in structure.fields.values_mut() {
                *field = gathered_value_fact(field);
            }
            for element in &mut structure.elements {
                for field in element.values_mut() {
                    *field = gathered_value_fact(field);
                }
            }
        }
        ValueKindFact::Object(object) if object.handle_semantics != Some(true) => {
            // Gather visits value-object properties, including value elements
            // of ObjectArray; handle objects pass through. ObjectFact may leave
            // those handle semantics unknown, so preserve the conservative
            // identity/gather join rather than claiming every property is Host.
            for property in object.properties.values_mut() {
                *property = property.join(&gathered_value_fact(property));
            }
        }
        ValueKindFact::Callable(callable) => {
            // Only Value::Closure captures are gathered at runtime. CallableFact
            // also represents handles that pass through unchanged, without a
            // representation discriminator. Parameters/results are contracts,
            // not payloads traversed by gather, and must remain untouched.
            for capture in &mut callable.captures {
                *capture = capture.join(&gathered_value_fact(capture));
            }
        }
        ValueKindFact::Unknown if matches!(input.residency, ResidencyFact::Device { .. }) => {
            // Could download a numeric handle or pass through a foreign handle.
            output.residency = ResidencyFact::Unknown;
            output.layout = LayoutFact::Unknown;
            output.contiguity = ContiguityFact::Unknown;
            output.alias = AliasFact::Unknown;
            output.view = ViewFact::Unknown;
            output.mutation = MutationFact::Unknown;
        }
        // This transient carrier is rejected rather than recursively gathered.
        ValueKindFact::OutputList(_) => return ValueFact::never(),
        // Foreign/distributed/composite/handle values are not traversed. Neither
        // pending execution results nor symbolic values are evaluated by gather.
        _ => {}
    }
    output
}

#[cfg(test)]
mod tests;
