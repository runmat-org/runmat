use runmat_types::{DynamicReason, StructFact, ValueFact, ValueKindFact};
use std::collections::BTreeMap;

pub(super) fn fact(input: Option<&ValueFact>) -> ValueFact {
    input
        .and_then(remove_schema)
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
}

fn remove_schema(input: &ValueFact) -> Option<ValueFact> {
    let mut output = input.clone();
    output.kind = match &input.kind {
        ValueKindFact::Struct(_) => erased_structure(),
        _ => return None,
    };
    Some(output)
}

fn erased_structure() -> ValueKindFact {
    ValueKindFact::Struct(StructFact::array(BTreeMap::new(), false, Vec::new(), false))
}
