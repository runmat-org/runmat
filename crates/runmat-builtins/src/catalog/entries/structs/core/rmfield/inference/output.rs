use runmat_types::{CellFact, DynamicReason, StructFact, ValueFact, ValueKindFact};
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
        ValueKindFact::Cell(cell) => ValueKindFact::Cell(erased_cell(cell)),
        _ => return None,
    };
    Some(output)
}

fn erased_cell(input: &CellFact) -> CellFact {
    CellFact {
        element: Box::new(erase_element(&input.element)),
        elements: input.elements.iter().map(erase_element).collect(),
        elements_complete: input.elements_complete,
    }
}

fn erase_element(input: &ValueFact) -> ValueFact {
    let mut output = input.clone();
    output.kind = if matches!(input.kind, ValueKindFact::Struct(_)) {
        erased_structure()
    } else {
        ValueKindFact::Unknown
    };
    output
}

fn erased_structure() -> ValueKindFact {
    ValueKindFact::Struct(StructFact {
        fields: BTreeMap::new(),
        fields_complete: false,
    })
}
