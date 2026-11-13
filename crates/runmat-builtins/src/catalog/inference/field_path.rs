//! Shared static interpretation of functional field-path selectors.

use runmat_types::{IndexSelectorFact, ValueFact, ValueKindFact};

pub(in crate::catalog) fn is_selector(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Cell(_))
}

pub(in crate::catalog) fn selector_facts(value: Option<&ValueFact>) -> Vec<IndexSelectorFact> {
    let Some(ValueKindFact::Cell(cell)) = value.map(|fact| &fact.kind) else {
        return vec![IndexSelectorFact::Unknown];
    };
    if !cell.elements_complete {
        return vec![IndexSelectorFact::Unknown];
    }
    cell.elements.iter().map(selector_fact).collect()
}

fn selector_fact(value: &ValueFact) -> IndexSelectorFact {
    match &value.kind {
        ValueKindFact::Logical => IndexSelectorFact::Logical(value.clone()),
        ValueKindFact::Numeric(_) => {
            if value.is_scalar() {
                IndexSelectorFact::Scalar
            } else {
                IndexSelectorFact::Numeric(value.clone())
            }
        }
        _ => IndexSelectorFact::Unknown,
    }
}
