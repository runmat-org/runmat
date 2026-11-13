use runmat_types::{
    FactJoin, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

pub(super) fn text(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Unknown) || super::text_input::scalar(value)
}

pub(super) fn filter(value: &ValueFact) -> bool {
    let ValueKindFact::Cell(cell) = &value.kind else {
        return text(value);
    };
    if value.shape.element_count() == Some(0) {
        return true;
    }
    if value.shape.known_dims().is_some_and(|dims| {
        dims.iter().skip(2).any(|dim| dim.is_some_and(|n| n != 1))
            || dims
                .get(1)
                .is_some_and(|dim| dim.is_some_and(|n| n == 0 || n > 2))
    }) {
        return false;
    }
    if cell.elements_complete {
        cell.elements.iter().all(text)
    } else {
        text(&cell.element) && cell.elements.iter().all(text)
    }
}

pub(super) fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

pub(super) fn text_or_cancel() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
    .join(&double_scalar())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{CellFact, DynamicReason};
    #[test]
    fn filter_admission_keeps_unknowns_and_rejects_known_pages() {
        let element = ValueFact::scalar(ValueKindFact::String);
        for (tail, valid) in [(Some(1), true), (Some(2), false), (None, true)] {
            let value = ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(element.clone()),
                    elements: vec![],
                    elements_complete: false,
                }),
                ShapeFact::from(vec![Some(2), Some(2), tail]),
                StorageFact::Dense,
            );
            assert_eq!(filter(&value), valid);
        }
        assert!(filter(&ValueFact::unknown(DynamicReason::RuntimeValue)));
        assert!(!filter(&double_scalar()));
    }
}
