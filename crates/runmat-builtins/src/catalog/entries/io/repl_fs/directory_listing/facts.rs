use runmat_types::{
    CellFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact, StructFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn metadata_listing() -> ValueFact {
    let fields = BTreeMap::from([
        ("bytes".into(), double_scalar()),
        ("date".into(), character_row()),
        ("datenum".into(), double_scalar()),
        ("folder".into(), character_row()),
        ("isdir".into(), ValueFact::scalar(ValueKindFact::Logical)),
        ("name".into(), character_row()),
    ]);
    let element = ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete: true,
    }));
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    )
}

pub(super) fn name_listing() -> ValueFact {
    character_rows(None)
}

fn character_row() -> ValueFact {
    character_rows(Some(1))
}

fn character_rows(rows: Option<usize>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![rows, None]),
        StorageFact::Dense,
    )
}

fn double_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}
