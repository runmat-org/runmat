use runmat_types::{CellFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact};
use std::collections::BTreeMap;

pub(super) fn result() -> ValueFact {
    let names = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(character_row()),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![None, Some(1)]),
        StorageFact::Dense,
    );
    let fields = BTreeMap::from([
        ("classes".into(), names.clone()),
        ("m".into(), names.clone()),
        ("mat".into(), names.clone()),
        ("mex".into(), names.clone()),
        ("packages".into(), names),
        ("path".into(), character_row()),
    ]);
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete: true,
    }))
}

fn character_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}
