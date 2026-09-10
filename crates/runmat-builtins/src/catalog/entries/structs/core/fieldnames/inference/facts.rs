use runmat_types::{
    CellFact, DimensionFact, ObjectFact, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};

pub(super) fn output(input: Option<&ValueFact>) -> ValueFact {
    field_name_cell(input.and_then(field_count))
}

fn field_count(input: &ValueFact) -> Option<usize> {
    match &input.kind {
        ValueKindFact::Struct(value) => complete_struct_count(value),
        ValueKindFact::Object(value) => complete_object_count(value),
        _ => None,
    }
}

fn complete_struct_count(value: &StructFact) -> Option<usize> {
    value.fields_complete.then_some(value.fields.len())
}

fn complete_object_count(value: &ObjectFact) -> Option<usize> {
    value.properties_complete.then_some(value.properties.len())
}

fn field_name_cell(rows: Option<usize>) -> ValueFact {
    let character_row = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Unknown],
        },
        StorageFact::Dense,
    );
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(character_row),
            elements: Vec::new(),
            elements_complete: rows == Some(0),
        }),
        ShapeFact::Shaped {
            dims: vec![
                rows.map_or(DimensionFact::Unknown, DimensionFact::Known),
                DimensionFact::Known(1),
            ],
        },
        StorageFact::Dense,
    )
}
