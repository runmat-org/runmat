use crate::catalog::inference::materialize;
use runmat_types::{
    CellFact, DimensionFact, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, StructFact, ValueFact, ValueKindFact,
};

pub(super) fn facts(input: Option<&ValueFact>) -> Vec<ValueFact> {
    vec![ordered(input), permutation(input)]
}

fn ordered(input: Option<&ValueFact>) -> ValueFact {
    let Some(input) = input.filter(|fact| valid_target(&fact.kind)) else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let mut output = input.clone();
    materialize(&mut output);
    output
}

fn permutation(input: Option<&ValueFact>) -> ValueFact {
    let count = input.and_then(field_count);
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        count.map_or_else(
            || ShapeFact::Shaped {
                dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)],
            },
            |count| {
                if count == 1 {
                    ShapeFact::Scalar
                } else {
                    ShapeFact::Shaped {
                        dims: vec![DimensionFact::Known(count), DimensionFact::Known(1)],
                    }
                }
            },
        ),
        if count == Some(1) {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    )
}

fn valid_target(kind: &ValueKindFact) -> bool {
    matches!(kind, ValueKindFact::Struct(_) | ValueKindFact::Cell(_))
}

fn field_count(input: &ValueFact) -> Option<usize> {
    match &input.kind {
        ValueKindFact::Struct(structure) => complete_field_count(structure),
        ValueKindFact::Cell(cell) => represented_field_count(cell),
        _ => None,
    }
}

fn complete_field_count(structure: &StructFact) -> Option<usize> {
    structure.fields_complete.then_some(structure.fields.len())
}

fn represented_field_count(cell: &CellFact) -> Option<usize> {
    if cell.elements_complete {
        let Some(first) = cell.elements.first() else {
            return Some(0);
        };
        let ValueKindFact::Struct(first) = &first.kind else {
            return None;
        };
        if !first.fields_complete {
            return None;
        }
        let names = first.fields.keys().collect::<Vec<_>>();
        let uniform = cell.elements.iter().all(|element| {
            matches!(&element.kind, ValueKindFact::Struct(structure)
                if structure.fields_complete
                    && structure.fields.keys().eq(names.iter().copied()))
        });
        return uniform.then_some(names.len());
    }
    match &cell.element.kind {
        ValueKindFact::Struct(structure) => complete_field_count(structure),
        _ => None,
    }
}
