use crate::catalog::inference::materialize;
use runmat_types::{
    DimensionFact, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    StructFact, ValueFact, ValueKindFact,
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
    matches!(kind, ValueKindFact::Struct(_))
}

fn field_count(input: &ValueFact) -> Option<usize> {
    match &input.kind {
        ValueKindFact::Struct(structure) => complete_field_count(structure),
        _ => None,
    }
}

fn complete_field_count(structure: &StructFact) -> Option<usize> {
    structure.fields_complete.then_some(structure.fields.len())
}
