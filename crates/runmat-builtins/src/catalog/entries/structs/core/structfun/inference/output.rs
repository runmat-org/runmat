use super::callback::FieldOutputs;
use runmat_types::{
    DimensionFact, DynamicReason, FactJoin, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn collect(
    fields: &FieldOutputs,
    count: usize,
    uniform: Option<bool>,
) -> Vec<ValueFact> {
    (0..count)
        .map(|slot| match uniform {
            Some(false) => structure(fields, slot),
            Some(true) if fields.is_empty() => empty_double_column(),
            Some(true) => uniform_column(fields, slot),
            None => ValueFact::unknown(DynamicReason::RuntimeValue),
        })
        .collect()
}

fn structure(fields: &FieldOutputs, slot: usize) -> ValueFact {
    let mapped = fields
        .iter()
        .filter(|(name, _)| !name.is_empty())
        .map(|(name, outputs)| {
            (
                name.clone(),
                outputs
                    .get(slot)
                    .cloned()
                    .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
            )
        })
        .collect::<BTreeMap<_, _>>();
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields: mapped,
        fields_complete: fields.iter().all(|(name, _)| !name.is_empty()),
    }))
}

fn uniform_column(fields: &FieldOutputs, slot: usize) -> ValueFact {
    let joined = fields
        .iter()
        .filter_map(|(_, outputs)| outputs.get(slot))
        .cloned()
        .reduce(|left, right| left.join(&right))
        .unwrap_or_else(empty_double_column);
    let rows = if fields.iter().all(|(name, _)| !name.is_empty()) {
        DimensionFact::Known(fields.len())
    } else {
        DimensionFact::Unknown
    };
    ValueFact::proven(
        joined.kind,
        ShapeFact::Shaped {
            dims: vec![rows, DimensionFact::Known(1)],
        },
        StorageFact::Dense,
    )
}

fn empty_double_column() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(0), DimensionFact::Known(1)],
        },
        StorageFact::Dense,
    )
}
