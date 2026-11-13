//! Shared facts for the host text-model domain; declaration-selected policy,
//! never builtin-name dispatch. Runtime-dependent vocabulary/length stays open.
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
mod facts;
use facts::{numeric, object, rows, text_shape};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) enum ModelOperation {
    Document,
    Bag,
    Filter,
    Select,
    Stop,
    Indices,
    Words,
    Membership,
    Sequences,
    Vectors,
    Neighbors,
    Write,
    Similarity,
}

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    op: ModelOperation,
) -> CallInference {
    use ModelOperation::*;
    let count = request.arguments.len();
    let mut diagnostics = Vec::new();
    let valid = match op {
        Document => count == 0 || count % 2 == 1,
        Bag => count <= 2,
        Filter => count == 2,
        Select => count >= 2 && count.is_multiple_of(2),
        Stop => count >= 1 && count % 2 == 1,
        Indices | Membership | Vectors | Sequences => count >= 2 && count.is_multiple_of(2),
        Words | Write => count == 2,
        Neighbors => count >= 2,
        Similarity => (1..=2).contains(&count),
    };
    if !valid {
        diagnostics.push(argument_error(
            "RM-CATALOG-TEXT-MODEL-ARITY",
            "inputs do not match the declared text-model form",
            count,
        ));
    }
    let first = request.arguments.first();
    let second = request.arguments.get(1);
    let required_class = match op {
        Indices | Words => Some(StaticClassIdentity::new("wordEncoding")),
        Vectors | Neighbors | Write => Some(StaticClassIdentity::new("wordEmbedding")),
        Stop => Some(standard::TOKENIZED_DOCUMENT),
        _ => None,
    };
    if let (Some(class), Some(value)) = (required_class, first) {
        match &value.kind {
            ValueKindFact::Object(value)
                if value
                    .runtime_class
                    .as_ref()
                    .is_none_or(|actual| *actual == class.owned()) => {}
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-TEXT-MODEL-CLASS",
                "input requires the declared host model object",
                0,
            )),
        }
    }
    let classes: &[StaticClassIdentity] = match op {
        Filter | Select => &[standard::TOKENIZED_DOCUMENT, standard::BAG_OF_WORDS],
        Membership | Sequences => &[
            StaticClassIdentity::new("wordEncoding"),
            StaticClassIdentity::new("wordEmbedding"),
        ],
        _ => &[],
    };
    if !classes.is_empty() {
        if let Some(value) = first {
            let valid = match &value.kind {
                ValueKindFact::Object(value) => value
                    .runtime_class
                    .as_ref()
                    .is_none_or(|actual| classes.iter().any(|class| class.owned() == *actual)),
                ValueKindFact::Unknown => true,
                _ => false,
            };
            if !valid {
                diagnostics.push(argument_error(
                    "RM-CATALOG-TEXT-MODEL-CLASS",
                    "input requires the declared host model object",
                    0,
                ));
            }
        }
    }
    if matches!(op, Document)
        && first
            .is_some_and(|v| matches!(v.kind, ValueKindFact::Numeric(_) | ValueKindFact::Logical))
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-TEXT-MODEL-TEXT",
            "document input requires text or supported text containers",
            0,
        ));
    }
    let unknown = || ValueFact::unknown(DynamicReason::RuntimeValue);
    let outputs = if !diagnostics.is_empty() {
        vec![unknown()]
    } else {
        match op {
            Document => vec![object(Some(standard::TOKENIZED_DOCUMENT))],
            Bag => vec![object(Some(standard::BAG_OF_WORDS))],
            Stop => vec![object(Some(standard::TOKENIZED_DOCUMENT))],
            Filter | Select => vec![match first.map(|v| &v.kind) {
                Some(ValueKindFact::Object(v))
                    if v.runtime_class.as_ref() == Some(&standard::TOKENIZED_DOCUMENT.owned()) =>
                {
                    object(Some(standard::TOKENIZED_DOCUMENT))
                }
                Some(ValueKindFact::Object(v))
                    if v.runtime_class.as_ref() == Some(&standard::BAG_OF_WORDS.owned()) =>
                {
                    object(Some(standard::BAG_OF_WORDS))
                }
                _ => unknown(),
            }],
            Indices => vec![numeric(
                NumericDomain::Real,
                text_shape(second),
                StorageFact::Dense,
            )],
            Words => vec![ValueFact::proven(
                ValueKindFact::String,
                second
                    .map(|v| v.shape.clone())
                    .unwrap_or(ShapeFact::Unknown),
                StorageFact::Dense,
            )],
            Membership => vec![ValueFact::proven(
                ValueKindFact::Logical,
                text_shape(second),
                StorageFact::Dense,
            )],
            Vectors => vec![numeric(
                NumericDomain::Real,
                ShapeFact::from(vec![None, None]),
                StorageFact::Dense,
            )],
            Neighbors => {
                let shape = ShapeFact::from(vec![rows(second), None]);
                vec![
                    ValueFact::proven(ValueKindFact::String, shape.clone(), StorageFact::Dense),
                    numeric(NumericDomain::Real, shape, StorageFact::Dense),
                ]
            }
            Sequences => vec![ValueFact::proven(
                ValueKindFact::Cell(CellFact {
                    element: Box::new(numeric(
                        NumericDomain::Real,
                        ShapeFact::from(vec![None, None]),
                        StorageFact::Dense,
                    )),
                    elements: vec![],
                    elements_complete: false,
                }),
                ShapeFact::Unknown,
                StorageFact::Dense,
            )],
            Write => vec![],
            Similarity => {
                let values = request.arguments.iter();
                let complex = values.clone().any(|v| {
                    matches!(
                        &v.kind,
                        ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Complex,
                            ..
                        })
                    )
                });
                let known = values.clone().all(|v| {
                    matches!(
                        &v.kind,
                        ValueKindFact::Numeric(_)
                            | ValueKindFact::Object(_)
                            | ValueKindFact::String
                            | ValueKindFact::Character
                            | ValueKindFact::Cell(_)
                            | ValueKindFact::Logical
                    )
                });
                if known {
                    let shape = ShapeFact::from(vec![rows(first), rows(second.or(first))]);
                    vec![numeric(
                        if complex {
                            NumericDomain::Complex
                        } else {
                            NumericDomain::Real
                        },
                        shape,
                        if complex {
                            StorageFact::Dense
                        } else {
                            StorageFact::Sparse
                        },
                    )]
                } else {
                    vec![unknown()]
                }
            }
        }
    };
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
