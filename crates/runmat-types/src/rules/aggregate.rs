use crate::{
    reduction_shape, repmat_shape, reshape_shape, CellFact, DimensionFact, DynamicReason,
    FactInference, FactJoin, InferenceDiagnostic, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

pub fn infer_tensor_aggregate(rows: &[Vec<ValueFact>]) -> FactInference {
    if rows.is_empty() || rows.iter().all(Vec::is_empty) {
        return FactInference::exact(ValueFact::proven(
            ValueKindFact::Numeric(crate::NumericFact {
                class: crate::NumericClass::Double,
                domain: crate::NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        ));
    }
    infer_concatenated_rows(rows, None)
}

pub fn infer_cell_aggregate(rows: &[Vec<ValueFact>]) -> FactInference {
    let width = rows.first().map(Vec::len).unwrap_or(0);
    if rows.iter().any(|row| row.len() != width) {
        return invalid("RM-TYPE-CELL-LITERAL", "cell rows must have equal widths");
    }
    // MATLAB linear indexing and comma-list expansion are column-major.
    let elements = (0..width)
        .flat_map(|column| rows.iter().map(move |row| row[column].clone()))
        .collect::<Vec<_>>();
    let element = elements
        .iter()
        .cloned()
        .reduce(|left, right| left.join(&right))
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    FactInference::exact(ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements,
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(rows.len()), Some(width)]),
        StorageFact::Dense,
    ))
}

pub fn infer_struct(fields: BTreeMap<String, ValueFact>) -> FactInference {
    FactInference::exact(ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete: true,
    })))
}

pub fn infer_concatenate(dimension: usize, values: &[ValueFact]) -> FactInference {
    if dimension == 0 {
        return invalid(
            "RM-TYPE-CONCAT",
            "concatenation dimension uses one-based indexing",
        );
    }
    if values.is_empty() {
        return FactInference::exact(ValueFact::proven(
            ValueKindFact::Unknown,
            ShapeFact::from(vec![Some(0), Some(0)]),
            StorageFact::Dense,
        ));
    }
    let rank = values
        .iter()
        .filter_map(|value| value.shape.rank())
        .max()
        .unwrap_or(dimension)
        .max(dimension)
        .max(2);
    let mut output = vec![DimensionFact::Known(1); rank];
    let mut concat_total = Some(0usize);
    for (value_index, value) in values.iter().enumerate() {
        let dims = dimensions(&value.shape, rank);
        for index in 0..rank {
            if index == dimension - 1 {
                concat_total = match (concat_total, &dims[index]) {
                    (Some(total), DimensionFact::Known(value)) => total.checked_add(*value),
                    _ => None,
                };
                continue;
            }
            if value_index == 0 {
                output[index] = dims[index].clone();
            } else if output[index] != dims[index] {
                if let (DimensionFact::Known(left), DimensionFact::Known(right)) =
                    (&output[index], &dims[index])
                {
                    return FactInference {
                        fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
                        diagnostics: vec![InferenceDiagnostic::error(
                            "RM-TYPE-CONCAT",
                            format!(
                                "non-concatenated dimension {} disagrees across inputs ({left} versus {right})",
                                index + 1
                            ),
                        )
                        .at_dimension(index)],
                    };
                }
                output[index] = DimensionFact::Unknown;
            }
        }
    }
    output[dimension - 1] = concat_total.map_or(DimensionFact::Unknown, DimensionFact::Known);
    let kind = if values.iter().all(|value| value.kind == values[0].kind) {
        values[0].kind.clone()
    } else {
        ValueKindFact::Unknown
    };
    let storage = if values
        .iter()
        .all(|value| matches!(value.storage, StorageFact::Sparse))
    {
        StorageFact::Sparse
    } else if values
        .iter()
        .all(|value| matches!(value.storage, StorageFact::Dense | StorageFact::Scalar))
    {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    FactInference::exact(ValueFact::proven(
        kind,
        ShapeFact::Shaped { dims: output },
        storage,
    ))
}

pub fn infer_reshape(source: &ValueFact, dimensions: Vec<DimensionFact>) -> FactInference {
    match reshape_shape(&source.shape, dimensions) {
        Ok(shape) => FactInference::exact(reshaped(source, shape)),
        Err(diagnostic) => FactInference {
            fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
            diagnostics: vec![diagnostic],
        },
    }
}

pub fn infer_repmat(source: &ValueFact, repetitions: &[DimensionFact]) -> FactInference {
    FactInference::exact(reshaped(source, repmat_shape(&source.shape, repetitions)))
}

pub fn infer_reduction(source: &ValueFact, dimension: Option<usize>) -> FactInference {
    if dimension == Some(0) {
        return invalid(
            "RM-TYPE-REDUCTION-DIMENSION",
            "reduction dimensions use one-based indexing",
        );
    }
    FactInference::exact(reshaped(source, reduction_shape(&source.shape, dimension)))
}

fn infer_concatenated_rows(
    rows: &[Vec<ValueFact>],
    _expected_kind: Option<&ValueKindFact>,
) -> FactInference {
    let horizontal = rows
        .iter()
        .map(|row| infer_concatenate(2, row))
        .collect::<Vec<_>>();
    if let Some(failure) = horizontal
        .iter()
        .find(|inference| !inference.diagnostics.is_empty())
    {
        return failure.clone();
    }
    infer_concatenate(
        1,
        &horizontal
            .into_iter()
            .map(|inference| inference.fact)
            .collect::<Vec<_>>(),
    )
}

fn dimensions(shape: &ShapeFact, rank: usize) -> Vec<DimensionFact> {
    let mut dims = match shape {
        ShapeFact::Unknown => vec![DimensionFact::Unknown; rank],
        ShapeFact::Scalar => vec![DimensionFact::Known(1), DimensionFact::Known(1)],
        ShapeFact::Ranked { rank } => vec![DimensionFact::Unknown; *rank],
        ShapeFact::Shaped { dims } => dims.clone(),
    };
    dims.resize(rank, DimensionFact::Known(1));
    dims
}

fn reshaped(source: &ValueFact, shape: ShapeFact) -> ValueFact {
    let mut output = source.clone();
    output.shape = shape;
    output.view = crate::ViewFact::Materialized;
    output.alias = crate::AliasFact::Unique;
    output.contiguity = crate::ContiguityFact::Contiguous;
    output
}

fn invalid(code: &str, message: &str) -> FactInference {
    FactInference {
        fact: ValueFact::unknown(DynamicReason::ConflictingControlFlow),
        diagnostics: vec![InferenceDiagnostic::error(code, message)],
    }
}
