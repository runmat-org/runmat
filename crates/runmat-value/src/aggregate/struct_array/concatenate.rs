use super::{column_major_strides, total_len, StructArray, StructValue};
use crate::Value;
use indexmap::IndexMap;
use std::collections::HashSet;

pub enum StructArrayOperand {
    Scalar(StructValue),
    Array(StructArray),
}

struct Columns {
    fields: IndexMap<String, Vec<Value>>,
    shape: Vec<usize>,
}

impl StructArray {
    pub fn concatenate(
        dimension: usize,
        operands: Vec<StructArrayOperand>,
    ) -> Result<Value, String> {
        if operands.is_empty() {
            return Err("structure array concatenation requires at least one input".into());
        }
        let mut inputs = operands.into_iter().map(Columns::from).collect::<Vec<_>>();
        let schema = inputs[0].fields.keys().cloned().collect::<Vec<_>>();
        reorder_schemas(&mut inputs, &schema)?;
        let rank = inputs
            .iter()
            .map(|input| input.shape.len())
            .max()
            .unwrap_or(2)
            .max(dimension + 1)
            .max(2);
        let shapes = padded_shapes(&inputs, rank);
        validate_shapes(&shapes, dimension)?;
        let mut output_shape = shapes[0].clone();
        output_shape[dimension] = shapes.iter().try_fold(0usize, |sum, shape| {
            sum.checked_add(shape[dimension])
                .ok_or_else(|| "structure array concatenation exceeds platform limits".to_string())
        })?;
        let mapping = concatenate_mapping(&shapes, &output_shape, dimension)?;
        let fields = schema
            .into_iter()
            .map(|name| {
                let mut sources = inputs
                    .iter_mut()
                    .map(|input| {
                        Ok(input
                            .fields
                            .shift_remove(&name)
                            .ok_or_else(|| {
                                "structure array concatenation field is missing".to_string()
                            })?
                            .into_iter()
                            .map(Some)
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let values = mapping
                    .iter()
                    .map(|(input, index)| {
                        sources[*input]
                            .get_mut(*index)
                            .and_then(Option::take)
                            .ok_or_else(|| {
                                "structure array concatenation mapping is invalid".to_string()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((name, values))
            })
            .collect::<Result<IndexMap<_, _>, String>>()?;
        Self::normalize_columns(fields, output_shape)
    }
}

impl From<StructArrayOperand> for Columns {
    fn from(operand: StructArrayOperand) -> Self {
        match operand {
            StructArrayOperand::Scalar(value) => Self {
                fields: value
                    .fields
                    .into_iter()
                    .map(|(name, value)| (name, vec![value]))
                    .collect(),
                shape: vec![1, 1],
            },
            StructArrayOperand::Array(array) => Self {
                fields: array.fields,
                shape: array.shape,
            },
        }
    }
}

fn reorder_schemas(inputs: &mut [Columns], schema: &[String]) -> Result<(), String> {
    let expected = schema.iter().collect::<HashSet<_>>();
    for input in inputs {
        let actual = input.fields.keys().collect::<HashSet<_>>();
        if actual != expected || input.fields.len() != schema.len() {
            return Err("structure array concatenation requires one field set".into());
        }
        let mut reordered = IndexMap::with_capacity(schema.len());
        for name in schema {
            let values = input
                .fields
                .shift_remove(name)
                .ok_or_else(|| "structure array concatenation field is missing".to_string())?;
            reordered.insert(name.clone(), values);
        }
        input.fields = reordered;
    }
    Ok(())
}

fn padded_shapes(inputs: &[Columns], rank: usize) -> Vec<Vec<usize>> {
    inputs
        .iter()
        .map(|input| {
            let mut shape = input.shape.clone();
            shape.resize(rank, 1);
            shape
        })
        .collect()
}

fn validate_shapes(shapes: &[Vec<usize>], dimension: usize) -> Result<(), String> {
    for axis in 0..shapes[0].len() {
        if axis != dimension && shapes.iter().any(|shape| shape[axis] != shapes[0][axis]) {
            return Err("structure array concatenation dimensions do not agree".into());
        }
    }
    Ok(())
}

fn concatenate_mapping(
    shapes: &[Vec<usize>],
    output_shape: &[usize],
    dimension: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let length = total_len(output_shape)?;
    let strides = shapes
        .iter()
        .map(|shape| column_major_strides(shape))
        .collect::<Result<Vec<_>, _>>()?;
    (0..length)
        .map(|index| {
            let mut remainder = index;
            let mut coordinates = output_shape
                .iter()
                .map(|extent| {
                    let coordinate = remainder % extent;
                    remainder /= extent;
                    coordinate
                })
                .collect::<Vec<_>>();
            let mut local = coordinates[dimension];
            let input = shapes
                .iter()
                .position(|shape| {
                    if local < shape[dimension] {
                        true
                    } else {
                        local -= shape[dimension];
                        false
                    }
                })
                .ok_or_else(|| "structure array concatenation mapping is invalid".to_string())?;
            coordinates[dimension] = local;
            let source = coordinates
                .iter()
                .zip(&strides[input])
                .try_fold(0usize, |sum, (coordinate, stride)| {
                    coordinate
                        .checked_mul(*stride)
                        .and_then(|offset| sum.checked_add(offset))
                })
                .ok_or_else(|| "structure array concatenation index overflow".to_string())?;
            Ok((input, source))
        })
        .collect()
}
