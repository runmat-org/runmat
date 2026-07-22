//! MATLAB-compatible `num2cell` builtin.

use std::collections::HashSet;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ComplexTensor, IntegerComplexStorage, LogicalArray, StringArray, Tensor,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "num2cell";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array result.",
}];

const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const INPUTS_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Dimensions to preserve inside each cell.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "C = num2cell(A)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = num2cell(A, dims)",
        inputs: &INPUTS_DIMS,
        outputs: &OUTPUT,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2CELL.INVALID_INPUT",
    identifier: Some("RunMat:num2cell:InvalidInput"),
    when: "Input value or dimension selector is unsupported.",
    message: "num2cell: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2CELL.INTERNAL",
    identifier: Some("RunMat:num2cell:Internal"),
    when: "Internal cell or slice allocation failed.",
    message: "num2cell: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const NUM2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "num2cell",
    category = "cells/core",
    summary = "Convert an array into a cell array, optionally grouping dimensions.",
    keywords = "num2cell,cell,conversion,array,dimensions",
    accel = "gather",
    descriptor(crate::builtins::cells::core::num2cell::NUM2CELL_DESCRIPTOR),
    builtin_path = "crate::builtins::cells::core::num2cell"
)]
async fn num2cell_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "num2cell: expected A or A,dims",
        ));
    }
    let value = gather_if_needed_async(&value).await?;
    let dims = rest
        .first()
        .map(parse_dims)
        .transpose()?
        .unwrap_or_default();
    num2cell_value(value, &dims)
}

fn num2cell_value(value: Value, grouped_dims: &[usize]) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => num2cell_array(
            tensor.shape.clone(),
            grouped_dims,
            |coords| Value::Num(tensor.data[linear_col_major(coords, &tensor.shape)]),
            |shape, coords| {
                let data = coords
                    .iter()
                    .map(|coord| tensor.data[linear_col_major(coord, &tensor.shape)])
                    .collect();
                Tensor::new(data, shape.to_vec())
                    .map(Value::Tensor)
                    .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
            },
        ),
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = tensor.integer_data {
                return num2cell_typed_complex_integer(tensor.shape, storage, grouped_dims);
            }
            num2cell_array(
                tensor.shape.clone(),
                grouped_dims,
                |coords| {
                    let (re, im) = tensor.data[linear_col_major(coords, &tensor.shape)];
                    Value::Complex(re, im)
                },
                |shape, coords| {
                    let data = coords
                        .iter()
                        .map(|coord| tensor.data[linear_col_major(coord, &tensor.shape)])
                        .collect();
                    ComplexTensor::new(data, shape.to_vec())
                        .map(Value::ComplexTensor)
                        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
                },
            )
        }
        Value::LogicalArray(array) => num2cell_array(
            array.shape.clone(),
            grouped_dims,
            |coords| Value::Bool(array.data[linear_col_major(coords, &array.shape)] != 0),
            |shape, coords| {
                let data = coords
                    .iter()
                    .map(|coord| array.data[linear_col_major(coord, &array.shape)])
                    .collect();
                LogicalArray::new(data, shape.to_vec())
                    .map(Value::LogicalArray)
                    .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
            },
        ),
        Value::StringArray(array) => num2cell_array(
            array.shape.clone(),
            grouped_dims,
            |coords| Value::String(array.data[linear_col_major(coords, &array.shape)].clone()),
            |shape, coords| {
                let data = coords
                    .iter()
                    .map(|coord| array.data[linear_col_major(coord, &array.shape)].clone())
                    .collect();
                StringArray::new(data, shape.to_vec())
                    .map(Value::StringArray)
                    .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
            },
        ),
        Value::CharArray(array) => {
            let shape = vec![array.rows, array.cols];
            num2cell_array(
                shape.clone(),
                grouped_dims,
                |coords| {
                    Value::CharArray(
                        CharArray::new(vec![array.data[coords[0] * array.cols + coords[1]]], 1, 1)
                            .unwrap(),
                    )
                },
                |slice_shape, coords| {
                    let rows = *slice_shape.first().unwrap_or(&1);
                    let cols = *slice_shape.get(1).unwrap_or(&1);
                    let data = coords
                        .iter()
                        .map(|coord| array.data[coord[0] * array.cols + coord[1]])
                        .collect();
                    CharArray::new(data, rows, cols)
                        .map(Value::CharArray)
                        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
                },
            )
        }
        scalar if grouped_dims.is_empty() => CellArray::new(vec![scalar], 1, 1)
            .map(Value::Cell)
            .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}"))),
        _ => Err(error(
            &ERROR_INVALID_INPUT,
            "num2cell: grouped dimensions require an array input",
        )),
    }
}

fn num2cell_typed_complex_integer(
    shape: Vec<usize>,
    storage: IntegerComplexStorage,
    grouped_dims: &[usize],
) -> BuiltinResult<Value> {
    num2cell_array(
        shape.clone(),
        grouped_dims,
        |coords| {
            let index = linear_col_major(coords, &shape);
            typed_complex_integer_cell_value(&storage, vec![1, 1], [index])
                .expect("valid typed complex scalar cell")
        },
        |slice_shape, coords| {
            let indices = coords.iter().map(|coord| linear_col_major(coord, &shape));
            typed_complex_integer_cell_value(&storage, slice_shape.to_vec(), indices)
        },
    )
}

fn typed_complex_integer_cell_value(
    storage: &IntegerComplexStorage,
    shape: Vec<usize>,
    indices: impl IntoIterator<Item = usize>,
) -> BuiltinResult<Value> {
    let indices = indices.into_iter().collect::<Vec<_>>();
    let real = storage
        .real
        .from_exact_values_like(
            indices
                .iter()
                .map(|&index| storage.real.value_at(index).expect("index is in bounds"))
                .collect(),
        )
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))?;
    let imag = storage
        .imag
        .from_exact_values_like(
            indices
                .iter()
                .map(|&index| storage.imag.value_at(index).expect("index is in bounds"))
                .collect(),
        )
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))?;
    let storage = IntegerComplexStorage::new(real, imag)
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))?;
    ComplexTensor::new_integer(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
}

fn num2cell_array<F, G>(
    shape: Vec<usize>,
    grouped_dims: &[usize],
    scalar_at: F,
    slice_at: G,
) -> BuiltinResult<Value>
where
    F: Fn(&[usize]) -> Value,
    G: Fn(&[usize], &[Vec<usize>]) -> BuiltinResult<Value>,
{
    if grouped_dims.is_empty() {
        return scalar_cell_array(shape, scalar_at);
    }

    let grouped = grouped_dim_set(grouped_dims, shape.len());
    if grouped.is_empty() {
        return scalar_cell_array(shape, scalar_at);
    }

    let mut output_shape = Vec::with_capacity(shape.len());
    let mut slice_shape = Vec::with_capacity(shape.len());
    for (axis, &extent) in shape.iter().enumerate() {
        if grouped.contains(&axis) {
            output_shape.push(1);
            slice_shape.push(extent);
        } else {
            output_shape.push(extent);
            slice_shape.push(1);
        }
    }

    let total = output_shape.iter().product::<usize>();
    let mut cells = Vec::with_capacity(total);
    for row_major in 0..total {
        let output_coords = coords_row_major(row_major, &output_shape);
        let slice_coords = slice_coords_for_output(&shape, &grouped, &output_coords)?;
        cells.push(slice_at(&slice_shape, &slice_coords)?);
    }
    CellArray::new_with_shape(cells, output_shape)
        .map(Value::Cell)
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
}

fn scalar_cell_array<F>(shape: Vec<usize>, scalar_at: F) -> BuiltinResult<Value>
where
    F: Fn(&[usize]) -> Value,
{
    let total = shape.iter().product::<usize>();
    let mut cells = Vec::with_capacity(total);
    for row_major in 0..total {
        let coords = coords_row_major(row_major, &shape);
        cells.push(scalar_at(&coords));
    }
    CellArray::new_with_shape(cells, shape)
        .map(Value::Cell)
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))
}

fn grouped_dim_set(grouped_dims: &[usize], rank: usize) -> HashSet<usize> {
    grouped_dims
        .iter()
        .filter_map(|dim| dim.checked_sub(1))
        .filter(|&dim| dim < rank)
        .collect()
}

fn slice_coords_for_output(
    shape: &[usize],
    grouped: &HashSet<usize>,
    output_coords: &[usize],
) -> BuiltinResult<Vec<Vec<usize>>> {
    let slice_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(axis, &extent)| if grouped.contains(&axis) { extent } else { 1 })
        .collect();
    let total = slice_shape.iter().product::<usize>();
    let mut coords = Vec::with_capacity(total);
    for slice_linear in 0..total {
        let slice_coords = coords_col_major(slice_linear, &slice_shape);
        let mut full = Vec::with_capacity(shape.len());
        for axis in 0..shape.len() {
            if grouped.contains(&axis) {
                full.push(slice_coords[axis]);
            } else {
                full.push(output_coords[axis]);
            }
        }
        coords.push(full);
    }
    Ok(coords)
}

fn parse_dims(value: &Value) -> BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    match value {
        Value::Num(value) => dims.push(parse_dim(*value)?),
        Value::Int(value) => dims.push(parse_dim(value.to_f64())?),
        Value::Tensor(tensor) => {
            for &value in &tensor.data {
                dims.push(parse_dim(value)?);
            }
        }
        _ => {
            return Err(error(
                &ERROR_INVALID_INPUT,
                "num2cell: dims must be a positive integer scalar or vector",
            ))
        }
    }
    dims.sort_unstable();
    dims.dedup();
    Ok(dims)
}

fn parse_dim(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "num2cell: dims must contain positive integers",
        ));
    }
    Ok(value as usize)
}

fn coords_row_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let extent = shape[axis];
        coords[axis] = if extent == 0 { 0 } else { linear % extent };
        if extent != 0 {
            linear /= extent;
        }
    }
    coords
}

fn coords_col_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        coords.push(if extent == 0 { 0 } else { linear % extent });
        if extent != 0 {
            linear /= extent;
        }
    }
    coords
}

fn linear_col_major(coords: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (&coord, &extent) in coords.iter().zip(shape.iter()) {
        linear += coord * stride;
        stride *= extent;
    }
    linear
}

fn error(desc: &'static BuiltinErrorDescriptor, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = desc.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn converts_numeric_matrix_to_scalar_cells() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let out = block_on(num2cell_builtin(Value::Tensor(matrix), Vec::new())).unwrap();
        let Value::Cell(cell) = out else {
            panic!("expected cell");
        };
        assert_eq!(cell.shape, vec![2, 2]);
        assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
        assert_eq!(cell.get(0, 1).unwrap(), Value::Num(3.0));
        assert_eq!(cell.get(1, 0).unwrap(), Value::Num(2.0));
        assert_eq!(cell.get(1, 1).unwrap(), Value::Num(4.0));
    }

    #[test]
    fn groups_columns_or_rows_by_dimension() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let by_cols = block_on(num2cell_builtin(
            Value::Tensor(matrix.clone()),
            vec![Value::Num(2.0)],
        ))
        .unwrap();
        let Value::Cell(by_cols) = by_cols else {
            panic!("expected grouped cells");
        };
        assert_eq!(by_cols.shape, vec![2, 1]);
        match by_cols.get(0, 0).unwrap() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![1.0, 3.0, 5.0]);
            }
            other => panic!("expected row slice tensor, got {other:?}"),
        }

        let by_rows = block_on(num2cell_builtin(
            Value::Tensor(matrix),
            vec![Value::Num(1.0)],
        ))
        .unwrap();
        let Value::Cell(by_rows) = by_rows else {
            panic!("expected grouped cells");
        };
        assert_eq!(by_rows.shape, vec![1, 3]);
        match by_rows.get(0, 1).unwrap() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![3.0, 4.0]);
            }
            other => panic!("expected column slice tensor, got {other:?}"),
        }

        let ignored_dim = block_on(num2cell_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            vec![Value::Num(3.0)],
        ))
        .unwrap();
        let Value::Cell(ignored_dim) = ignored_dim else {
            panic!("expected scalar cells");
        };
        assert_eq!(ignored_dim.shape, vec![1, 2]);
        assert_eq!(ignored_dim.get(0, 1).unwrap(), Value::Num(2.0));
    }
}
