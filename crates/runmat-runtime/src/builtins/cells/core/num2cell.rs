//! MATLAB-compatible `num2cell` builtin.

use std::collections::HashSet;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, CellArray, CharArray, ComplexTensor, IntValue,
    IntegerComplexStorage, LogicalArray, NumericScalar, NumericStorage, StringArray, Tensor, Value,
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

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer element or grouped block retains A's native integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "dims",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dimension selectors are exact positive integer scalars/vectors; duplicate and out-of-range dimensions reject.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = num2cell(A, dims)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Scalar cells and grouped numeric blocks preserve exact native storage; resident inputs gather before cell construction.",
    }];

#[runtime_builtin(
    name = "num2cell",
    category = "cells/core",
    summary = "Convert an array into a cell array, optionally grouping dimensions.",
    keywords = "num2cell,cell,conversion,array,dimensions",
    accel = "gather",
    descriptor(crate::builtins::cells::core::num2cell::NUM2CELL_DESCRIPTOR),
    integer_capabilities(crate::builtins::cells::core::num2cell::INTEGER_CAPABILITIES),
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
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            let storage = tensor
                .into_numeric_storage()
                .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))?;
            num2cell_numeric(shape, storage, grouped_dims)
        }
        Value::ComplexTensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return num2cell_typed_complex_integer(
                    tensor.shape.clone(),
                    storage.clone(),
                    grouped_dims,
                );
            }
            num2cell_array(
                tensor.shape.clone(),
                grouped_dims,
                |coords| {
                    let (re, im) =
                        tensor.materialize_f64()[linear_col_major(coords, &tensor.shape)];
                    Value::Complex(re, im)
                },
                |shape, coords| {
                    let data = coords
                        .iter()
                        .map(|coord| {
                            tensor.materialize_f64()[linear_col_major(coord, &tensor.shape)]
                        })
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

fn num2cell_numeric(
    shape: Vec<usize>,
    storage: NumericStorage,
    grouped_dims: &[usize],
) -> BuiltinResult<Value> {
    num2cell_array(
        shape.clone(),
        grouped_dims,
        |coords| {
            let index = linear_col_major(coords, &shape);
            numeric_cell_value(&storage, vec![1, 1], &[index]).expect("valid numeric scalar cell")
        },
        |slice_shape, coords| {
            let indices = coords
                .iter()
                .map(|coord| linear_col_major(coord, &shape))
                .collect::<Vec<_>>();
            numeric_cell_value(&storage, slice_shape.to_vec(), &indices)
        },
    )
}

fn numeric_cell_value(
    storage: &NumericStorage,
    shape: Vec<usize>,
    indices: &[usize],
) -> BuiltinResult<Value> {
    let storage = storage
        .gather(indices)
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))?;
    let preserve_single = matches!(storage, NumericStorage::F32(_));
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|err| error(&ERROR_INTERNAL, format!("num2cell: {err}")))?;
    Ok(if preserve_single {
        Value::Tensor(tensor)
    } else {
        crate::builtins::common::tensor::tensor_into_value(tensor)
    })
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

    let grouped_axes = validate_grouped_dims(grouped_dims, shape.len())?;
    let grouped = grouped_axes.iter().copied().collect::<HashSet<_>>();
    let mut sorted_axes = grouped_axes.clone();
    sorted_axes.sort_unstable();

    let mut output_shape = Vec::with_capacity(shape.len());
    let mut slice_shape = vec![1; shape.len()];
    for (axis, &extent) in shape.iter().enumerate() {
        if grouped.contains(&axis) {
            output_shape.push(1);
        } else {
            output_shape.push(extent);
        }
    }
    let mut source_to_slice_axis = vec![None; shape.len()];
    for (&source_axis, &slice_axis) in grouped_axes.iter().zip(sorted_axes.iter()) {
        slice_shape[slice_axis] = shape[source_axis];
        source_to_slice_axis[source_axis] = Some(slice_axis);
    }

    let total = output_shape.iter().product::<usize>();
    let mut cells = Vec::with_capacity(total);
    for row_major in 0..total {
        let output_coords = coords_row_major(row_major, &output_shape);
        let slice_coords =
            slice_coords_for_output(&shape, &slice_shape, &source_to_slice_axis, &output_coords)?;
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

fn validate_grouped_dims(grouped_dims: &[usize], rank: usize) -> BuiltinResult<Vec<usize>> {
    let mut axes = Vec::with_capacity(grouped_dims.len());
    let mut seen = HashSet::with_capacity(grouped_dims.len());
    for &dim in grouped_dims {
        let axis = dim
            .checked_sub(1)
            .filter(|&axis| axis < rank)
            .ok_or_else(|| {
                error(
                    &ERROR_INVALID_INPUT,
                    format!("num2cell: dims must be between 1 and {rank}"),
                )
            })?;
        if !seen.insert(axis) {
            return Err(error(
                &ERROR_INVALID_INPUT,
                "num2cell: dims must not contain duplicates",
            ));
        }
        axes.push(axis);
    }
    Ok(axes)
}

fn slice_coords_for_output(
    shape: &[usize],
    slice_shape: &[usize],
    source_to_slice_axis: &[Option<usize>],
    output_coords: &[usize],
) -> BuiltinResult<Vec<Vec<usize>>> {
    let total = slice_shape.iter().product::<usize>();
    let mut coords = Vec::with_capacity(total);
    for slice_linear in 0..total {
        let slice_coords = coords_col_major(slice_linear, slice_shape);
        let mut full = Vec::with_capacity(shape.len());
        for axis in 0..shape.len() {
            if let Some(slice_axis) = source_to_slice_axis[axis] {
                full.push(slice_coords[slice_axis]);
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
        Value::Int(value) => dims.push(parse_integer_dim(value)?),
        Value::Tensor(tensor) => {
            for index in 0..tensor.len() {
                match tensor.numeric_value_at(index).ok_or_else(|| {
                    error(
                        &ERROR_INVALID_INPUT,
                        "num2cell: dims must be a positive integer scalar or vector",
                    )
                })? {
                    NumericScalar::F64(value) => dims.push(parse_dim(value)?),
                    NumericScalar::F32(value) => dims.push(parse_dim(f64::from(value))?),
                    value => {
                        dims.push(parse_integer_dim(
                            &value
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer"),
                        )?);
                    }
                }
            }
        }
        _ => {
            return Err(error(
                &ERROR_INVALID_INPUT,
                "num2cell: dims must be a positive integer scalar or vector",
            ))
        }
    }
    Ok(dims)
}

fn parse_dim(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite()
        || value < 1.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        return Err(error(
            &ERROR_INVALID_INPUT,
            "num2cell: dims must contain positive integers",
        ));
    }
    Ok(value as usize)
}

fn parse_integer_dim(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|value| *value >= 1)
        .ok_or_else(|| {
            error(
                &ERROR_INVALID_INPUT,
                "num2cell: dims must contain positive integers",
            )
        })
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
    use runmat_builtins::IntegerStorage;

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
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 5.0]);
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
                assert_eq!(t.materialize_f64(), vec![3.0, 4.0]);
            }
            other => panic!("expected column slice tensor, got {other:?}"),
        }

        let error = block_on(num2cell_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            vec![Value::Num(3.0)],
        ))
        .err()
        .expect("dimension beyond ndims must fail");
        assert!(error.message().contains("between 1 and 2"));
    }

    #[test]
    fn nonsorted_grouped_dims_permute_cell_dimensions_in_requested_order() {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let output = block_on(num2cell_builtin(
            Value::Tensor(matrix),
            vec![Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![2, 1]), vec![1, 2]).unwrap(),
            )],
        ))
        .unwrap();
        let Value::Cell(output) = output else {
            panic!("expected cell array");
        };
        assert_eq!(output.shape, vec![1, 1]);
        let Value::Tensor(block) = output.get(0, 0).unwrap() else {
            panic!("expected grouped tensor");
        };
        assert_eq!(block.shape, vec![3, 2]);
        assert_eq!(block.materialize_f64(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn duplicate_grouped_dims_error() {
        let error = block_on(num2cell_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            vec![Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![1, 1]), vec![1, 2]).unwrap(),
            )],
        ))
        .err()
        .expect("duplicate dimensions must fail");
        assert!(error.message().contains("must not contain duplicates"));
    }

    #[test]
    fn typed_integer_cells_preserve_scalars_and_grouped_blocks() {
        let matrix = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1, u64::MAX, 7]),
            vec![2, 2],
        )
        .unwrap();
        let scalars =
            block_on(num2cell_builtin(Value::Tensor(matrix.clone()), Vec::new())).unwrap();
        let Value::Cell(scalars) = scalars else {
            panic!("expected scalar cells");
        };
        assert_eq!(
            scalars.get(0, 1).unwrap(),
            Value::Int(IntValue::U64(u64::MAX))
        );

        let grouped = block_on(num2cell_builtin(
            Value::Tensor(matrix),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .unwrap();
        let Value::Cell(grouped) = grouped else {
            panic!("expected grouped cells");
        };
        let Value::Tensor(block) = &grouped.data[0] else {
            panic!("expected typed integer block");
        };
        assert_eq!(
            block.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX - 2, u64::MAX - 1]))
        );
    }

    #[test]
    fn real_cells_preserve_single_and_every_integer_class() {
        let single = Tensor::from_f32(vec![1.25, 2.5], vec![1, 2]).unwrap();
        let output = block_on(num2cell_builtin(Value::Tensor(single), Vec::new())).unwrap();
        let Value::Cell(output) = output else {
            panic!("expected single cells");
        };
        for (index, expected) in [1.25f32, 2.5].into_iter().enumerate() {
            let Value::Tensor(value) = &output.data[index] else {
                panic!("single scalar must retain tensor class");
            };
            assert_eq!(
                value.clone().into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![expected])
            );
        }

        let cases = vec![
            (IntegerStorage::I8(vec![-1]), IntValue::I8(-1)),
            (IntegerStorage::I16(vec![-2]), IntValue::I16(-2)),
            (IntegerStorage::I32(vec![-3]), IntValue::I32(-3)),
            (IntegerStorage::I64(vec![i64::MIN]), IntValue::I64(i64::MIN)),
            (IntegerStorage::U8(vec![1]), IntValue::U8(1)),
            (IntegerStorage::U16(vec![2]), IntValue::U16(2)),
            (IntegerStorage::U32(vec![3]), IntValue::U32(3)),
            (IntegerStorage::U64(vec![u64::MAX]), IntValue::U64(u64::MAX)),
        ];
        for (storage, expected) in cases {
            let input = Tensor::new_integer(storage, vec![1, 1]).unwrap();
            let output = block_on(num2cell_builtin(Value::Tensor(input), Vec::new())).unwrap();
            let Value::Cell(output) = output else {
                panic!("expected integer cell");
            };
            assert_eq!(output.data, vec![Value::Int(expected)]);
        }
    }

    #[test]
    fn negative_typed_dims_error() {
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let dims = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap();
        let err = block_on(num2cell_builtin(
            Value::Tensor(input),
            vec![Value::Tensor(dims)],
        ))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("positive integers"),
            "unexpected error message: {err}"
        );

        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = block_on(num2cell_builtin(
            Value::Tensor(input),
            vec![Value::Num(boundary)],
        ))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("positive integers"),
            "unexpected error message: {err}"
        );
    }
}
