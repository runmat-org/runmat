use runmat_builtins::{
    BuiltinErrorDescriptor, IND2SUB_ERROR_INVALID_INPUT, SUB2IND_ERROR_INVALID_INPUT,
};
use runmat_value::{IntValue, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone, Copy)]
pub(crate) enum IndexingOperation {
    Ind2Sub,
    Sub2Ind,
}

impl IndexingOperation {
    fn name(self) -> &'static str {
        match self {
            Self::Ind2Sub => "ind2sub",
            Self::Sub2Ind => "sub2ind",
        }
    }

    fn input_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Ind2Sub => &IND2SUB_ERROR_INVALID_INPUT,
            Self::Sub2Ind => &SUB2IND_ERROR_INVALID_INPUT,
        }
    }
}

/// Materialise a value so indexing helpers can operate on host tensors.
pub(crate) async fn materialize_value(
    value: Value,
    _operation: IndexingOperation,
) -> BuiltinResult<(Value, bool)> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
            Ok((Value::Tensor(gathered), true))
        }
        other => Ok((other, false)),
    }
}

/// Parse a MATLAB-style size vector into concrete dimension extents.
pub(crate) async fn parse_dims(
    value: &Value,
    operation: IndexingOperation,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(_) | Value::Int(_) => parse_scalar_dims(value, operation).await,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            parse_scalar_dims(value, operation).await
        }
        Value::GpuTensor(handle) if tensor::element_count(handle.shape()) == 1 => {
            parse_scalar_dims(value, operation).await
        }
        Value::Tensor(_) | Value::GpuTensor(_) => parse_vector_dims(value, operation).await,
        Value::Cell(ca) => {
            if ca.data().is_empty() {
                return Err(indexing_error(
                    operation,
                    "Size vector must have at least one element.",
                ));
            }
            let mut dims = Vec::with_capacity(ca.data().len());
            for cell in ca.data() {
                let coerced = match cell {
                    Value::Num(n) => coerce_positive_int(*n, operation)?,
                    Value::Int(i) => coerce_positive_integer(i, operation)?,
                    _ => {
                        return Err(indexing_error(
                            operation,
                            "Size vector must contain numeric values.",
                        ))
                    }
                };
                dims.push(coerced);
            }
            Ok(dims)
        }
        _ => Err(indexing_error(
            operation,
            "Size vector must be a numeric vector.",
        )),
    }
}

async fn parse_scalar_dims(
    value: &Value,
    operation: IndexingOperation,
) -> BuiltinResult<Vec<usize>> {
    let Some(dim) = tensor::dimension_from_value_async(value, operation.name(), false)
        .await
        .map_err(|_| indexing_error(operation, "Size arguments must be positive integers."))?
    else {
        return Err(indexing_error(
            operation,
            "Size vector must be a numeric vector.",
        ));
    };
    Ok(vec![dim])
}

async fn parse_vector_dims(
    value: &Value,
    operation: IndexingOperation,
) -> BuiltinResult<Vec<usize>> {
    let dims = tensor::dims_from_value_async(value)
        .await
        .map_err(|_| indexing_error(operation, "Size arguments must be positive integers."))?
        .ok_or_else(|| indexing_error(operation, "Size vector must be a row vector."))?;
    if dims.is_empty() {
        return Err(indexing_error(
            operation,
            "Size vector must have at least one element.",
        ));
    }
    if dims.contains(&0) {
        return Err(indexing_error(
            operation,
            "Size arguments must be positive integers.",
        ));
    }
    Ok(dims)
}

/// Coerce a floating-point value into a strictly positive integer.
pub(crate) fn coerce_positive_int(
    value: f64,
    operation: IndexingOperation,
) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(indexing_error(
            operation,
            "Size arguments must be positive integers.",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(indexing_error(
            operation,
            "Size arguments must be positive integers.",
        ));
    }
    if rounded < 1.0 {
        return Err(indexing_error(
            operation,
            "Size arguments must be positive integers.",
        ));
    }
    if !fits_positive_platform_index(rounded) {
        return Err(indexing_error(
            operation,
            "Size arguments exceed the maximum supported size.",
        ));
    }
    Ok(rounded as usize)
}

fn coerce_positive_integer(value: &IntValue, operation: IndexingOperation) -> BuiltinResult<usize> {
    value.try_to_usize().filter(|&dim| dim >= 1).ok_or_else(|| {
        indexing_error(
            operation,
            "Size arguments must be positive integers within the supported range.",
        )
    })
}

pub(crate) fn fits_positive_platform_index(value: f64) -> bool {
    if value > usize::MAX.saturating_sub(1) as f64 {
        return false;
    }
    let parsed = value as usize;
    parsed as f64 == value && parsed != usize::MAX
}

/// Build column-major strides for the supplied dimensions, checking overflow.
pub(crate) fn build_strides(
    dims: &[usize],
    operation: IndexingOperation,
) -> BuiltinResult<Vec<usize>> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride = stride.checked_mul(dim).ok_or_else(|| {
            indexing_error(
                operation,
                "Size vector elements overflow the maximum supported size.",
            )
        })?;
    }
    Ok(strides)
}

/// Compute the total number of elements implied by the size vector.
pub(crate) fn total_elements(dims: &[usize], operation: IndexingOperation) -> BuiltinResult<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            indexing_error(
                operation,
                "Size vector elements overflow the maximum supported size.",
            )
        })
    })
}

fn indexing_error(operation: IndexingOperation, message: impl Into<String>) -> RuntimeError {
    let descriptor = operation.input_error();
    let mut builder = build_runtime_error(message).with_builtin(operation.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::{CellArray, IntegerStorage, Tensor};

    #[test]
    fn parse_dims_accepts_scalar() {
        let dims = block_on(parse_dims(&Value::Num(3.0), IndexingOperation::Ind2Sub))
            .expect("scalar dimension");
        assert_eq!(dims, vec![3]);
    }

    #[test]
    fn parse_dims_accepts_vector() {
        let vector = Tensor::new(vec![2.0, 4.0], vec![1, 2]).expect("size vector");
        let dims = block_on(parse_dims(
            &Value::Tensor(vector),
            IndexingOperation::Sub2Ind,
        ))
        .expect("vector dimensions");
        assert_eq!(dims, vec![2, 4]);
    }

    #[test]
    fn parse_dims_rejects_non_numeric() {
        let values = [
            Value::String("bad".to_string().into()),
            Value::Cell(
                CellArray::new(vec![Value::String("bad".to_string().into())], 1, 1)
                    .expect("cell size vector"),
            ),
        ];
        for value in values {
            assert!(block_on(parse_dims(&value, IndexingOperation::Ind2Sub)).is_err());
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn parse_dims_preserves_typed_cell_dimensions_exactly() {
        let dims = block_on(parse_dims(
            &Value::Cell(
                CellArray::new(
                    vec![
                        Value::Int(IntValue::U8(2)),
                        Value::Int(IntValue::U64(9_007_199_254_740_993)),
                    ],
                    1,
                    2,
                )
                .expect("cell"),
            ),
            IndexingOperation::Ind2Sub,
        ))
        .expect("dimensions");

        assert_eq!(dims, vec![2, 9_007_199_254_740_993]);
        assert!(block_on(parse_dims(
            &Value::Cell(CellArray::new(vec![Value::Int(IntValue::I8(-1))], 1, 1).expect("cell"),),
            IndexingOperation::Ind2Sub,
        ))
        .is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "32")]
    fn parse_dims_rejects_typed_cell_dimensions_outside_platform_range() {
        let value = Value::Cell(
            CellArray::new(
                vec![
                    Value::Int(IntValue::U8(2)),
                    Value::Int(IntValue::U64(9_007_199_254_740_993)),
                ],
                1,
                2,
            )
            .expect("cell"),
        );

        let err = block_on(parse_dims(&value, IndexingOperation::Ind2Sub))
            .expect_err("dimension must fit usize");
        assert!(err.to_string().contains("supported range"));
    }

    #[test]
    fn parse_dims_reads_typed_integer_tensor_storage_exactly() {
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar dim");
        let dims = block_on(parse_dims(
            &Value::Tensor(scalar),
            IndexingOperation::Ind2Sub,
        ))
        .expect("scalar dims");
        assert_eq!(dims, vec![7]);

        let vector =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("vector dims");
        let dims = block_on(parse_dims(
            &Value::Tensor(vector),
            IndexingOperation::Ind2Sub,
        ))
        .expect("vector dims");
        assert_eq!(dims, vec![2, 3]);
    }

    #[test]
    fn numeric_dimension_coercion_rejects_out_of_range_float_values() {
        assert!(coerce_positive_int(usize::MAX as f64, IndexingOperation::Ind2Sub).is_err());
        assert!(block_on(parse_dims(
            &Value::Num(usize::MAX as f64),
            IndexingOperation::Ind2Sub,
        ))
        .is_err());
        for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1.5] {
            assert!(coerce_positive_int(invalid, IndexingOperation::Ind2Sub).is_err());
        }
        assert!(coerce_positive_integer(&IntValue::I64(-1), IndexingOperation::Ind2Sub).is_err());
        #[cfg(target_pointer_width = "32")]
        assert!(
            coerce_positive_integer(&IntValue::U64(u64::MAX), IndexingOperation::Ind2Sub).is_err()
        );
        let vector = Value::Cell(
            CellArray::new(
                vec![Value::Int(IntValue::U16(2)), Value::Int(IntValue::U32(4))],
                1,
                2,
            )
            .expect("integer size vector"),
        );
        assert_eq!(
            block_on(parse_dims(&vector, IndexingOperation::Ind2Sub)).expect("integer dimensions"),
            vec![2, 4]
        );
    }

    #[test]
    fn shared_dimension_preflight_retains_the_operation_error_owner() {
        for (operation, expected) in [
            (IndexingOperation::Ind2Sub, "RunMat:ind2sub:InvalidInput"),
            (IndexingOperation::Sub2Ind, "RunMat:sub2ind:InvalidInput"),
        ] {
            let error = block_on(parse_dims(&Value::Num(-1.0), operation))
                .expect_err("size vector must reject");
            assert_eq!(error.identifier(), Some(expected));
        }
    }
}
