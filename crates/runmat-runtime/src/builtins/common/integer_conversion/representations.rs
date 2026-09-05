use runmat_types::IntegerClass;
use runmat_value::{ComplexTensor, IntegerComplexStorage, SymbolicArray, Tensor, Value};

use super::class::IntegerClassExt;
use super::error::{CastError, UnsupportedValueKind};
use super::storage::integer_values;

pub(super) fn cast_symbolic_array(
    target: IntegerClass,
    array: SymbolicArray,
) -> Result<Value, CastError> {
    let values = array
        .data
        .into_iter()
        .map(|expression| {
            expression
                .numeric_constant_value()
                .map(|value| target.cast_scalar(value))
                .ok_or(CastError::Unsupported(UnsupportedValueKind::Symbolic))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Tensor::new_integer(target.storage(values), array.shape)
        .map(Value::Tensor)
        .map_err(CastError::Internal)
}

pub(crate) fn cast_sparse_value(
    target: IntegerClass,
    sparse: runmat_value::SparseTensor,
) -> Result<Value, CastError> {
    let values = match sparse.integer_storage() {
        Some(storage) => storage
            .exact_values()
            .iter()
            .map(|value| target.cast_int(value))
            .collect(),
        None => sparse
            .materialize_f64()
            .iter()
            .map(|&value| target.cast_scalar(value))
            .collect(),
    };
    runmat_value::SparseTensor::new_integer(
        sparse.rows,
        sparse.cols,
        sparse.col_ptrs,
        sparse.row_indices,
        target.storage(values),
    )
    .map(Value::SparseTensor)
    .map_err(CastError::Internal)
}

pub(crate) fn cast_complex_value(value: Value, target: IntegerClass) -> Result<Value, CastError> {
    let (real, imag, shape) = match value {
        Value::Complex(real, imag) => (
            vec![target.cast_scalar(real)],
            vec![target.cast_scalar(imag)],
            vec![1, 1],
        ),
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape.clone();
            if let Some(storage) = tensor.integer_storage() {
                (
                    integer_values(storage.real.clone())
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                    integer_values(storage.imag.clone())
                        .iter()
                        .map(|value| target.cast_int(value))
                        .collect(),
                    shape,
                )
            } else {
                let (real, imag) = tensor
                    .materialize_f64()
                    .into_iter()
                    .map(|(real, imag)| (target.cast_scalar(real), target.cast_scalar(imag)))
                    .unzip();
                (real, imag, shape)
            }
        }
        _ => return Err(CastError::Unsupported(UnsupportedValueKind::Complex)),
    };

    let storage = IntegerComplexStorage::new(target.storage(real), target.storage(imag))
        .map_err(CastError::Internal)?;
    ComplexTensor::new_integer(storage, shape)
        .map(Value::ComplexTensor)
        .map_err(CastError::Internal)
}

pub(super) fn cast_tensor_value(target: IntegerClass, tensor: Tensor) -> Result<Value, CastError> {
    let tensor = target.cast_tensor(tensor).map_err(CastError::Internal)?;
    if !crate::builtins::common::tensor::is_scalar_tensor(&tensor) {
        return Ok(Value::Tensor(tensor));
    }
    let scalar = tensor
        .integer_storage()
        .and_then(|storage| storage.value_at(0))
        .ok_or_else(|| {
            CastError::Internal("scalar integer conversion produced invalid storage".into())
        })?;
    Ok(Value::Int(scalar))
}
