//! Exact, typed identity for constants captured by a fusion plan.
//!
//! A value fact cannot distinguish signed zeros, NaN payloads, or two
//! different values of the same class and shape. The digest here is source
//! identity only; it does not authorize a backend literal conversion.

use anyhow::{anyhow, Result};
use runmat_execution::Digest;
use runmat_value::{IntValue, NumericDType, NumericScalar, Value};
use serde::Serialize;

#[derive(Serialize)]
enum ScalarBits {
    F64(u64),
    F32(u32),
    Integer(IntValue),
}

impl From<NumericScalar> for ScalarBits {
    fn from(value: NumericScalar) -> Self {
        match value {
            NumericScalar::F64(value) => Self::F64(value.to_bits()),
            NumericScalar::F32(value) => Self::F32(value.to_bits()),
            NumericScalar::I8(value) => Self::Integer(IntValue::I8(value)),
            NumericScalar::I16(value) => Self::Integer(IntValue::I16(value)),
            NumericScalar::I32(value) => Self::Integer(IntValue::I32(value)),
            NumericScalar::I64(value) => Self::Integer(IntValue::I64(value)),
            NumericScalar::U8(value) => Self::Integer(IntValue::U8(value)),
            NumericScalar::U16(value) => Self::Integer(IntValue::U16(value)),
            NumericScalar::U32(value) => Self::Integer(IntValue::U32(value)),
            NumericScalar::U64(value) => Self::Integer(IntValue::U64(value)),
        }
    }
}

#[derive(Serialize)]
enum ConstantPayload<'a> {
    Num(u64),
    Int(&'a IntValue),
    Complex(u64, u64),
    Bool(bool),
    String(&'a str),
    StringArray {
        shape: &'a [usize],
        data: &'a [String],
    },
    CharArray {
        shape: &'a [usize],
        data: &'a [char],
    },
    LogicalArray {
        shape: &'a [usize],
        data: &'a [u8],
    },
    Tensor {
        shape: &'a [usize],
        dtype: NumericDType,
        data: Vec<ScalarBits>,
    },
    ComplexTensor {
        shape: &'a [usize],
        dtype: NumericDType,
        data: Vec<(ScalarBits, ScalarBits)>,
    },
    SparseReal {
        rows: usize,
        cols: usize,
        col_ptrs: &'a [usize],
        row_indices: &'a [usize],
        logical: bool,
        dtype: Option<NumericDType>,
        data: Vec<ScalarBits>,
    },
}

pub(crate) fn exact_constant_digest(value: &Value) -> Result<Digest> {
    let payload = match value {
        Value::Num(value) => ConstantPayload::Num(value.to_bits()),
        Value::Int(value) => ConstantPayload::Int(value),
        Value::Complex(real, imaginary) => {
            ConstantPayload::Complex(real.to_bits(), imaginary.to_bits())
        }
        Value::Bool(value) => ConstantPayload::Bool(*value),
        Value::String(value) => ConstantPayload::String(value),
        Value::StringArray(array) => ConstantPayload::StringArray {
            shape: &array.shape,
            data: &array.data,
        },
        Value::CharArray(array) => ConstantPayload::CharArray {
            shape: &array.shape,
            data: &array.data,
        },
        Value::LogicalArray(array) => ConstantPayload::LogicalArray {
            shape: &array.shape,
            data: &array.data,
        },
        Value::Tensor(tensor) => ConstantPayload::Tensor {
            shape: tensor.shape(),
            dtype: tensor.numeric_dtype(),
            data: (0..tensor.len())
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .map(ScalarBits::from)
                        .ok_or_else(|| anyhow!("fusion constant has invalid numeric storage"))
                })
                .collect::<Result<Vec<_>>>()?,
        },
        Value::ComplexTensor(tensor) => ConstantPayload::ComplexTensor {
            shape: tensor.shape(),
            dtype: tensor.numeric_dtype(),
            data: (0..tensor.len())
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .map(|(real, imaginary)| (real.into(), imaginary.into()))
                        .ok_or_else(|| anyhow!("fusion constant has invalid complex storage"))
                })
                .collect::<Result<Vec<_>>>()?,
        },
        Value::SparseTensor(tensor) if !tensor.is_complex() => ConstantPayload::SparseReal {
            rows: tensor.rows,
            cols: tensor.cols,
            col_ptrs: &tensor.col_ptrs,
            row_indices: &tensor.row_indices,
            logical: tensor.is_logical(),
            dtype: tensor.numeric_dtype(),
            data: (0..tensor.nnz())
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .map(ScalarBits::from)
                        .ok_or_else(|| anyhow!("fusion constant has invalid sparse storage"))
                })
                .collect::<Result<Vec<_>>>()?,
        },
        _ => return Err(anyhow!("fusion constant has no exact source identity")),
    };
    Ok(Digest::sha256(serde_json::to_vec(&payload)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{ComplexTensor, IntegerStorage, SparseTensor, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn numeric_constant_identity_includes_rank_and_exact_native_storage() {
        let scalar = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), Vec::new()).unwrap(),
        );
        let vector = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1]).unwrap(),
        );
        assert_ne!(
            exact_constant_digest(&scalar).unwrap(),
            exact_constant_digest(&vector).unwrap()
        );
        let complex =
            Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.0, -0.0)], Vec::new()).unwrap());
        let matrix =
            Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.0, -0.0)], vec![1, 1]).unwrap());
        assert_ne!(
            exact_constant_digest(&complex).unwrap(),
            exact_constant_digest(&matrix).unwrap()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn empty_numeric_payload_identity_retains_storage_class() {
        let shape = vec![0, 1];
        let double = Value::Tensor(Tensor::new(Vec::new(), shape.clone()).unwrap());
        let single = Value::Tensor(Tensor::from_f32(Vec::new(), shape.clone()).unwrap());
        let integer = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(Vec::new()), shape.clone()).unwrap(),
        );
        assert_ne!(
            exact_constant_digest(&double).unwrap(),
            exact_constant_digest(&single).unwrap()
        );
        assert_ne!(
            exact_constant_digest(&single).unwrap(),
            exact_constant_digest(&integer).unwrap()
        );

        let complex_double =
            Value::ComplexTensor(ComplexTensor::new(Vec::new(), shape.clone()).unwrap());
        let complex_single =
            Value::ComplexTensor(ComplexTensor::from_f32(Vec::new(), shape).unwrap());
        assert_ne!(
            exact_constant_digest(&complex_double).unwrap(),
            exact_constant_digest(&complex_single).unwrap()
        );

        let sparse_double = Value::SparseTensor(
            SparseTensor::new(0, 1, vec![0, 0], Vec::<usize>::new(), Vec::new()).unwrap(),
        );
        let sparse_single = Value::SparseTensor(
            SparseTensor::new_f32(0, 1, vec![0, 0], Vec::<usize>::new(), Vec::new()).unwrap(),
        );
        let sparse_logical = Value::SparseTensor(
            SparseTensor::new_logical(0, 1, vec![0, 0], Vec::<usize>::new()).unwrap(),
        );
        assert_ne!(
            exact_constant_digest(&sparse_double).unwrap(),
            exact_constant_digest(&sparse_single).unwrap()
        );
        assert_ne!(
            exact_constant_digest(&sparse_single).unwrap(),
            exact_constant_digest(&sparse_logical).unwrap()
        );
    }
}
