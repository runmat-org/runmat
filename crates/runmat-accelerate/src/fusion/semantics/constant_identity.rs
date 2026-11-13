//! Exact, typed identity for constants captured by a fusion plan.
//!
//! A value fact cannot distinguish signed zeros, NaN payloads, or two
//! different values of the same class and shape. The digest here is source
//! identity only; it does not authorize a backend literal conversion.

use anyhow::{anyhow, Result};
use runmat_execution::Digest;
use runmat_value::{
    ComplexTensor, IntValue, NumericDType, NumericScalar, SparseTensor, StringElement, Tensor,
    Value,
};
use serde::{ser::SerializeSeq, Serialize, Serializer};

#[derive(Serialize)]
enum ScalarBits {
    F64(u64),
    F32(u32),
    Integer(IntValue),
}

struct RealScalars<'a>(&'a Tensor);

impl Serialize for RealScalars<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for index in 0..self.0.len() {
            let value = self.0.numeric_value_at(index).ok_or_else(|| {
                serde::ser::Error::custom("fusion constant has invalid numeric storage")
            })?;
            sequence.serialize_element(&ScalarBits::from(value))?;
        }
        sequence.end()
    }
}

struct ComplexScalars<'a>(&'a ComplexTensor);

impl Serialize for ComplexScalars<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for index in 0..self.0.len() {
            let (real, imaginary) = self.0.numeric_value_at(index).ok_or_else(|| {
                serde::ser::Error::custom("fusion constant has invalid complex storage")
            })?;
            sequence.serialize_element(&(ScalarBits::from(real), ScalarBits::from(imaginary)))?;
        }
        sequence.end()
    }
}

struct SparseScalars<'a>(&'a SparseTensor);

impl Serialize for SparseScalars<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.nnz()))?;
        for index in 0..self.0.nnz() {
            let value = self.0.numeric_value_at(index).ok_or_else(|| {
                serde::ser::Error::custom("fusion constant has invalid sparse storage")
            })?;
            sequence.serialize_element(&ScalarBits::from(value))?;
        }
        sequence.end()
    }
}

struct StringScalars<'a>(&'a [StringElement]);

impl Serialize for StringScalars<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for value in self.0 {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
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
    String(&'a StringElement),
    StringArray {
        shape: &'a [usize],
        data: StringScalars<'a>,
    },
    CharArray {
        shape: &'a [usize],
        data: &'a [runmat_value::Utf16CodeUnit],
    },
    LogicalArray {
        shape: &'a [usize],
        data: &'a [u8],
    },
    Tensor {
        shape: &'a [usize],
        dtype: NumericDType,
        data: RealScalars<'a>,
    },
    ComplexTensor {
        shape: &'a [usize],
        dtype: NumericDType,
        data: ComplexScalars<'a>,
    },
    SparseReal {
        rows: usize,
        cols: usize,
        col_ptrs: &'a [usize],
        row_indices: &'a [usize],
        logical: bool,
        dtype: Option<NumericDType>,
        data: SparseScalars<'a>,
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
            data: StringScalars(array.data()),
        },
        Value::CharArray(array) => ConstantPayload::CharArray {
            shape: &array.shape,
            data: array.data(),
        },
        Value::LogicalArray(array) => ConstantPayload::LogicalArray {
            shape: &array.shape,
            data: &array.data,
        },
        Value::Tensor(tensor) => ConstantPayload::Tensor {
            shape: tensor.shape(),
            dtype: tensor.numeric_dtype(),
            data: RealScalars(tensor),
        },
        Value::ComplexTensor(tensor) => ConstantPayload::ComplexTensor {
            shape: tensor.shape(),
            dtype: tensor.numeric_dtype(),
            data: ComplexScalars(tensor),
        },
        Value::SparseTensor(tensor) if !tensor.is_complex() => ConstantPayload::SparseReal {
            rows: tensor.rows,
            cols: tensor.cols,
            col_ptrs: &tensor.col_ptrs,
            row_indices: &tensor.row_indices,
            logical: tensor.is_logical(),
            dtype: tensor.numeric_dtype(),
            data: SparseScalars(tensor),
        },
        _ => return Err(anyhow!("fusion constant has no exact source identity")),
    };
    Ok(Digest::sha256_serialized(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{ComplexTensor, IntegerStorage, SparseTensor, StringArray, Tensor};

    #[derive(Serialize)]
    enum LegacyStringPayload<'a> {
        StringArray {
            shape: &'a [usize],
            data: &'a [String],
        },
    }

    #[test]
    fn streamed_string_array_identity_matches_original_json_bytes() {
        let strings = vec!["a\"b".to_owned(), "λ\n".to_owned(), String::new()];
        let shape = vec![1, strings.len()];
        let legacy = LegacyStringPayload::StringArray {
            shape: &shape,
            data: &strings,
        };
        let expected = Digest::sha256(serde_json::to_vec(&legacy).unwrap());
        let value = Value::StringArray(StringArray::new(strings, shape).unwrap());
        assert_eq!(exact_constant_digest(&value).unwrap(), expected);
    }

    #[derive(Serialize)]
    enum LegacyNumericPayload<'a> {
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn streamed_numeric_identity_matches_original_json_bytes() {
        let values = [
            Value::Tensor(
                Tensor::new(
                    vec![-0.0, f64::from_bits(0x7ff8_0000_0000_1234)],
                    vec![1, 2],
                )
                .unwrap(),
            ),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ),
            Value::ComplexTensor(
                ComplexTensor::from_f32(vec![(-0.0, f32::from_bits(0x7fc0_1234))], vec![1, 1])
                    .unwrap(),
            ),
            Value::SparseTensor(
                SparseTensor::new(2, 2, vec![0, 1, 1], vec![1], vec![-0.0]).unwrap(),
            ),
        ];
        for value in &values {
            let legacy = match value {
                Value::Tensor(tensor) => LegacyNumericPayload::Tensor {
                    shape: tensor.shape(),
                    dtype: tensor.numeric_dtype(),
                    data: (0..tensor.len())
                        .map(|index| ScalarBits::from(tensor.numeric_value_at(index).unwrap()))
                        .collect(),
                },
                Value::ComplexTensor(tensor) => LegacyNumericPayload::ComplexTensor {
                    shape: tensor.shape(),
                    dtype: tensor.numeric_dtype(),
                    data: (0..tensor.len())
                        .map(|index| {
                            let (real, imaginary) = tensor.numeric_value_at(index).unwrap();
                            (real.into(), imaginary.into())
                        })
                        .collect(),
                },
                Value::SparseTensor(tensor) => LegacyNumericPayload::SparseReal {
                    rows: tensor.rows,
                    cols: tensor.cols,
                    col_ptrs: &tensor.col_ptrs,
                    row_indices: &tensor.row_indices,
                    logical: tensor.is_logical(),
                    dtype: tensor.numeric_dtype(),
                    data: (0..tensor.nnz())
                        .map(|index| ScalarBits::from(tensor.numeric_value_at(index).unwrap()))
                        .collect(),
                },
                _ => unreachable!("numeric identity fixture"),
            };
            assert_eq!(
                exact_constant_digest(value).unwrap(),
                Digest::sha256(serde_json::to_vec(&legacy).unwrap())
            );
        }
    }

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
