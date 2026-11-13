use runmat_value::{ComplexStorage, ComplexTensor, IntegerComplexStorage};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use runmat_filesystem as fs;
use runmat_filesystem::data_contract::{
    DataChunkDescriptor, DataChunkUploadRequest, DataChunkUploadTarget,
};
use runmat_value::{
    ForeignRef, IntValue, IntegerStorage, NumericScalar, NumericStorage, ObjectInstance, Tensor,
    Value,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::builtins::common::integer_conversion::{IntegerClass, IntegerClassExt};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod cooperation;
mod publication;
pub(crate) use cooperation::cooperate_data_work;
pub(crate) use publication::{create_array_revision, relative_storage_path, ChunkStorageIdentity};
pub(crate) mod authority;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManifest {
    pub schema_version: u32,
    pub format: String,
    pub dataset_id: String,
    pub name: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub arrays: BTreeMap<String, DataArrayMeta>,
    pub attrs: BTreeMap<String, serde_json::Value>,
    pub txn_sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataArrayMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub chunk_shape: Vec<usize>,
    #[serde(default = "default_array_order")]
    pub order: String,
    pub codec: String,
    #[serde(default)]
    pub chunk_index_path: Option<String>,
    pub data_path: String,
}

fn default_array_order() -> String {
    "column_major".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataArrayPayload {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub values: DataArrayValues,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imaginary_values: Option<DataArrayValues>,
}

/// The persisted backing values of a data-array payload.
///
/// JSON arrays written before integer storage was introduced are decoded as
/// `F64`; new writes use the tagged representation below so every integer
/// class can round-trip without passing through a floating point value.
#[derive(Debug, Clone, PartialEq)]
pub enum DataArrayValues {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "encoding", content = "data", rename_all = "snake_case")]
enum TaggedDataArrayValues {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[derive(Serialize)]
#[serde(tag = "encoding", content = "data", rename_all = "snake_case")]
enum TaggedDataArrayValuesRef<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DataArrayValuesWire {
    Tagged(TaggedDataArrayValues),
    Legacy(Vec<f64>),
}

impl Serialize for DataArrayValues {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let tagged = match self {
            Self::F64(values) => TaggedDataArrayValuesRef::F64(values),
            Self::F32(values) => TaggedDataArrayValuesRef::F32(values),
            Self::I8(values) => TaggedDataArrayValuesRef::I8(values),
            Self::I16(values) => TaggedDataArrayValuesRef::I16(values),
            Self::I32(values) => TaggedDataArrayValuesRef::I32(values),
            Self::I64(values) => TaggedDataArrayValuesRef::I64(values),
            Self::U8(values) => TaggedDataArrayValuesRef::U8(values),
            Self::U16(values) => TaggedDataArrayValuesRef::U16(values),
            Self::U32(values) => TaggedDataArrayValuesRef::U32(values),
            Self::U64(values) => TaggedDataArrayValuesRef::U64(values),
        };
        tagged.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DataArrayValues {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match DataArrayValuesWire::deserialize(deserializer)? {
            DataArrayValuesWire::Legacy(values) => Self::F64(values),
            DataArrayValuesWire::Tagged(tagged) => match tagged {
                TaggedDataArrayValues::F64(values) => Self::F64(values),
                TaggedDataArrayValues::F32(values) => Self::F32(values),
                TaggedDataArrayValues::I8(values) => Self::I8(values),
                TaggedDataArrayValues::I16(values) => Self::I16(values),
                TaggedDataArrayValues::I32(values) => Self::I32(values),
                TaggedDataArrayValues::I64(values) => Self::I64(values),
                TaggedDataArrayValues::U8(values) => Self::U8(values),
                TaggedDataArrayValues::U16(values) => Self::U16(values),
                TaggedDataArrayValues::U32(values) => Self::U32(values),
                TaggedDataArrayValues::U64(values) => Self::U64(values),
            },
        })
    }
}

impl DataArrayValues {
    pub fn zeros(dtype: &str, len: usize) -> BuiltinResult<Self> {
        if is_single_dtype(dtype) {
            return Ok(Self::F32(vec![0.0; len]));
        }
        Ok(match integer_dtype(dtype) {
            Some("int8") => Self::I8(vec![0; len]),
            Some("int16") => Self::I16(vec![0; len]),
            Some("int32") => Self::I32(vec![0; len]),
            Some("int64") => Self::I64(vec![0; len]),
            Some("uint8") => Self::U8(vec![0; len]),
            Some("uint16") => Self::U16(vec![0; len]),
            Some("uint32") => Self::U32(vec![0; len]),
            Some("uint64") => Self::U64(vec![0; len]),
            None if is_double_dtype(dtype) => Self::F64(vec![0.0; len]),
            _ => return Err(unsupported_data_dtype(dtype)),
        })
    }

    fn append_zeroes(&mut self, count: usize) {
        let next = self.len().checked_add(count).expect("validated data shape");
        match self {
            Self::F64(values) => values.resize(next, 0.0),
            Self::F32(values) => values.resize(next, 0.0),
            Self::I8(values) => values.resize(next, 0),
            Self::I16(values) => values.resize(next, 0),
            Self::I32(values) => values.resize(next, 0),
            Self::I64(values) => values.resize(next, 0),
            Self::U8(values) => values.resize(next, 0),
            Self::U16(values) => values.resize(next, 0),
            Self::U32(values) => values.resize(next, 0),
            Self::U64(values) => values.resize(next, 0),
        }
    }

    pub async fn zeros_cooperative(
        dtype: &str,
        len: usize,
        budget: &mut Option<crate::execution::CpuCooperationBudget>,
    ) -> BuiltinResult<Self> {
        const WORK_CHUNK: usize = 4_096;
        let mut values = Self::zeros(dtype, 0)?;
        let mut remaining = len;
        while remaining != 0 {
            let count = remaining.min(WORK_CHUNK);
            values.append_zeroes(count);
            remaining -= count;
            cooperate_data_work(budget, count as u64).await?;
        }
        Ok(values)
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_tensor(self, shape: Vec<usize>) -> Result<Tensor, String> {
        match self {
            Self::F64(values) => Tensor::new(values, shape),
            Self::F32(values) => Tensor::from_f32(values, shape),
            Self::I8(values) => Tensor::new_integer(IntegerStorage::I8(values), shape),
            Self::I16(values) => Tensor::new_integer(IntegerStorage::I16(values), shape),
            Self::I32(values) => Tensor::new_integer(IntegerStorage::I32(values), shape),
            Self::I64(values) => Tensor::new_integer(IntegerStorage::I64(values), shape),
            Self::U8(values) => Tensor::new_integer(IntegerStorage::U8(values), shape),
            Self::U16(values) => Tensor::new_integer(IntegerStorage::U16(values), shape),
            Self::U32(values) => Tensor::new_integer(IntegerStorage::U32(values), shape),
            Self::U64(values) => Tensor::new_integer(IntegerStorage::U64(values), shape),
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            Self::F64(values) => values.clone(),
            Self::F32(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::I8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::I64(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U8(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U16(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U32(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U64(values) => values.iter().map(|&value| value as f64).collect(),
        }
    }

    /// Returns at most `limit` values converted to the numeric preview format.
    ///
    /// Data-file previews cross the WASM boundary as JavaScript numbers. Keep
    /// that established representation while only converting the requested
    /// prefix: a fallback full-payload read may contain substantially more
    /// values than the preview is allowed to return.
    pub fn preview_f64(&self, limit: usize) -> Vec<f64> {
        match self {
            Self::F64(values) => values.iter().take(limit).copied().collect(),
            Self::F32(values) => values
                .iter()
                .take(limit)
                .map(|&value| f64::from(value))
                .collect(),
            Self::I8(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::I16(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::I32(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::I64(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::U8(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::U16(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::U32(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
            Self::U64(values) => values
                .iter()
                .take(limit)
                .map(|&value| value as f64)
                .collect(),
        }
    }

    pub fn get(&self, index: usize) -> BuiltinResult<DataScalar> {
        match self {
            Self::F64(values) => values.get(index).copied().map(DataScalar::F64),
            Self::F32(values) => values.get(index).copied().map(DataScalar::F32),
            Self::I8(values) => values.get(index).copied().map(|v| DataScalar::I8(v)),
            Self::I16(values) => values.get(index).copied().map(|v| DataScalar::I16(v)),
            Self::I32(values) => values.get(index).copied().map(|v| DataScalar::I32(v)),
            Self::I64(values) => values.get(index).copied().map(|v| DataScalar::I64(v)),
            Self::U8(values) => values.get(index).copied().map(|v| DataScalar::U8(v)),
            Self::U16(values) => values.get(index).copied().map(|v| DataScalar::U16(v)),
            Self::U32(values) => values.get(index).copied().map(|v| DataScalar::U32(v)),
            Self::U64(values) => values.get(index).copied().map(|v| DataScalar::U64(v)),
        }
        .ok_or_else(|| data_error(format!("data payload index {index} is out of bounds")))
    }

    pub fn push(&mut self, value: DataScalar) -> BuiltinResult<()> {
        match (self, value) {
            (Self::F64(values), DataScalar::F64(value)) => values.push(value),
            (Self::F32(values), DataScalar::F32(value)) => values.push(value),
            (Self::I8(values), DataScalar::I8(value)) => values.push(value),
            (Self::I16(values), DataScalar::I16(value)) => values.push(value),
            (Self::I32(values), DataScalar::I32(value)) => values.push(value),
            (Self::I64(values), DataScalar::I64(value)) => values.push(value),
            (Self::U8(values), DataScalar::U8(value)) => values.push(value),
            (Self::U16(values), DataScalar::U16(value)) => values.push(value),
            (Self::U32(values), DataScalar::U32(value)) => values.push(value),
            (Self::U64(values), DataScalar::U64(value)) => values.push(value),
            _ => return Err(data_error("data payload storage class mismatch")),
        }
        Ok(())
    }

    pub fn set(&mut self, index: usize, value: DataScalar) -> BuiltinResult<()> {
        match (self, value) {
            (Self::F64(values), DataScalar::F64(value)) => set_at(values, index, value),
            (Self::F32(values), DataScalar::F32(value)) => set_at(values, index, value),
            (Self::I8(values), DataScalar::I8(value)) => set_at(values, index, value),
            (Self::I16(values), DataScalar::I16(value)) => set_at(values, index, value),
            (Self::I32(values), DataScalar::I32(value)) => set_at(values, index, value),
            (Self::I64(values), DataScalar::I64(value)) => set_at(values, index, value),
            (Self::U8(values), DataScalar::U8(value)) => set_at(values, index, value),
            (Self::U16(values), DataScalar::U16(value)) => set_at(values, index, value),
            (Self::U32(values), DataScalar::U32(value)) => set_at(values, index, value),
            (Self::U64(values), DataScalar::U64(value)) => set_at(values, index, value),
            _ => return Err(data_error("data payload storage class mismatch")),
        }?;
        Ok(())
    }

    fn cast_to_dtype(self, dtype: &str) -> BuiltinResult<Self> {
        if is_single_dtype(dtype) {
            return Ok(Self::F32(
                self.to_f64_vec()
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
            ));
        }
        if is_double_dtype(dtype) {
            return Ok(Self::F64(self.to_f64_vec()));
        }
        let Some(target) = integer_target(dtype) else {
            return Err(unsupported_data_dtype(dtype));
        };
        let mut values = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            let value = self.get(index)?;
            values.push(match value {
                DataScalar::F64(value) => target.cast_scalar(value),
                DataScalar::F32(value) => target.cast_scalar(f64::from(value)),
                value => target.cast_int(&value.to_int_value()),
            });
        }
        Ok(Self::from_integer_storage(target.storage(values)))
    }

    fn from_integer_storage(storage: IntegerStorage) -> Self {
        match storage {
            IntegerStorage::I8(values) => Self::I8(values),
            IntegerStorage::I16(values) => Self::I16(values),
            IntegerStorage::I32(values) => Self::I32(values),
            IntegerStorage::I64(values) => Self::I64(values),
            IntegerStorage::U8(values) => Self::U8(values),
            IntegerStorage::U16(values) => Self::U16(values),
            IntegerStorage::U32(values) => Self::U32(values),
            IntegerStorage::U64(values) => Self::U64(values),
        }
    }

    fn from_numeric_storage(storage: NumericStorage) -> Self {
        match storage {
            NumericStorage::F64(values) => Self::F64(values),
            NumericStorage::F32(values) => Self::F32(values),
            NumericStorage::I8(values) => Self::I8(values),
            NumericStorage::I16(values) => Self::I16(values),
            NumericStorage::I32(values) => Self::I32(values),
            NumericStorage::I64(values) => Self::I64(values),
            NumericStorage::U8(values) => Self::U8(values),
            NumericStorage::U16(values) => Self::U16(values),
            NumericStorage::U32(values) => Self::U32(values),
            NumericStorage::U64(values) => Self::U64(values),
        }
    }

    fn into_numeric_storage(self) -> NumericStorage {
        match self {
            Self::F64(values) => NumericStorage::F64(values),
            Self::F32(values) => NumericStorage::F32(values),
            Self::I8(values) => NumericStorage::I8(values),
            Self::I16(values) => NumericStorage::I16(values),
            Self::I32(values) => NumericStorage::I32(values),
            Self::I64(values) => NumericStorage::I64(values),
            Self::U8(values) => NumericStorage::U8(values),
            Self::U16(values) => NumericStorage::U16(values),
            Self::U32(values) => NumericStorage::U32(values),
            Self::U64(values) => NumericStorage::U64(values),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DataScalar {
    F64(f64),
    F32(f32),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl DataScalar {
    fn to_int_value(self) -> IntValue {
        match self {
            Self::F64(value) => IntValue::I64(value as i64),
            Self::F32(value) => IntValue::I64(value as i64),
            Self::I8(value) => IntValue::I8(value),
            Self::I16(value) => IntValue::I16(value),
            Self::I32(value) => IntValue::I32(value),
            Self::I64(value) => IntValue::I64(value),
            Self::U8(value) => IntValue::U8(value),
            Self::U16(value) => IntValue::U16(value),
            Self::U32(value) => IntValue::U32(value),
            Self::U64(value) => IntValue::U64(value),
        }
    }
}

fn set_at<T>(values: &mut [T], index: usize, value: T) -> BuiltinResult<()> {
    let target = values
        .get_mut(index)
        .ok_or_else(|| data_error(format!("data payload index {index} is out of bounds")))?;
    *target = value;
    Ok(())
}

fn integer_dtype(dtype: &str) -> Option<&'static str> {
    match dtype.to_ascii_lowercase().as_str() {
        "int8" => Some("int8"),
        "int16" => Some("int16"),
        "int32" => Some("int32"),
        "int64" => Some("int64"),
        "uint8" => Some("uint8"),
        "uint16" => Some("uint16"),
        "uint32" => Some("uint32"),
        "uint64" => Some("uint64"),
        _ => None,
    }
}

fn is_single_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "single" | "f32" | "float32"
    )
}

fn is_double_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "double" | "f64" | "float64"
    )
}

fn unsupported_data_dtype(dtype: &str) -> RuntimeError {
    data_error(format!(
        "unsupported data array dtype '{dtype}'; expected f64, f32, or a built-in integer class"
    ))
}

fn integer_target(dtype: &str) -> Option<IntegerClass> {
    IntegerClass::from_class_name(dtype)
}

impl DataArrayPayload {
    /// Validate zero-payload semantics without allocating the eventual array.
    /// A multi-array create calls this for every schema entry before writing
    /// any earlier array, so a late bad dtype or shape has no filesystem effect.
    pub fn validate_zeros(dtype: &str, shape: &[usize]) -> BuiltinResult<()> {
        checked_shape_element_count(shape)?;
        let _ = DataArrayValues::zeros(dtype, 0)?;
        Ok(())
    }

    pub fn zeros(dtype: String, shape: Vec<usize>) -> BuiltinResult<Self> {
        let values = DataArrayValues::zeros(&dtype, checked_shape_element_count(&shape)?)?;
        Ok(Self {
            dtype,
            shape,
            values,
            imaginary_values: None,
        })
    }

    pub async fn zeros_cooperative(
        dtype: String,
        shape: Vec<usize>,
        budget: &mut Option<crate::execution::CpuCooperationBudget>,
    ) -> BuiltinResult<Self> {
        let len = checked_shape_element_count(&shape)?;
        let values = DataArrayValues::zeros_cooperative(&dtype, len, budget).await?;
        Ok(Self {
            dtype,
            shape,
            values,
            imaginary_values: None,
        })
    }

    pub fn from_value(dtype: String, value: &Value) -> BuiltinResult<Self> {
        let (shape, values, imaginary_values) = data_values_from_value(value)?;
        Ok(Self {
            dtype: dtype.clone(),
            shape,
            values: values.cast_to_dtype(&dtype)?,
            imaginary_values: imaginary_values
                .map(|values| values.cast_to_dtype(&dtype))
                .transpose()?,
        })
    }

    pub fn filled(dtype: String, shape: Vec<usize>, value: &Value) -> BuiltinResult<Self> {
        let scalar = Self::from_value(dtype.clone(), value)?;
        if scalar.values.len() != 1 {
            return Err(data_error("expected numeric scalar"));
        }
        let imaginary_scalar = scalar
            .imaginary_values
            .as_ref()
            .map(|values| values.get(0))
            .transpose()?;
        let scalar = scalar.values.get(0)?;
        let len = checked_shape_element_count(&shape)?;
        let mut values = DataArrayValues::zeros(&dtype, len)?;
        let mut imaginary_values = imaginary_scalar
            .map(|_| DataArrayValues::zeros(&dtype, len))
            .transpose()?;
        for index in 0..len {
            values.set(index, scalar)?;
            if let (Some(values), Some(scalar)) = (&mut imaginary_values, imaginary_scalar) {
                values.set(index, scalar)?;
            }
        }
        Ok(Self {
            dtype,
            shape,
            values,
            imaginary_values,
        })
    }

    pub async fn filled_cooperative(
        dtype: String,
        shape: Vec<usize>,
        value: &Value,
        budget: &mut Option<crate::execution::CpuCooperationBudget>,
    ) -> BuiltinResult<Self> {
        let scalar = Self::from_value(dtype.clone(), value)?;
        if scalar.values.len() != 1 {
            return Err(data_error("expected numeric scalar"));
        }
        let imaginary_scalar = scalar
            .imaginary_values
            .as_ref()
            .map(|values| values.get(0))
            .transpose()?;
        let scalar = scalar.values.get(0)?;
        let len = checked_shape_element_count(&shape)?;
        let mut values = DataArrayValues::zeros_cooperative(&dtype, len, budget).await?;
        let mut imaginary_values = match imaginary_scalar {
            Some(_) => Some(DataArrayValues::zeros_cooperative(&dtype, len, budget).await?),
            None => None,
        };
        for index in 0..len {
            values.set(index, scalar)?;
            if let (Some(values), Some(scalar)) = (&mut imaginary_values, imaginary_scalar) {
                values.set(index, scalar)?;
            }
            cooperate_data_work(budget, 1).await?;
        }
        Ok(Self {
            dtype,
            shape,
            values,
            imaginary_values,
        })
    }

    pub fn normalize_for_dtype(mut self, dtype: &str) -> BuiltinResult<Self> {
        self.values = self.values.cast_to_dtype(dtype)?;
        self.imaginary_values = self
            .imaginary_values
            .map(|values| values.cast_to_dtype(dtype))
            .transpose()?;
        self.dtype = dtype.to_string();
        Ok(self)
    }

    pub fn into_value(self) -> BuiltinResult<Value> {
        let Some(imaginary_values) = self.imaginary_values else {
            return self
                .values
                .into_tensor(self.shape)
                .map(Value::Tensor)
                .map_err(|err| data_error(format!("invalid data payload: {err}")));
        };
        let real = self.values.into_numeric_storage();
        let imag = imaginary_values.into_numeric_storage();
        let storage = match (real, imag) {
            (NumericStorage::F64(real), NumericStorage::F64(imag)) => {
                ComplexStorage::F64(real.into_iter().zip(imag).collect())
            }
            (NumericStorage::F32(real), NumericStorage::F32(imag)) => {
                ComplexStorage::F32(real.into_iter().zip(imag).collect())
            }
            (real, imag) => {
                let real = real.into_integer_storage().map_err(|_| {
                    data_error("complex data payload components have mismatched storage classes")
                })?;
                let imag = imag.into_integer_storage().map_err(|_| {
                    data_error("complex data payload components have mismatched storage classes")
                })?;
                ComplexStorage::Integer(IntegerComplexStorage::new(real, imag).map_err(
                    |error| data_error(format!("invalid complex data payload: {error}")),
                )?)
            }
        };
        ComplexTensor::from_complex_storage(storage, self.shape)
            .map(Value::ComplexTensor)
            .map_err(|error| data_error(format!("invalid complex data payload: {error}")))
    }
}

fn checked_shape_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| data_error("data array shape exceeds platform element-count limits"))
    })
}

fn data_values_from_value(
    value: &Value,
) -> BuiltinResult<(Vec<usize>, DataArrayValues, Option<DataArrayValues>)> {
    match value {
        Value::Tensor(tensor) => {
            let storage = tensor
                .clone()
                .into_numeric_storage()
                .map_err(|error| data_error(format!("invalid numeric tensor storage: {error}")))?;
            let values = DataArrayValues::from_numeric_storage(storage);
            Ok((tensor.shape().to_vec(), values, None))
        }
        Value::Num(value) => Ok((vec![1, 1], DataArrayValues::F64(vec![*value]), None)),
        Value::Int(IntValue::I8(value)) => {
            Ok((vec![1, 1], DataArrayValues::I8(vec![*value]), None))
        }
        Value::Int(IntValue::I16(value)) => {
            Ok((vec![1, 1], DataArrayValues::I16(vec![*value]), None))
        }
        Value::Int(IntValue::I32(value)) => {
            Ok((vec![1, 1], DataArrayValues::I32(vec![*value]), None))
        }
        Value::Int(IntValue::I64(value)) => {
            Ok((vec![1, 1], DataArrayValues::I64(vec![*value]), None))
        }
        Value::Int(IntValue::U8(value)) => {
            Ok((vec![1, 1], DataArrayValues::U8(vec![*value]), None))
        }
        Value::Int(IntValue::U16(value)) => {
            Ok((vec![1, 1], DataArrayValues::U16(vec![*value]), None))
        }
        Value::Int(IntValue::U32(value)) => {
            Ok((vec![1, 1], DataArrayValues::U32(vec![*value]), None))
        }
        Value::Int(IntValue::U64(value)) => {
            Ok((vec![1, 1], DataArrayValues::U64(vec![*value]), None))
        }
        Value::Complex(real, imag) => Ok((
            vec![1, 1],
            DataArrayValues::F64(vec![*real]),
            Some(DataArrayValues::F64(vec![*imag])),
        )),
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape().to_vec();
            let (real, imag) = match tensor.clone().into_complex_storage() {
                ComplexStorage::F64(values) => {
                    let (real, imag): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                    (DataArrayValues::F64(real), DataArrayValues::F64(imag))
                }
                ComplexStorage::F32(values) => {
                    let (real, imag): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                    (DataArrayValues::F32(real), DataArrayValues::F32(imag))
                }
                ComplexStorage::Integer(storage) => {
                    let (real, imag) = storage.into_unbounded_components();
                    (
                        DataArrayValues::from_integer_storage(real),
                        DataArrayValues::from_integer_storage(imag),
                    )
                }
            };
            Ok((shape, real, Some(imag)))
        }
        _ => Err(data_error(
            "DataArray.write supports tensor or numeric scalar values",
        )),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunkIndex {
    pub schema_version: u32,
    pub array: String,
    pub chunks: Vec<DataChunkIndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunkIndexEntry {
    pub key: String,
    pub object_id: String,
    pub hash: String,
    pub bytes_raw: u64,
    pub bytes_stored: u64,
    #[serde(default)]
    pub coords: Vec<usize>,
    #[serde(default)]
    pub shape: Vec<usize>,
    pub data_path: String,
}

#[derive(Debug, Clone)]
pub struct DataSchema {
    pub arrays: BTreeMap<String, DataArrayMeta>,
}

#[derive(Debug, Clone)]
pub struct PendingTxn {
    pub dataset_path: String,
    pub base_sequence: u64,
    /// Monotonic mutation identity within this registry-issued transaction.
    pub revision: u64,
    pub writes: Vec<PendingWrite>,
    pub resizes: Vec<PendingResize>,
    pub fills: Vec<PendingFill>,
    pub create_arrays: Vec<PendingCreateArray>,
    pub delete_arrays: Vec<String>,
    pub attrs: BTreeMap<String, Value>,
    pub status: TxnStatus,
}

#[derive(Debug, Clone)]
pub struct PendingWrite {
    pub array: String,
    pub slice_spec: Option<Value>,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct PendingResize {
    pub array: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PendingFill {
    pub array: String,
    pub slice_spec: Option<Value>,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct PendingCreateArray {
    pub array: String,
    pub meta: DataArrayMeta,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnStatus {
    Open,
    Committed,
    Aborted,
}

thread_local! {
    static FALLBACK_TX_REGISTRY: RefCell<HashMap<String, PendingTxn>> = RefCell::new(HashMap::new());
}

#[cfg(not(target_arch = "wasm32"))]
tokio::task_local! {
    static TASK_TX_REGISTRY: RefCell<HashMap<String, PendingTxn>>;
}

pub async fn with_tx_registry_scope<F>(future: F) -> F::Output
where
    F: Future,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        if TASK_TX_REGISTRY.try_with(|_| ()).is_ok() {
            future.await
        } else {
            TASK_TX_REGISTRY
                .scope(RefCell::new(HashMap::new()), future)
                .await
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        future.await
    }
}

fn with_tx_registry<T>(f: impl FnOnce(&mut HashMap<String, PendingTxn>) -> T) -> BuiltinResult<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if TASK_TX_REGISTRY.try_with(|_| ()).is_ok() {
            return TASK_TX_REGISTRY.with(|registry| {
                let mut registry = registry.try_borrow_mut().map_err(|_| {
                    data_error("data transaction registry is already mutably borrowed")
                })?;
                Ok(f(&mut registry))
            });
        }
    }

    FALLBACK_TX_REGISTRY.with(|registry| {
        let mut registry = registry
            .try_borrow_mut()
            .map_err(|_| data_error("data transaction registry is already mutably borrowed"))?;
        Ok(f(&mut registry))
    })
}

pub fn data_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RUNMAT:Data:Error")
        .with_builtin("data")
        .build()
}

pub fn data_error_with_source(
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier("RUNMAT:Data:Error")
        .with_builtin("data")
        .with_source(source)
        .build()
}

fn data_error_with_identifier(
    message: impl Into<String>,
    identifier: &'static str,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(identifier)
        .with_builtin("data")
        .build()
}

const DATA_MANIFEST_CONFLICT_IDENTIFIER: &str = "RunMat:data:ManifestConflict";
const DATA_TRANSACTION_NOT_FOUND_IDENTIFIER: &str = "RunMat:data:TransactionNotFound";

pub fn parse_string(value: &Value, context: &str) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
        if matches!(value, Value::CharArray(_)) {
            data_error(format!("{context}: expected character row vector"))
        } else {
            data_error(format!("{context}: expected string value"))
        }
    })
}

pub fn dataset_root(path: &str) -> PathBuf {
    PathBuf::from(path)
}

pub fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

pub fn arrays_root(root: &Path) -> PathBuf {
    root.join("arrays")
}

pub async fn write_manifest_async(root: &Path, manifest: &DataManifest) -> BuiltinResult<()> {
    let (provider, resolved) = captured_storage_root(root)?;
    write_manifest_on(provider.as_ref(), &resolved, manifest).await
}

pub async fn write_manifest_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    manifest: &DataManifest,
) -> BuiltinResult<()> {
    provider.create_dir_all(root).await.map_err(|err| {
        data_error_with_source(
            format!("failed to create dataset root '{}': {err}", root.display()),
            err,
        )
    })?;
    let path = manifest_path(root);
    let bytes = serde_json::to_vec_pretty(manifest).map_err(|err| {
        data_error_with_source(format!("failed to encode manifest json: {err}"), err)
    })?;
    provider.write(&path, &bytes).await.map_err(|err| {
        data_error_with_source(
            format!("failed to write manifest '{}': {err}", path.display()),
            err,
        )
    })?;
    Ok(())
}

pub async fn read_manifest_async(root: &Path) -> BuiltinResult<DataManifest> {
    let (provider, resolved) = captured_storage_root(root)?;
    read_manifest_on(provider.as_ref(), &resolved).await
}

pub async fn read_manifest_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
) -> BuiltinResult<DataManifest> {
    let path = manifest_path(root);
    let bytes = provider.read(&path).await.map_err(|err| {
        data_error_with_source(
            format!("failed to read manifest '{}': {err}", path.display()),
            err,
        )
    })?;
    let manifest = serde_json::from_slice::<DataManifest>(&bytes).map_err(|err| {
        data_error_with_source(
            format!("failed to parse manifest '{}': {err}", path.display()),
            err,
        )
    })?;
    Ok(manifest)
}

pub async fn write_array_payload_async(
    root: &Path,
    array: &str,
    payload: &DataArrayPayload,
    chunk_shape: &[usize],
) -> BuiltinResult<(PathBuf, PathBuf)> {
    let (provider, resolved) = captured_storage_root(root)?;
    let mut write_budget = None;
    let (payload_path, index_path) = write_array_payload_on(
        provider.as_ref(),
        &resolved,
        array,
        payload,
        chunk_shape,
        &mut write_budget,
    )
    .await?;
    let rebase = |path: PathBuf| {
        path.strip_prefix(&resolved)
            .map(|relative| root.join(relative))
            .map_err(|_| data_error("written array path escaped its dataset root"))
    };
    Ok((rebase(payload_path)?, rebase(index_path)?))
}

fn captured_storage_root(
    root: &Path,
) -> BuiltinResult<(std::sync::Arc<dyn fs::FsProvider>, PathBuf)> {
    fs::capture_provider_path(root).map_err(|source| {
        data_error_with_source(
            format!("failed to capture data storage directory: {source}"),
            source,
        )
    })
}

pub async fn write_array_payload_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    array: &str,
    payload: &DataArrayPayload,
    chunk_shape: &[usize],
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<(PathBuf, PathBuf)> {
    cooperate_data_work(budget, 0).await?;
    let array_dir = create_array_revision(provider, root, array).await?;
    let payload_path = array_dir.join("data.f64.json");
    let bytes = serde_json::to_vec(payload).map_err(|err| {
        data_error_with_source(format!("failed to encode array payload json: {err}"), err)
    })?;
    provider.write(&payload_path, &bytes).await.map_err(|err| {
        data_error_with_source(
            format!(
                "failed to write payload '{}': {err}",
                payload_path.display()
            ),
            err,
        )
    })?;

    let chunk_dir = array_dir.join("chunks");
    provider.create_dir_all(&chunk_dir).await.map_err(|err| {
        data_error_with_source(
            format!(
                "failed to create chunk dir '{}': {err}",
                chunk_dir.display()
            ),
            err,
        )
    })?;

    let mut index = DataChunkIndex {
        schema_version: 1,
        array: array.to_string(),
        chunks: Vec::new(),
    };
    let mut upload_chunks = Vec::new();
    let grid_shape = chunk_grid_shape(&payload.shape, chunk_shape);
    let mut coords = vec![0usize; payload.shape.len()];
    loop {
        cooperate_data_work(budget, 0).await?;
        let chunk_start = chunk_start_for_coords(&coords, chunk_shape);
        let chunk_extent = chunk_extent_for_start(&chunk_start, chunk_shape, &payload.shape);
        let chunk_payload = DataArrayPayload {
            dtype: payload.dtype.clone(),
            shape: chunk_extent.clone(),
            values: collect_chunk_values(payload, &chunk_start, &chunk_extent, budget).await?,
            imaginary_values: if let Some(values) = payload.imaginary_values.as_ref() {
                Some(
                    collect_chunk_component_values(
                        values,
                        &payload.dtype,
                        &payload.shape,
                        &chunk_start,
                        &chunk_extent,
                        budget,
                    )
                    .await?,
                )
            } else {
                None
            },
        };
        let key = chunk_key(&coords);
        let chunk_bytes = serde_json::to_vec(&chunk_payload).map_err(|err| {
            data_error_with_source(format!("failed to encode chunk payload: {err}"), err)
        })?;
        let identity = ChunkStorageIdentity::for_payload(&chunk_bytes);
        let hash = identity.hash;
        let object_id = identity.object_id;
        let data_path = chunk_dir.join(identity.file_name);
        provider
            .write(&data_path, &chunk_bytes)
            .await
            .map_err(|err| {
                data_error_with_source(
                    format!("failed to write chunk '{}': {err}", data_path.display()),
                    err,
                )
            })?;
        let rel_chunk_path = relative_storage_path(root, &data_path)?;
        index.chunks.push(DataChunkIndexEntry {
            key: key.clone(),
            object_id: object_id.clone(),
            hash: hash.clone(),
            bytes_raw: chunk_bytes.len() as u64,
            bytes_stored: chunk_bytes.len() as u64,
            coords: coords.clone(),
            shape: chunk_extent,
            data_path: rel_chunk_path,
        });
        upload_chunks.push((
            DataChunkDescriptor {
                key,
                object_id,
                hash,
                bytes_raw: chunk_bytes.len() as u64,
                bytes_stored: chunk_bytes.len() as u64,
            },
            chunk_bytes,
        ));
        cooperate_data_work(budget, 1).await?;
        if !advance_index(&mut coords, &grid_shape) {
            break;
        }
    }

    maybe_upload_chunks_on(provider, root, array, upload_chunks, budget).await?;

    tracing::info!(
        target: "runmat.data",
        dataset = %root.display(),
        array = array,
        chunks = index.chunks.len(),
        payload_bytes = bytes.len(),
        "data chunk write planned"
    );

    let chunk_index_path = chunk_dir.join("index.json");
    let chunk_index_bytes = serde_json::to_vec(&index).map_err(|err| {
        data_error_with_source(format!("failed to encode chunk index json: {err}"), err)
    })?;
    provider
        .write(&chunk_index_path, &chunk_index_bytes)
        .await
        .map_err(|err| {
            data_error_with_source(
                format!(
                    "failed to write chunk index '{}': {err}",
                    chunk_index_path.display()
                ),
                err,
            )
        })?;
    Ok((payload_path, chunk_index_path))
}

pub async fn read_array_payload_async(
    root: &Path,
    meta: &DataArrayMeta,
) -> BuiltinResult<DataArrayPayload> {
    let (provider, resolved) = captured_storage_root(root)?;
    read_array_payload_on(provider.as_ref(), &resolved, meta).await
}

pub async fn read_array_payload_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    meta: &DataArrayMeta,
) -> BuiltinResult<DataArrayPayload> {
    let mut budget = None;
    read_array_payload_with_budget_on(provider, root, meta, &mut budget).await
}

/// The invocation-owned budget follows the same provider and decoding route;
/// callers without an execution context use the compatibility wrapper above.
pub async fn read_array_payload_with_budget_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    meta: &DataArrayMeta,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<DataArrayPayload> {
    cooperate_data_work(budget, 0).await?;
    if let Some(index_path) = &meta.chunk_index_path {
        let path = root.join(index_path);
        if provider.metadata(&path).await.is_ok() {
            return read_array_payload_chunked_on(provider, root, meta, &path, budget).await;
        }
    }
    let payload_path = root.join(&meta.data_path);
    let bytes = provider.read(&payload_path).await.map_err(|err| {
        data_error_with_source(
            format!("failed to read payload '{}': {err}", payload_path.display()),
            err,
        )
    })?;
    cooperate_data_work(budget, bytes.len() as u64).await?;
    let payload = serde_json::from_slice::<DataArrayPayload>(&bytes)
        .map_err(|err| {
            data_error_with_source(
                format!(
                    "failed to parse payload '{}': {err}",
                    payload_path.display()
                ),
                err,
            )
        })?
        .normalize_for_dtype(&meta.dtype)?;
    cooperate_data_work(budget, 0).await?;
    Ok(payload)
}

pub async fn read_array_slice_payload_async(
    root: &Path,
    meta: &DataArrayMeta,
    start: &[usize],
    shape: &[usize],
) -> BuiltinResult<DataArrayPayload> {
    let (provider, resolved) = captured_storage_root(root)?;
    read_array_slice_payload_on(provider.as_ref(), &resolved, meta, start, shape).await
}

pub async fn read_array_slice_payload_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    meta: &DataArrayMeta,
    start: &[usize],
    shape: &[usize],
) -> BuiltinResult<DataArrayPayload> {
    let (slice_start, slice_shape) = normalize_slice_bounds(&meta.shape, start, shape)?;
    if let Some(index_path) = &meta.chunk_index_path {
        let path = root.join(index_path);
        if provider.metadata(&path).await.is_ok() {
            return read_array_payload_chunked_slice_on(
                provider,
                root,
                meta,
                &path,
                &slice_start,
                &slice_shape,
            )
            .await;
        }
    }
    let full = read_array_payload_on(provider, root, meta).await?;
    extract_slice_payload(&full, &slice_start, &slice_shape)
}

async fn read_array_payload_chunked_slice_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    meta: &DataArrayMeta,
    index_path: &Path,
    slice_start: &[usize],
    slice_shape: &[usize],
) -> BuiltinResult<DataArrayPayload> {
    let bytes = provider.read(index_path).await.map_err(|err| {
        data_error_with_source(
            format!(
                "failed to read chunk index '{}': {err}",
                index_path.display()
            ),
            err,
        )
    })?;
    let index: DataChunkIndex = serde_json::from_slice(&bytes).map_err(|err| {
        data_error_with_source(
            format!(
                "failed to parse chunk index '{}': {err}",
                index_path.display()
            ),
            err,
        )
    })?;

    let mut values =
        DataArrayValues::zeros(&meta.dtype, checked_shape_element_count(slice_shape)?)?;
    let mut imaginary_values: Option<DataArrayValues> = None;
    for chunk in index.chunks {
        let coords = chunk_coords_from_entry(&chunk, meta.shape.len())?;
        let chunk_start = chunk_start_for_coords(&coords, &meta.chunk_shape);
        let chunk_extent = if chunk.shape.is_empty() {
            chunk_extent_for_start(&chunk_start, &meta.chunk_shape, &meta.shape)
        } else {
            chunk.shape.clone()
        };
        if !chunk_intersects_slice(&chunk_start, &chunk_extent, slice_start, slice_shape) {
            continue;
        }

        let chunk_path = root.join(&chunk.data_path);
        let bytes = provider.read(&chunk_path).await.map_err(|err| {
            data_error_with_source(
                format!(
                    "failed to read chunk payload '{}': {err}",
                    chunk_path.display()
                ),
                err,
            )
        })?;
        let payload: DataArrayPayload = serde_json::from_slice::<DataArrayPayload>(&bytes)
            .map_err(|err| {
                data_error_with_source(
                    format!(
                        "failed to parse chunk payload '{}': {err}",
                        chunk_path.display()
                    ),
                    err,
                )
            })?
            .normalize_for_dtype(&meta.dtype)?;
        if payload.shape != chunk_extent {
            return Err(data_error(format!(
                "chunk payload shape mismatch for key '{}': {:?} != {:?}",
                chunk.key, payload.shape, chunk_extent
            )));
        }

        let mut local = vec![0usize; chunk_extent.len()];
        loop {
            let mut global = Vec::with_capacity(chunk_extent.len());
            for dim in 0..chunk_extent.len() {
                global.push(chunk_start[dim] + local[dim]);
            }
            if coordinate_in_slice(&global, slice_start, slice_shape) {
                let src_linear = linear_index_column_major(&local, &chunk_extent)?;
                let mut dst = Vec::with_capacity(slice_shape.len());
                for dim in 0..slice_shape.len() {
                    dst.push(global[dim].saturating_sub(slice_start[dim]));
                }
                let dst_linear = linear_index_column_major(&dst, slice_shape)?;
                values.set(dst_linear, payload.values.get(src_linear)?)?;
                if let Some(payload_imaginary) = &payload.imaginary_values {
                    let target = match &mut imaginary_values {
                        Some(values) => values,
                        None => imaginary_values.insert(DataArrayValues::zeros(
                            &meta.dtype,
                            checked_shape_element_count(slice_shape)?,
                        )?),
                    };
                    target.set(dst_linear, payload_imaginary.get(src_linear)?)?;
                }
            }
            if !advance_index(&mut local, &chunk_extent) {
                break;
            }
        }
    }

    Ok(DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: slice_shape.to_vec(),
        values,
        imaginary_values,
    })
}

async fn read_array_payload_chunked_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    meta: &DataArrayMeta,
    index_path: &Path,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<DataArrayPayload> {
    cooperate_data_work(budget, 0).await?;
    let bytes = provider.read(index_path).await.map_err(|err| {
        data_error_with_source(
            format!(
                "failed to read chunk index '{}': {err}",
                index_path.display()
            ),
            err,
        )
    })?;
    let index: DataChunkIndex = serde_json::from_slice(&bytes).map_err(|err| {
        data_error_with_source(
            format!(
                "failed to parse chunk index '{}': {err}",
                index_path.display()
            ),
            err,
        )
    })?;
    cooperate_data_work(budget, bytes.len() as u64).await?;
    let mut values = DataArrayValues::zeros_cooperative(
        &meta.dtype,
        checked_shape_element_count(&meta.shape)?,
        budget,
    )
    .await?;
    let mut imaginary_values: Option<DataArrayValues> = None;
    for chunk in index.chunks {
        cooperate_data_work(budget, 0).await?;
        let chunk_path = root.join(&chunk.data_path);
        let bytes = provider.read(&chunk_path).await.map_err(|err| {
            data_error_with_source(
                format!(
                    "failed to read chunk payload '{}': {err}",
                    chunk_path.display()
                ),
                err,
            )
        })?;
        let payload: DataArrayPayload = serde_json::from_slice::<DataArrayPayload>(&bytes)
            .map_err(|err| {
                data_error_with_source(
                    format!(
                        "failed to parse chunk payload '{}': {err}",
                        chunk_path.display()
                    ),
                    err,
                )
            })?
            .normalize_for_dtype(&meta.dtype)?;
        cooperate_data_work(budget, bytes.len() as u64).await?;
        let coords = chunk_coords_from_entry(&chunk, meta.shape.len())?;
        let chunk_start = chunk_start_for_coords(&coords, &meta.chunk_shape);
        let chunk_extent = if chunk.shape.is_empty() {
            chunk_extent_for_start(&chunk_start, &meta.chunk_shape, &meta.shape)
        } else {
            chunk.shape.clone()
        };
        if payload.shape != chunk_extent {
            return Err(data_error(format!(
                "chunk payload shape mismatch for key '{}': {:?} != {:?}",
                chunk.key, payload.shape, chunk_extent
            )));
        }
        let mut local = vec![0usize; chunk_extent.len()];
        loop {
            cooperate_data_work(budget, 1).await?;
            let mut global = Vec::with_capacity(chunk_extent.len());
            for dim in 0..chunk_extent.len() {
                global.push(chunk_start[dim] + local[dim]);
            }
            let src_linear = linear_index_column_major(&local, &chunk_extent)?;
            let dst_linear = linear_index_column_major(&global, &meta.shape)?;
            values.set(dst_linear, payload.values.get(src_linear)?)?;
            if let Some(payload_imaginary) = &payload.imaginary_values {
                if imaginary_values.is_none() {
                    imaginary_values = Some(
                        DataArrayValues::zeros_cooperative(
                            &meta.dtype,
                            checked_shape_element_count(&meta.shape)?,
                            budget,
                        )
                        .await?,
                    );
                }
                let target = imaginary_values
                    .as_mut()
                    .expect("imaginary storage initialized");
                target.set(dst_linear, payload_imaginary.get(src_linear)?)?;
            }
            if !advance_index(&mut local, &chunk_extent) {
                break;
            }
        }
    }
    Ok(DataArrayPayload {
        dtype: meta.dtype.clone(),
        shape: meta.shape.clone(),
        values,
        imaginary_values,
    })
}

async fn maybe_upload_chunks_on(
    provider: &dyn fs::FsProvider,
    root: &Path,
    array: &str,
    chunks: Vec<(DataChunkDescriptor, Vec<u8>)>,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    let request = DataChunkUploadRequest {
        dataset_path: root.to_string_lossy().to_string(),
        array: array.to_string(),
        chunks: chunks.iter().map(|(desc, _)| desc.clone()).collect(),
    };
    let targets = match provider.data_chunk_upload_targets(&request).await {
        Ok(targets) => targets,
        Err(err) if err.kind() == std::io::ErrorKind::Unsupported => return Ok(()),
        Err(err) => {
            return Err(data_error_with_source(
                format!("failed to request data chunk upload targets: {err}"),
                err,
            ))
        }
    };
    for (descriptor, bytes) in chunks {
        cooperate_data_work(budget, 1).await?;
        let target = find_chunk_target(&targets, &descriptor.key)?;
        provider
            .data_upload_chunk(target, &bytes)
            .await
            .map_err(|err| {
                data_error_with_source(
                    format!("failed to upload chunk '{}': {err}", descriptor.key),
                    err,
                )
            })?;
        tracing::info!(
            target: "runmat.data",
            dataset = %root.display(),
            array = array,
            chunk_key = descriptor.key,
            bytes = bytes.len(),
            "data chunk uploaded"
        );
    }
    Ok(())
}

fn find_chunk_target<'a>(
    targets: &'a [DataChunkUploadTarget],
    key: &str,
) -> BuiltinResult<&'a DataChunkUploadTarget> {
    targets
        .iter()
        .find(|target| target.key == key)
        .ok_or_else(|| data_error(format!("missing upload target for chunk '{key}'")))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("sha256:{}", hex::encode(digest))
}

fn chunk_key(coords: &[usize]) -> String {
    coords
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

fn chunk_grid_shape(shape: &[usize], chunk_shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .map(|(idx, extent)| {
            let chunk = chunk_shape.get(idx).copied().unwrap_or(1).max(1);
            extent.div_ceil(chunk)
        })
        .collect()
}

fn chunk_start_for_coords(coords: &[usize], chunk_shape: &[usize]) -> Vec<usize> {
    coords
        .iter()
        .enumerate()
        .map(|(idx, coord)| coord * chunk_shape.get(idx).copied().unwrap_or(1).max(1))
        .collect()
}

fn chunk_extent_for_start(
    start: &[usize],
    chunk_shape: &[usize],
    full_shape: &[usize],
) -> Vec<usize> {
    start
        .iter()
        .enumerate()
        .map(|(idx, start)| {
            let chunk = chunk_shape.get(idx).copied().unwrap_or(1).max(1);
            let end = (*start + chunk).min(full_shape[idx]);
            end.saturating_sub(*start)
        })
        .collect()
}

async fn collect_chunk_values(
    payload: &DataArrayPayload,
    chunk_start: &[usize],
    chunk_extent: &[usize],
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<DataArrayValues> {
    collect_chunk_component_values(
        &payload.values,
        &payload.dtype,
        &payload.shape,
        chunk_start,
        chunk_extent,
        budget,
    )
    .await
}

async fn collect_chunk_component_values(
    source: &DataArrayValues,
    dtype: &str,
    full_shape: &[usize],
    chunk_start: &[usize],
    chunk_extent: &[usize],
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<DataArrayValues> {
    let mut local = vec![0usize; chunk_extent.len()];
    let mut values = DataArrayValues::zeros(dtype, 0)?;
    loop {
        cooperate_data_work(budget, 1).await?;
        let mut global = Vec::with_capacity(chunk_extent.len());
        for dim in 0..chunk_extent.len() {
            global.push(chunk_start[dim] + local[dim]);
        }
        let linear = linear_index_column_major(&global, full_shape)?;
        values.push(source.get(linear)?)?;
        if !advance_index(&mut local, chunk_extent) {
            break;
        }
    }
    Ok(values)
}

fn chunk_coords_from_entry(entry: &DataChunkIndexEntry, rank: usize) -> BuiltinResult<Vec<usize>> {
    if !entry.coords.is_empty() {
        if entry.coords.len() != rank {
            return Err(data_error(format!(
                "chunk coords rank mismatch for key '{}': expected {rank}, got {}",
                entry.key,
                entry.coords.len()
            )));
        }
        return Ok(entry.coords.clone());
    }
    let coords = entry
        .key
        .split('.')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| data_error(format!("invalid chunk key '{}'", entry.key)))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    if coords.len() != rank {
        return Err(data_error(format!(
            "chunk key rank mismatch for key '{}': expected {rank}, got {}",
            entry.key,
            coords.len()
        )));
    }
    Ok(coords)
}

fn normalize_slice_bounds(
    full_shape: &[usize],
    start: &[usize],
    shape: &[usize],
) -> BuiltinResult<(Vec<usize>, Vec<usize>)> {
    if full_shape.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut normalized_start = Vec::with_capacity(full_shape.len());
    let mut normalized_shape = Vec::with_capacity(full_shape.len());
    for (axis, axis_len) in full_shape.iter().copied().enumerate() {
        if axis_len == 0 {
            return Err(data_error("slice axis length must be greater than zero"));
        }
        let requested_start = start.get(axis).copied().unwrap_or(0);
        let clamped_start = requested_start.min(axis_len.saturating_sub(1));
        let requested_span = shape.get(axis).copied().unwrap_or(axis_len);
        let clamped_span = requested_span
            .max(1)
            .min(axis_len.saturating_sub(clamped_start));
        normalized_start.push(clamped_start);
        normalized_shape.push(clamped_span);
    }
    Ok((normalized_start, normalized_shape))
}

fn coordinate_in_slice(global: &[usize], slice_start: &[usize], slice_shape: &[usize]) -> bool {
    for dim in 0..slice_shape.len() {
        let start = slice_start[dim];
        let end = start.saturating_add(slice_shape[dim]);
        let value = global[dim];
        if value < start || value >= end {
            return false;
        }
    }
    true
}

fn chunk_intersects_slice(
    chunk_start: &[usize],
    chunk_extent: &[usize],
    slice_start: &[usize],
    slice_shape: &[usize],
) -> bool {
    for dim in 0..slice_shape.len() {
        let chunk_lo = chunk_start[dim];
        let chunk_hi = chunk_lo.saturating_add(chunk_extent[dim]);
        let slice_lo = slice_start[dim];
        let slice_hi = slice_lo.saturating_add(slice_shape[dim]);
        if chunk_hi <= slice_lo || slice_hi <= chunk_lo {
            return false;
        }
    }
    true
}

fn extract_slice_payload(
    payload: &DataArrayPayload,
    start: &[usize],
    shape: &[usize],
) -> BuiltinResult<DataArrayPayload> {
    let mut values = DataArrayValues::zeros(&payload.dtype, 0)?;
    let mut imaginary_values = payload
        .imaginary_values
        .as_ref()
        .map(|_| DataArrayValues::zeros(&payload.dtype, 0))
        .transpose()?;
    if shape.is_empty() {
        return Ok(DataArrayPayload {
            dtype: payload.dtype.clone(),
            shape: Vec::new(),
            values,
            imaginary_values,
        });
    }
    let mut local = vec![0usize; shape.len()];
    loop {
        let mut global = Vec::with_capacity(shape.len());
        for dim in 0..shape.len() {
            global.push(start[dim] + local[dim]);
        }
        let linear = linear_index_column_major(&global, &payload.shape)?;
        values.push(payload.values.get(linear)?)?;
        if let (Some(source), Some(target)) = (&payload.imaginary_values, &mut imaginary_values) {
            target.push(source.get(linear)?)?;
        }
        if !advance_index(&mut local, shape) {
            break;
        }
    }
    Ok(DataArrayPayload {
        dtype: payload.dtype.clone(),
        shape: shape.to_vec(),
        values,
        imaginary_values,
    })
}

fn linear_index_column_major(index: &[usize], shape: &[usize]) -> BuiltinResult<usize> {
    if index.len() != shape.len() {
        return Err(data_error("chunk index rank mismatch"));
    }
    let mut stride = 1usize;
    let mut linear = 0usize;
    for (idx, extent) in index.iter().zip(shape.iter()) {
        if *idx >= *extent {
            return Err(data_error("chunk index out of bounds"));
        }
        linear += idx * stride;
        stride = stride.saturating_mul(*extent);
    }
    Ok(linear)
}

fn advance_index(index: &mut [usize], shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    for dim in 0..shape.len() {
        index[dim] += 1;
        if index[dim] < shape[dim] {
            return true;
        }
        index[dim] = 0;
    }
    false
}

pub fn parse_schema(schema: &Value) -> BuiltinResult<DataSchema> {
    let Value::Struct(schema_struct) = schema else {
        return Err(data_error("data.create: schema must be a struct"));
    };
    let arrays_value = schema_struct
        .fields
        .get("arrays")
        .ok_or_else(|| data_error("data.create: schema missing 'arrays' field"))?;
    let Value::Struct(arrays_struct) = arrays_value else {
        return Err(data_error("data.create: schema.arrays must be a struct"));
    };

    let mut arrays = BTreeMap::new();
    for (name, meta_value) in &arrays_struct.fields {
        let Value::Struct(meta_struct) = meta_value else {
            return Err(data_error(format!(
                "data.create: schema.arrays.{name} must be a struct"
            )));
        };
        let dtype = meta_struct
            .fields
            .get("dtype")
            .map(|v| parse_string(v, "data.create schema dtype"))
            .transpose()?
            .unwrap_or_else(|| "f64".to_string());
        let shape = meta_struct
            .fields
            .get("shape")
            .map(parse_usize_vector)
            .transpose()?
            .unwrap_or_else(|| vec![0, 0]);
        let chunk_shape = meta_struct
            .fields
            .get("chunk")
            .map(parse_usize_vector)
            .transpose()?
            .unwrap_or_else(|| default_chunk_shape(&shape));
        validate_chunk_shape(&shape, &chunk_shape)?;
        let codec = meta_struct
            .fields
            .get("codec")
            .map(|v| parse_string(v, "data.create schema codec"))
            .transpose()?
            .unwrap_or_else(|| "zstd".to_string());
        let data_path = format!("arrays/{name}/data.f64.json");
        let chunk_index_path = format!("arrays/{name}/chunks/index.json");
        arrays.insert(
            name.to_owned(),
            DataArrayMeta {
                dtype,
                shape,
                chunk_shape,
                order: default_array_order(),
                codec,
                chunk_index_path: Some(chunk_index_path),
                data_path,
            },
        );
    }

    Ok(DataSchema { arrays })
}

fn default_chunk_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = shape.to_vec();
    if out.len() == 1 {
        out[0] = out[0].clamp(1, 65_536);
        return out;
    }
    out[0] = out[0].clamp(1, 256);
    out[1] = out[1].clamp(1, 256);
    for dim in out.iter_mut().skip(2) {
        *dim = (*dim).clamp(1, 8);
    }
    out
}

pub fn validate_chunk_shape(shape: &[usize], chunk_shape: &[usize]) -> BuiltinResult<()> {
    if chunk_shape.len() != shape.len() {
        return Err(data_error(
            "data array chunk shape must have the same rank as its array shape",
        ));
    }
    if chunk_shape.contains(&0) {
        return Err(data_error(
            "data array chunk dimensions must be strictly positive",
        ));
    }
    Ok(())
}

fn parse_usize_vector(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => tensor_to_usize_vector(t),
        Value::Num(n) => floating_dimension_to_usize(*n).map(|value| vec![value]),
        Value::Int(i) => i
            .try_to_usize()
            .map(|n| vec![n])
            .ok_or_else(|| data_error("data schema dimensions must be non-negative integers")),
        _ => Err(data_error(
            "data schema dimension field must be numeric tensor/vector",
        )),
    }
}

fn floating_dimension_to_usize(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return Err(data_error(
            "data schema dimensions must be non-negative finite integers",
        ));
    }
    let integer = value as u128;
    if integer > usize::MAX as u128 || integer as f64 != value {
        return Err(data_error("data schema dimensions exceed platform limits"));
    }
    usize::try_from(integer)
        .map_err(|_| data_error("data schema dimensions exceed platform limits"))
}

fn tensor_to_usize_vector(t: &Tensor) -> BuiltinResult<Vec<usize>> {
    let mut out = Vec::with_capacity(t.len());
    for index in 0..t.len() {
        let value = t
            .numeric_value_at(index)
            .ok_or_else(|| data_error("data schema dimensions require valid numeric storage"))?;
        out.push(match value {
            NumericScalar::F64(value) => floating_dimension_to_usize(value)?,
            NumericScalar::F32(value) => floating_dimension_to_usize(f64::from(value))?,
            value => value
                .into_int_value()
                .and_then(|value| value.try_to_usize())
                .ok_or_else(|| {
                    data_error("data schema dimensions must be non-negative integers")
                })?,
        });
    }
    Ok(out)
}

pub fn dataset_object(
    runtime: Option<&crate::context::RuntimeContext>,
    path: &str,
    manifest: &DataManifest,
    authority: ForeignRef,
) -> BuiltinResult<crate::execution::RootedValue> {
    data_object(
        runtime,
        runmat_builtins::DATASET_CLASS.display_name(),
        [
            ("__data_path", Value::string(path)),
            ("__data_id", Value::string(manifest.dataset_id.clone())),
            (
                "__data_version",
                Value::string(manifest_version_token(manifest)),
            ),
        ],
        authority,
    )
}

pub fn manifest_version_token(manifest: &DataManifest) -> String {
    format!("{}:{}", manifest.updated_at, manifest.txn_sequence)
}

pub fn ensure_manifest_sequence(expected: u64, manifest: &DataManifest) -> BuiltinResult<()> {
    if manifest.txn_sequence != expected {
        tracing::warn!(
            target: "runmat.data",
            expected_sequence = expected,
            actual_sequence = manifest.txn_sequence,
            "manifest conflict detected"
        );
        return Err(data_error_with_identifier(
            "MANIFEST_CONFLICT: dataset changed since transaction begin",
            DATA_MANIFEST_CONFLICT_IDENTIFIER,
        ));
    }
    Ok(())
}

pub fn array_object(
    runtime: Option<&crate::context::RuntimeContext>,
    dataset_path: &str,
    array_name: &str,
    authority: ForeignRef,
) -> BuiltinResult<crate::execution::RootedValue> {
    data_object(
        runtime,
        runmat_builtins::DATA_ARRAY_CLASS.display_name(),
        [
            ("__data_path", Value::string(dataset_path)),
            ("__array_name", Value::string(array_name)),
        ],
        authority,
    )
}

pub fn transaction_object(
    runtime: Option<&crate::context::RuntimeContext>,
    dataset_path: &str,
    tx_id: &str,
    authority: ForeignRef,
) -> BuiltinResult<crate::execution::RootedValue> {
    data_object(
        runtime,
        runmat_builtins::DATA_TRANSACTION_CLASS.display_name(),
        [
            ("__data_path", Value::string(dataset_path)),
            ("__tx_id", Value::string(tx_id)),
        ],
        authority,
    )
}

fn data_object<const N: usize>(
    runtime: Option<&crate::context::RuntimeContext>,
    class: &str,
    fields: [(&str, Value); N],
    authority: ForeignRef,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::execution::object::construct_object_from_fields_with_native(
        runtime,
        class,
        fields.iter().map(|(name, value)| (*name, value)),
        Some(crate::execution::object::ObjectNativeInitialization::DatasetAuthority(authority)),
    )
}

pub fn get_object_prop<'a>(obj: &'a ObjectInstance, key: &str) -> BuiltinResult<&'a Value> {
    obj.properties()
        .get(key)
        .ok_or_else(|| data_error(format!("object missing internal property '{key}'")))
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn new_dataset_id() -> String {
    static NEXT_DATASET_ID: AtomicU64 = AtomicU64::new(1);
    let seq = NEXT_DATASET_ID.fetch_add(1, Ordering::Relaxed);
    format!("ds_{}_{}", Utc::now().timestamp_millis(), seq)
}

pub fn new_tx_id() -> BuiltinResult<String> {
    static NEXT_TX_ID: AtomicU64 = AtomicU64::new(1);
    let seq = NEXT_TX_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| data_error("transaction identifier space is exhausted"))?;
    Ok(format!("tx_{}_{}", Utc::now().timestamp_millis(), seq))
}

#[cfg(test)]
pub fn start_tx(dataset_path: String, base_sequence: u64) -> BuiltinResult<String> {
    let tx_id = new_tx_id()?;
    start_tx_with_id(tx_id.clone(), dataset_path, base_sequence)?;
    Ok(tx_id)
}

pub fn start_tx_with_id(
    tx_id: String,
    dataset_path: String,
    base_sequence: u64,
) -> BuiltinResult<()> {
    let pending = PendingTxn {
        dataset_path,
        base_sequence,
        revision: 0,
        writes: Vec::new(),
        resizes: Vec::new(),
        fills: Vec::new(),
        create_arrays: Vec::new(),
        delete_arrays: Vec::new(),
        attrs: BTreeMap::new(),
        status: TxnStatus::Open,
    };
    with_tx_registry(|registry| {
        if registry.contains_key(&tx_id) {
            return Err(data_error("transaction identifier already exists"));
        }
        registry.insert(tx_id, pending);
        Ok(())
    })??;
    Ok(())
}

pub fn with_tx_mut<T>(
    tx_id: &str,
    f: impl FnOnce(&mut PendingTxn) -> BuiltinResult<T>,
) -> BuiltinResult<T> {
    with_tx_registry(|registry| {
        let tx = registry.get_mut(tx_id).ok_or_else(|| {
            data_error_with_identifier(
                format!("transaction '{tx_id}' not found"),
                DATA_TRANSACTION_NOT_FOUND_IDENTIFIER,
            )
        })?;
        let next_revision = tx
            .revision
            .checked_add(1)
            .ok_or_else(|| data_error("transaction mutation revision exhausted"))?;
        // Invalidate earlier admissions even if a fallible closure mutates
        // before returning an error. The sealed transaction ID never recycles.
        tx.revision = next_revision;
        f(tx)
    })?
}

pub fn with_tx<T>(
    tx_id: &str,
    f: impl FnOnce(&PendingTxn) -> BuiltinResult<T>,
) -> BuiltinResult<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if TASK_TX_REGISTRY.try_with(|_| ()).is_ok() {
            return TASK_TX_REGISTRY.with(|registry| {
                let registry = registry
                    .try_borrow()
                    .map_err(|_| data_error("data transaction registry is already borrowed"))?;
                let tx = registry.get(tx_id).ok_or_else(|| {
                    data_error_with_identifier(
                        format!("transaction '{tx_id}' not found"),
                        DATA_TRANSACTION_NOT_FOUND_IDENTIFIER,
                    )
                })?;
                f(tx)
            });
        }
    }

    FALLBACK_TX_REGISTRY.with(|registry| {
        let registry = registry
            .try_borrow()
            .map_err(|_| data_error("data transaction registry is already borrowed"))?;
        let tx = registry.get(tx_id).ok_or_else(|| {
            data_error_with_identifier(
                format!("transaction '{tx_id}' not found"),
                DATA_TRANSACTION_NOT_FOUND_IDENTIFIER,
            )
        })?;
        f(tx)
    })
}

pub fn remove_tx(tx_id: &str) -> BuiltinResult<()> {
    with_tx_registry(|registry| {
        let _ = registry.remove(tx_id);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn typed_value_encoding_preserves_low_rank_shape_and_payload() {
        for shape in [Vec::new(), vec![2]] {
            let count = shape.iter().product::<usize>();
            let values = vec![u64::MAX; count];
            let input = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(values.clone()), shape.clone()).unwrap(),
            );
            let (encoded_shape, real, imag) = data_values_from_value(&input).unwrap();
            assert_eq!(encoded_shape, shape);
            assert_eq!(real, DataArrayValues::U64(values));
            assert_eq!(imag, None);

            let complex = Value::ComplexTensor(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(vec![(1.25, 2.5); count].into()),
                    shape.clone(),
                )
                .unwrap(),
            );
            let (encoded_shape, real, imag) = data_values_from_value(&complex).unwrap();
            assert_eq!(encoded_shape, shape);
            assert_eq!(real, DataArrayValues::F32(vec![1.25; count]));
            assert_eq!(imag, Some(DataArrayValues::F32(vec![2.5; count])));
        }
    }

    #[path = "cooperation.rs"]
    mod cooperation;

    #[test]
    fn schema_dimensions_preserve_large_typed_unsigned_values() {
        let expected = usize::try_from(u64::MAX).ok();
        let parsed = parse_usize_vector(&Value::Int(IntValue::U64(u64::MAX)));
        match expected {
            Some(value) => assert_eq!(parsed.expect("representable dimension"), vec![value]),
            None => assert!(parsed.is_err()),
        }
        assert!(parse_usize_vector(&Value::Int(IntValue::I64(-1))).is_err());
    }

    #[test]
    fn schema_dimension_tensors_preserve_exact_integer_storage() {
        let cases = [
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ];

        for storage in cases {
            let input = Tensor::new_integer(storage, vec![1, 2]).expect("dimension tensor");
            assert_eq!(
                parse_usize_vector(&Value::Tensor(input)).expect("typed dimensions"),
                vec![2, 3]
            );
        }

        #[cfg(target_pointer_width = "64")]
        {
            let input = Tensor::new_integer(
                IntegerStorage::U64(vec![1_u64 << 53, (1_u64 << 53) + 1]),
                vec![1, 2],
            )
            .expect("uint64 dimension tensor");
            assert_eq!(
                parse_usize_vector(&Value::Tensor(input)).expect("wide typed dimensions"),
                vec![
                    usize::try_from(1_u64 << 53).expect("representable first dimension"),
                    usize::try_from((1_u64 << 53) + 1).expect("representable second dimension"),
                ]
            );
        }
    }

    #[test]
    fn schema_dimension_tensors_accept_native_single_and_reject_fractional_floats() {
        let input = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).expect("single dimensions");
        assert_eq!(
            parse_usize_vector(&Value::Tensor(input)).expect("single dimensions"),
            vec![2, 3]
        );

        let fractional =
            Tensor::from_f32(vec![2.0, 3.5], vec![1, 2]).expect("fractional dimensions");
        assert!(parse_usize_vector(&Value::Tensor(fractional)).is_err());
        assert!(parse_usize_vector(&Value::Num(1.5)).is_err());
        assert!(parse_usize_vector(&Value::Num(f64::INFINITY)).is_err());
    }

    #[test]
    fn schema_dimension_tensors_reject_negative_integer_storage() {
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![2, -1]), vec![1, 2]).expect("int16 dims");
        assert!(parse_usize_vector(&Value::Tensor(input)).is_err());
    }

    #[test]
    fn payload_allocation_rejects_shape_product_overflow() {
        let error = DataArrayPayload::zeros("uint64".to_string(), vec![usize::MAX, 2])
            .expect_err("overflowing shape must reject before allocation");
        assert!(error
            .message()
            .contains("shape exceeds platform element-count limits"));
        let error = DataArrayPayload::filled(
            "uint64".to_string(),
            vec![usize::MAX, 2],
            &Value::Int(IntValue::U64(1)),
        )
        .expect_err("overflowing filled shape must reject before allocation");
        assert!(error
            .message()
            .contains("shape exceeds platform element-count limits"));
        for shape in [
            vec![0, usize::MAX, 2],
            vec![usize::MAX, 0, 2],
            vec![usize::MAX, 2, 0],
        ] {
            let payload = DataArrayPayload::zeros("uint64".to_string(), shape.clone())
                .expect("a zero dimension makes the total element count zero");
            assert_eq!(payload.shape, shape);
            assert_eq!(payload.values.len(), 0);
        }
    }

    #[test]
    fn chunk_shape_requires_positive_rank_matched_dimensions() {
        validate_chunk_shape(&[4, 5], &[2, 5]).expect("valid chunk shape");
        assert!(validate_chunk_shape(&[4, 5], &[2]).is_err());
        assert!(validate_chunk_shape(&[4, 5], &[2, 0]).is_err());
        validate_chunk_shape(&[], &[]).expect("rank-zero metadata remains self-consistent");
    }

    #[test]
    fn payload_rejects_unknown_dtype_instead_of_falling_back_to_f64() {
        let error = DataArrayPayload::zeros("mystery".to_string(), vec![1, 1])
            .expect_err("unknown dtype must reject");
        assert!(error.message().contains("unsupported data array dtype"));
        let error = DataArrayPayload::from_value("mystery".to_string(), &Value::Num(1.0))
            .expect_err("unknown cast target must reject");
        assert!(error.message().contains("unsupported data array dtype"));
    }

    #[test]
    fn payload_roundtrips_every_native_integer_storage_class() {
        let cases = vec![
            DataArrayValues::I8(vec![i8::MIN, i8::MAX]),
            DataArrayValues::I16(vec![i16::MIN, i16::MAX]),
            DataArrayValues::I32(vec![i32::MIN, i32::MAX]),
            DataArrayValues::I64(vec![i64::MIN, i64::MAX]),
            DataArrayValues::U8(vec![0, u8::MAX]),
            DataArrayValues::U16(vec![0, u16::MAX]),
            DataArrayValues::U32(vec![0, u32::MAX]),
            DataArrayValues::U64(vec![0, u64::MAX]),
        ];

        for values in cases {
            let dtype = match &values {
                DataArrayValues::I8(_) => "int8",
                DataArrayValues::I16(_) => "int16",
                DataArrayValues::I32(_) => "int32",
                DataArrayValues::I64(_) => "int64",
                DataArrayValues::U8(_) => "uint8",
                DataArrayValues::U16(_) => "uint16",
                DataArrayValues::U32(_) => "uint32",
                DataArrayValues::U64(_) => "uint64",
                DataArrayValues::F64(_) | DataArrayValues::F32(_) => unreachable!(),
            };
            let payload = DataArrayPayload {
                dtype: dtype.to_string(),
                shape: vec![1, 2],
                values: values.clone(),
                imaginary_values: None,
            };
            let bytes = serde_json::to_vec(&payload).expect("encode typed payload");
            let decoded: DataArrayPayload = serde_json::from_slice(&bytes).expect("decode payload");
            assert_eq!(decoded.values, values, "{dtype} payload must remain exact");
            let Value::Tensor(tensor) = decoded.into_value().expect("tensor value") else {
                panic!("expected tensor");
            };
            assert_eq!(
                tensor.integer_storage().map(IntegerStorage::class_name),
                Some(dtype)
            );
        }
    }

    #[test]
    fn payload_roundtrips_native_single_storage() {
        let values = DataArrayValues::F32(vec![f32::MIN, 0.1, f32::MAX]);
        let payload = DataArrayPayload {
            dtype: "f32".to_string(),
            shape: vec![1, 3],
            values: values.clone(),
            imaginary_values: None,
        };
        let bytes = serde_json::to_vec(&payload).expect("encode single payload");
        let decoded: DataArrayPayload =
            serde_json::from_slice(&bytes).expect("decode single payload");
        assert_eq!(decoded.values, values);

        let Value::Tensor(tensor) = decoded.into_value().expect("single tensor value") else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(
            tensor.materialize_f64(),
            vec![f64::from(f32::MIN), f64::from(0.1_f32), f64::from(f32::MAX)]
        );
    }

    #[test]
    fn payload_construction_preserves_native_single_tensor() {
        let input = Tensor::from_f32(vec![0.1, -2.5], vec![1, 2]).expect("single tensor");
        let payload = DataArrayPayload::from_value("f32".to_string(), &Value::Tensor(input))
            .expect("single payload");
        assert_eq!(payload.values, DataArrayValues::F32(vec![0.1, -2.5]));
    }

    #[test]
    fn payload_decodes_legacy_f64_arrays_and_normalizes_declared_integer_dtypes() {
        let legacy = br#"{"dtype":"uint64","shape":[1,2],"values":[1,2]}"#;
        let payload: DataArrayPayload =
            serde_json::from_slice(legacy).expect("decode legacy payload");
        assert_eq!(payload.values, DataArrayValues::F64(vec![1.0, 2.0]));

        let payload = payload
            .normalize_for_dtype("uint64")
            .expect("normalize legacy payload");
        assert_eq!(payload.values, DataArrayValues::U64(vec![1, 2]));
    }

    #[test]
    fn preview_conversion_is_bounded_for_typed_integer_payloads() {
        let values = DataArrayValues::I16(vec![-2, 0, 3, 7]);

        assert_eq!(values.preview_f64(3), vec![-2.0, 0.0, 3.0]);
        assert!(values.preview_f64(0).is_empty());
    }

    #[test]
    fn payload_construction_preserves_uint64_tensor_extrema() {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                .expect("uint64 tensor");
        let payload = DataArrayPayload::from_value("uint64".to_string(), &Value::Tensor(input))
            .expect("payload");
        assert_eq!(
            payload.values,
            DataArrayValues::U64(vec![1_u64 << 63, u64::MAX])
        );
    }

    #[test]
    fn payload_rejects_typed_complex_integers_without_float_coercion() {
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
        )
        .expect("matching typed complex components");
        let complex = runmat_value::ComplexTensor::new_integer(storage.clone(), vec![1, 2])
            .expect("typed complex tensor");

        let payload =
            DataArrayPayload::from_value("uint64".to_string(), &Value::ComplexTensor(complex))
                .expect("encode paired integer payload");
        assert_eq!(
            payload.imaginary_values,
            Some(DataArrayValues::U64(vec![u64::MAX, 1_u64 << 63]))
        );
        let bytes = serde_json::to_vec(&payload).expect("serialize paired payload");
        let decoded: DataArrayPayload =
            serde_json::from_slice(&bytes).expect("deserialize paired payload");
        let Value::ComplexTensor(decoded) = decoded.into_value().expect("decode paired payload")
        else {
            panic!("expected paired complex tensor");
        };
        assert_eq!(decoded.integer_storage(), Some(&storage));
    }

    #[test]
    fn ensure_manifest_sequence_accepts_matching_sequence() {
        let manifest = DataManifest {
            schema_version: 1,
            format: "runmat-data".to_string(),
            dataset_id: "ds_test".to_string(),
            name: Some("test".to_string()),
            created_at: "2026-03-01T00:00:00Z".to_string(),
            updated_at: "2026-03-01T00:00:00Z".to_string(),
            arrays: BTreeMap::new(),
            attrs: BTreeMap::new(),
            txn_sequence: 5,
        };
        ensure_manifest_sequence(5, &manifest).expect("expected sequence match");
    }

    #[test]
    fn ensure_manifest_sequence_rejects_conflict() {
        let manifest = DataManifest {
            schema_version: 1,
            format: "runmat-data".to_string(),
            dataset_id: "ds_test".to_string(),
            name: Some("test".to_string()),
            created_at: "2026-03-01T00:00:00Z".to_string(),
            updated_at: "2026-03-01T00:00:00Z".to_string(),
            arrays: BTreeMap::new(),
            attrs: BTreeMap::new(),
            txn_sequence: 6,
        };
        let err = ensure_manifest_sequence(5, &manifest).expect_err("expected conflict error");
        assert_eq!(
            err.identifier(),
            Some(DATA_MANIFEST_CONFLICT_IDENTIFIER),
            "manifest conflicts should expose a stable identifier"
        );
    }

    #[test]
    fn transaction_registry_roundtrip() {
        let tx_id = start_tx("/datasets/test.data".to_string(), 7).expect("start tx");
        let status = with_tx(&tx_id, |tx| Ok(tx.status.clone())).expect("tx lookup");
        assert_eq!(status, TxnStatus::Open);
        remove_tx(&tx_id).expect("remove tx");
        let err = with_tx(&tx_id, |_| Ok(())).expect_err("expected missing tx");
        assert_eq!(
            err.identifier(),
            Some(DATA_TRANSACTION_NOT_FOUND_IDENTIFIER),
            "missing transaction lookups should expose a stable identifier"
        );
    }

    #[test]
    fn transaction_registry_rejects_identity_collision_without_replacing_owner() {
        let tx_id = new_tx_id().expect("unique transaction identity");
        start_tx_with_id(tx_id.clone(), "first.data".into(), 7).expect("first transaction");
        let duplicate = start_tx_with_id(tx_id.clone(), "second.data".into(), 99);
        assert!(
            duplicate.is_err(),
            "duplicate identity must fail before replacing the first"
        );
        with_tx(&tx_id, |tx| {
            assert_eq!(tx.dataset_path, "first.data");
            assert_eq!(tx.base_sequence, 7);
            Ok(())
        })
        .expect("first transaction remains intact");
        remove_tx(&tx_id).expect("remove first transaction");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transaction_registry_scope_survives_await() {
        with_tx_registry_scope(async {
            let tx_id = start_tx("/datasets/task-local.data".to_string(), 11).expect("start tx");
            tokio::task::yield_now().await;
            let status = with_tx(&tx_id, |tx| Ok(tx.status.clone())).expect("tx lookup");
            assert_eq!(status, TxnStatus::Open);
            remove_tx(&tx_id).expect("remove tx");
            let err = with_tx(&tx_id, |_| Ok(())).expect_err("expected missing tx");
            assert_eq!(
                err.identifier(),
                Some(DATA_TRANSACTION_NOT_FOUND_IDENTIFIER)
            );
        })
        .await;
    }

    #[test]
    fn sha256_hash_format_matches_expected_prefix() {
        let hash = sha256_hex(b"runmat");
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), "sha256:".len() + 64);
    }
}
