use super::*;

/// Immutable typed view over authoritative real numeric storage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericStorageView<'a> {
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

/// Mutable typed view over authoritative real numeric storage.
#[derive(Debug)]
pub enum NumericStorageViewMut<'a> {
    F64(&'a mut [f64]),
    F32(&'a mut [f32]),
    I8(&'a mut [i8]),
    I16(&'a mut [i16]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
    U32(&'a mut [u32]),
    U64(&'a mut [u64]),
}

impl NumericStorageView<'_> {
    pub fn numeric_dtype(self) -> NumericDType {
        match self {
            Self::F64(_) => NumericDType::F64,
            Self::F32(_) => NumericDType::F32,
            Self::I8(_) => NumericDType::I8,
            Self::I16(_) => NumericDType::I16,
            Self::I32(_) => NumericDType::I32,
            Self::I64(_) => NumericDType::I64,
            Self::U8(_) => NumericDType::U8,
            Self::U16(_) => NumericDType::U16,
            Self::U32(_) => NumericDType::U32,
            Self::U64(_) => NumericDType::U64,
        }
    }

    pub fn len(self) -> usize {
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

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl NumericStorageViewMut<'_> {
    pub fn numeric_dtype(&self) -> NumericDType {
        match self {
            Self::F64(_) => NumericDType::F64,
            Self::F32(_) => NumericDType::F32,
            Self::I8(_) => NumericDType::I8,
            Self::I16(_) => NumericDType::I16,
            Self::I32(_) => NumericDType::I32,
            Self::I64(_) => NumericDType::I64,
            Self::U8(_) => NumericDType::U8,
            Self::U16(_) => NumericDType::U16,
            Self::U32(_) => NumericDType::U32,
            Self::U64(_) => NumericDType::U64,
        }
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
}

impl From<Vec<f64>> for NumericStorage {
    fn from(values: Vec<f64>) -> Self {
        Self::F64(values)
    }
}

impl From<Vec<f32>> for NumericStorage {
    fn from(values: Vec<f32>) -> Self {
        Self::F32(values)
    }
}

impl From<IntegerStorage> for NumericStorage {
    fn from(storage: IntegerStorage) -> Self {
        Self::from_integer_storage(storage)
    }
}

pub(super) fn cast_exact_unsigned(value: &IntValue, max: u64) -> u64 {
    match value {
        IntValue::U64(value) => (*value).min(max),
        value => (value.to_i64().max(0) as u64).min(max),
    }
}

pub(super) fn cast_f64_signed(value: f64, min: i64, max: i64) -> i64 {
    if value.is_nan() {
        0
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            min
        } else {
            max
        }
    } else {
        value.round().clamp(min as f64, max as f64) as i64
    }
}

pub(super) fn cast_f64_unsigned(value: f64, max: u64) -> u64 {
    if value.is_nan() || value.is_sign_negative() {
        0
    } else if value.is_infinite() {
        max
    } else {
        value.round().clamp(0.0, max as f64) as u64
    }
}

pub(super) fn set_integer_element<T>(
    values: &mut [T],
    index: usize,
    value: T,
) -> Result<(), String> {
    let slot = values
        .get_mut(index)
        .ok_or_else(|| format!("integer storage index {index} is out of bounds"))?;
    *slot = value;
    Ok(())
}

pub(super) fn set_numeric_element<T>(
    values: &mut [T],
    index: usize,
    value: T,
    class_name: &str,
) -> Result<(), String> {
    let slot = values
        .get_mut(index)
        .ok_or_else(|| format!("{class_name} numeric storage index {index} is out of bounds"))?;
    *slot = value;
    Ok(())
}

pub(super) fn gather_numeric_values<T: Copy>(
    values: &[T],
    indices: &[usize],
    class_name: &str,
) -> Result<Vec<T>, String> {
    indices
        .iter()
        .enumerate()
        .map(|(output_index, &source_index)| {
            values.get(source_index).copied().ok_or_else(|| {
                format!(
                    "{class_name} numeric storage index {source_index} at reorder position {output_index} is out of bounds"
                )
            })
        })
        .collect()
}
