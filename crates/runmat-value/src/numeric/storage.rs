use super::view::{gather_numeric_values, set_numeric_element};
use super::*;

/// Authoritative homogeneous host storage for a real numeric array.
///
/// Each variant owns values in the native Rust representation for its MATLAB
/// numeric class. Consumers must dispatch on the variant or request a matching
/// typed slice; this type deliberately provides no implicit `f64` view.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericStorage {
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

impl NumericStorage {
    pub fn zeros(dtype: NumericDType, len: usize) -> Self {
        match dtype {
            NumericDType::F64 => Self::F64(vec![0.0; len]),
            NumericDType::F32 => Self::F32(vec![0.0; len]),
            NumericDType::I8 => Self::I8(vec![0; len]),
            NumericDType::I16 => Self::I16(vec![0; len]),
            NumericDType::I32 => Self::I32(vec![0; len]),
            NumericDType::I64 => Self::I64(vec![0; len]),
            NumericDType::U8 => Self::U8(vec![0; len]),
            NumericDType::U16 => Self::U16(vec![0; len]),
            NumericDType::U32 => Self::U32(vec![0; len]),
            NumericDType::U64 => Self::U64(vec![0; len]),
        }
    }

    pub fn ones(dtype: NumericDType, len: usize) -> Self {
        match dtype {
            NumericDType::F64 => Self::F64(vec![1.0; len]),
            NumericDType::F32 => Self::F32(vec![1.0; len]),
            NumericDType::I8 => Self::I8(vec![1; len]),
            NumericDType::I16 => Self::I16(vec![1; len]),
            NumericDType::I32 => Self::I32(vec![1; len]),
            NumericDType::I64 => Self::I64(vec![1; len]),
            NumericDType::U8 => Self::U8(vec![1; len]),
            NumericDType::U16 => Self::U16(vec![1; len]),
            NumericDType::U32 => Self::U32(vec![1; len]),
            NumericDType::U64 => Self::U64(vec![1; len]),
        }
    }

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

    pub fn class_name(&self) -> &'static str {
        self.numeric_dtype().class_name()
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

    pub fn checked_byte_len(&self) -> Option<usize> {
        self.len().checked_mul(self.numeric_dtype().byte_size())
    }

    pub fn value_at(&self, index: usize) -> Option<NumericScalar> {
        match self {
            Self::F64(values) => values.get(index).copied().map(NumericScalar::F64),
            Self::F32(values) => values.get(index).copied().map(NumericScalar::F32),
            Self::I8(values) => values.get(index).copied().map(NumericScalar::I8),
            Self::I16(values) => values.get(index).copied().map(NumericScalar::I16),
            Self::I32(values) => values.get(index).copied().map(NumericScalar::I32),
            Self::I64(values) => values.get(index).copied().map(NumericScalar::I64),
            Self::U8(values) => values.get(index).copied().map(NumericScalar::U8),
            Self::U16(values) => values.get(index).copied().map(NumericScalar::U16),
            Self::U32(values) => values.get(index).copied().map(NumericScalar::U32),
            Self::U64(values) => values.get(index).copied().map(NumericScalar::U64),
        }
    }

    /// Stores a same-class scalar without numeric conversion.
    pub fn set_value(&mut self, index: usize, value: NumericScalar) -> Result<(), String> {
        let storage_class = self.class_name();
        let value_class = value.class_name();
        match (self, value) {
            (Self::F64(values), NumericScalar::F64(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::F32(values), NumericScalar::F32(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::I8(values), NumericScalar::I8(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::I16(values), NumericScalar::I16(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::I32(values), NumericScalar::I32(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::I64(values), NumericScalar::I64(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::U8(values), NumericScalar::U8(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::U16(values), NumericScalar::U16(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::U32(values), NumericScalar::U32(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            (Self::U64(values), NumericScalar::U64(value)) => {
                set_numeric_element(values, index, value, storage_class)
            }
            _ => Err(format!(
                "cannot store {value_class} in {storage_class} numeric storage"
            )),
        }
    }

    pub fn view(&self) -> NumericStorageView<'_> {
        match self {
            Self::F64(values) => NumericStorageView::F64(values),
            Self::F32(values) => NumericStorageView::F32(values),
            Self::I8(values) => NumericStorageView::I8(values),
            Self::I16(values) => NumericStorageView::I16(values),
            Self::I32(values) => NumericStorageView::I32(values),
            Self::I64(values) => NumericStorageView::I64(values),
            Self::U8(values) => NumericStorageView::U8(values),
            Self::U16(values) => NumericStorageView::U16(values),
            Self::U32(values) => NumericStorageView::U32(values),
            Self::U64(values) => NumericStorageView::U64(values),
        }
    }

    pub fn view_mut(&mut self) -> NumericStorageViewMut<'_> {
        match self {
            Self::F64(values) => NumericStorageViewMut::F64(values),
            Self::F32(values) => NumericStorageViewMut::F32(values),
            Self::I8(values) => NumericStorageViewMut::I8(values),
            Self::I16(values) => NumericStorageViewMut::I16(values),
            Self::I32(values) => NumericStorageViewMut::I32(values),
            Self::I64(values) => NumericStorageViewMut::I64(values),
            Self::U8(values) => NumericStorageViewMut::U8(values),
            Self::U16(values) => NumericStorageViewMut::U16(values),
            Self::U32(values) => NumericStorageViewMut::U32(values),
            Self::U64(values) => NumericStorageViewMut::U64(values),
        }
    }

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            Self::F64(values) => Some(values),
            Self::F32(_)
            | Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_) => None,
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::F64(_)
            | Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_) => None,
        }
    }

    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        match self {
            Self::F64(values) => Some(values),
            Self::F32(_)
            | Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_) => None,
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::F64(_)
            | Self::I8(_)
            | Self::I16(_)
            | Self::I32(_)
            | Self::I64(_)
            | Self::U8(_)
            | Self::U16(_)
            | Self::U32(_)
            | Self::U64(_) => None,
        }
    }

    pub fn validate_shape(&self, shape: &[usize]) -> Result<(), String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        let Some(expected) = expected else {
            return Err(format!("numeric tensor shape {shape:?} overflows usize"));
        };
        if self.len() != expected {
            return Err(format!(
                "{} storage length {} doesn't match shape {:?} ({} elements)",
                self.class_name(),
                self.len(),
                shape,
                expected
            ));
        }
        Ok(())
    }

    pub fn zeros_like(&self, len: usize) -> Self {
        Self::zeros(self.numeric_dtype(), len)
    }

    pub fn ones_like(&self, len: usize) -> Self {
        Self::ones(self.numeric_dtype(), len)
    }

    /// Resizes storage in class, filling new elements with exact zero.
    pub fn resize_zeroed(&mut self, len: usize) {
        match self {
            Self::F64(values) => values.resize(len, 0.0),
            Self::F32(values) => values.resize(len, 0.0),
            Self::I8(values) => values.resize(len, 0),
            Self::I16(values) => values.resize(len, 0),
            Self::I32(values) => values.resize(len, 0),
            Self::I64(values) => values.resize(len, 0),
            Self::U8(values) => values.resize(len, 0),
            Self::U16(values) => values.resize(len, 0),
            Self::U32(values) => values.resize(len, 0),
            Self::U64(values) => values.resize(len, 0),
        }
    }

    /// Removes zero-based positions while preserving class and relative order.
    pub fn remove_positions(&mut self, positions: &[usize]) -> Result<(), String> {
        let mut positions = positions.to_vec();
        positions.sort_unstable();
        positions.dedup();
        if let Some(&position) = positions.last() {
            if position >= self.len() {
                return Err(format!(
                    "{} numeric storage removal index {} is out of bounds for {} elements",
                    self.class_name(),
                    position,
                    self.len()
                ));
            }
        }
        positions.reverse();
        macro_rules! remove_positions {
            ($values:expr) => {
                for &position in &positions {
                    $values.remove(position);
                }
            };
        }
        match self {
            Self::F64(values) => remove_positions!(values),
            Self::F32(values) => remove_positions!(values),
            Self::I8(values) => remove_positions!(values),
            Self::I16(values) => remove_positions!(values),
            Self::I32(values) => remove_positions!(values),
            Self::I64(values) => remove_positions!(values),
            Self::U8(values) => remove_positions!(values),
            Self::U16(values) => remove_positions!(values),
            Self::U32(values) => remove_positions!(values),
            Self::U64(values) => remove_positions!(values),
        }
        Ok(())
    }

    /// Clones storage for a new shape after validating its element count.
    ///
    /// Shape remains container metadata rather than duplicated storage state.
    pub fn clone_for_shape(&self, shape: &[usize]) -> Result<Self, String> {
        self.validate_shape(shape)?;
        Ok(self.clone())
    }

    /// Selects elements by zero-based flat index without changing class.
    pub fn gather(&self, indices: &[usize]) -> Result<Self, String> {
        let class_name = self.class_name();
        match self {
            Self::F64(values) => gather_numeric_values(values, indices, class_name).map(Self::F64),
            Self::F32(values) => gather_numeric_values(values, indices, class_name).map(Self::F32),
            Self::I8(values) => gather_numeric_values(values, indices, class_name).map(Self::I8),
            Self::I16(values) => gather_numeric_values(values, indices, class_name).map(Self::I16),
            Self::I32(values) => gather_numeric_values(values, indices, class_name).map(Self::I32),
            Self::I64(values) => gather_numeric_values(values, indices, class_name).map(Self::I64),
            Self::U8(values) => gather_numeric_values(values, indices, class_name).map(Self::U8),
            Self::U16(values) => gather_numeric_values(values, indices, class_name).map(Self::U16),
            Self::U32(values) => gather_numeric_values(values, indices, class_name).map(Self::U32),
            Self::U64(values) => gather_numeric_values(values, indices, class_name).map(Self::U64),
        }
    }

    /// Reorders all elements using zero-based flat source indices.
    pub fn reorder(&self, indices: &[usize]) -> Result<Self, String> {
        if indices.len() != self.len() {
            return Err(format!(
                "{} reorder has {} indices for {} elements",
                self.class_name(),
                indices.len(),
                self.len()
            ));
        }
        self.gather(indices)
    }

    /// Explicitly materializes this storage in the `f64` computation domain.
    ///
    /// Integer values outside the exact binary64 range may lose precision.
    pub fn materialize_f64(&self) -> Vec<f64> {
        match self {
            Self::F64(values) => values.clone(),
            Self::F32(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::I8(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::I16(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::I32(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::I64(values) => values.iter().map(|&value| value as f64).collect(),
            Self::U8(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::U16(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::U32(values) => values.iter().map(|&value| f64::from(value)).collect(),
            Self::U64(values) => values.iter().map(|&value| value as f64).collect(),
        }
    }

    /// Explicitly materializes this storage in the `f32` computation domain.
    ///
    /// Wider floating and integer values may lose precision or overflow.
    pub fn materialize_f32(&self) -> Vec<f32> {
        match self {
            Self::F64(values) => values.iter().map(|&value| value as f32).collect(),
            Self::F32(values) => values.clone(),
            Self::I8(values) => values.iter().map(|&value| f32::from(value)).collect(),
            Self::I16(values) => values.iter().map(|&value| f32::from(value)).collect(),
            Self::I32(values) => values.iter().map(|&value| value as f32).collect(),
            Self::I64(values) => values.iter().map(|&value| value as f32).collect(),
            Self::U8(values) => values.iter().map(|&value| f32::from(value)).collect(),
            Self::U16(values) => values.iter().map(|&value| f32::from(value)).collect(),
            Self::U32(values) => values.iter().map(|&value| value as f32).collect(),
            Self::U64(values) => values.iter().map(|&value| value as f32).collect(),
        }
    }

    pub fn from_integer_storage(storage: IntegerStorage) -> Self {
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

    pub fn into_integer_storage(self) -> Result<IntegerStorage, Self> {
        match self {
            Self::I8(values) => Ok(IntegerStorage::I8(values)),
            Self::I16(values) => Ok(IntegerStorage::I16(values)),
            Self::I32(values) => Ok(IntegerStorage::I32(values)),
            Self::I64(values) => Ok(IntegerStorage::I64(values)),
            Self::U8(values) => Ok(IntegerStorage::U8(values)),
            Self::U16(values) => Ok(IntegerStorage::U16(values)),
            Self::U32(values) => Ok(IntegerStorage::U32(values)),
            Self::U64(values) => Ok(IntegerStorage::U64(values)),
            storage @ (Self::F64(_) | Self::F32(_)) => Err(storage),
        }
    }
}
