//! Exact CSC buffers which retain their own physical claims through validation.

use super::*;
use crate::{
    ComplexElement, HostComplexBuffer, HostNumericBuffer, SparseCscValidation, SparseTensor,
};
use std::num::NonZeroUsize;

#[derive(Debug)]
pub enum ExactHostSparsePublicationError {
    Allocation(ExactHostNumericAllocationError),
    Invalid(String),
}
impl std::fmt::Display for ExactHostSparsePublicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allocation(error) => write!(f, "{error}"),
            Self::Invalid(error) => write!(f, "{error}"),
        }
    }
}
impl std::error::Error for ExactHostSparsePublicationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactHostSparseClass {
    Logical,
    Numeric(NumericDType),
    Complex(NumericDType),
}

impl ExactHostSparseClass {
    fn validate(self) -> Result<(), ExactHostNumericAllocationError> {
        if matches!(self, Self::Complex(dtype) if !matches!(dtype, NumericDType::F32 | NumericDType::F64))
        {
            return Err(ExactHostNumericAllocationError::UnsupportedClass);
        }
        Ok(())
    }

    pub const fn retained_claim_count(self) -> usize {
        match self {
            Self::Logical => 2,
            Self::Numeric(_) | Self::Complex(_) => 3,
        }
    }
}

enum ExactComplexAllocation {
    F32(AdoptedHostAllocation, usize),
    F64(AdoptedHostAllocation, usize),
}

impl ExactComplexAllocation {
    fn requested_bytes(
        dtype: NumericDType,
        len: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let (payload, metadata) = match dtype {
            NumericDType::F32 => (
                Layout::array::<ComplexElement<f32>>(len),
                HostComplexBuffer::<f32>::exact_metadata_bytes(),
            ),
            NumericDType::F64 => (
                Layout::array::<ComplexElement<f64>>(len),
                HostComplexBuffer::<f64>::exact_metadata_bytes(),
            ),
            _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
        };
        payload
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size()
            .checked_add(Layout::new::<RustExactRelease>().size())
            .and_then(|bytes| bytes.checked_add(metadata))
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    fn new(
        dtype: NumericDType,
        len: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let layout = match dtype {
            NumericDType::F32 => Layout::array::<ComplexElement<f32>>(len),
            NumericDType::F64 => Layout::array::<ComplexElement<f64>>(len),
            _ => return Err(ExactHostNumericAllocationError::UnsupportedClass),
        }
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
        let claim = permit.reserve_exact_claim(Self::requested_bytes(dtype, len)?)?;
        let release = RustExactRelease::try_new(layout, claim)?;
        let pointer = if layout.size() == 0 {
            match dtype {
                NumericDType::F32 => NonNull::<ComplexElement<f32>>::dangling().cast(),
                NumericDType::F64 => NonNull::<ComplexElement<f64>>::dangling().cast(),
                _ => unreachable!("floating complex class validated"),
            }
        } else {
            NonNull::new(unsafe { alloc_zeroed(layout) })
                .ok_or(ExactHostNumericAllocationError::AllocationFailed)?
        };
        let provenance = HostAllocationProvenance {
            allocator: Cow::Borrowed("runmat.exact-host-sparse-complex.v1"),
            byte_length: layout.size(),
            capacity_bytes: layout.size(),
            alignment: layout.align(),
        };
        let allocation = unsafe { AdoptedHostAllocation::new_owned(pointer, provenance, release) };
        Ok(match dtype {
            NumericDType::F32 => Self::F32(allocation, len),
            NumericDType::F64 => Self::F64(allocation, len),
            _ => unreachable!("floating complex class validated"),
        })
    }

    fn set(&mut self, index: usize, real: f64, imag: f64) -> Result<(), String> {
        match self {
            Self::F32(allocation, len) if index < *len => {
                unsafe {
                    allocation
                        .pointer()
                        .as_ptr()
                        .cast::<ComplexElement<f32>>()
                        .add(index)
                        .write(ComplexElement(real as f32, imag as f32));
                }
                Ok(())
            }
            Self::F64(allocation, len) if index < *len => {
                unsafe {
                    allocation
                        .pointer()
                        .as_ptr()
                        .cast::<ComplexElement<f64>>()
                        .add(index)
                        .write(ComplexElement(real, imag));
                }
                Ok(())
            }
            _ => Err("sparse complex value index exceeds admitted nnz".into()),
        }
    }

    fn set_f32(&mut self, index: usize, value: ComplexElement<f32>) -> Result<(), String> {
        let Self::F32(allocation, len) = self else {
            return Err("sparse output is not single-precision complex".into());
        };
        if index >= *len {
            return Err("sparse complex value index exceeds admitted nnz".into());
        }
        // SAFETY: the builder exclusively owns this initialized native F32
        // complex allocation, and the index was checked against its length.
        unsafe {
            allocation
                .pointer()
                .as_ptr()
                .cast::<ComplexElement<f32>>()
                .add(index)
                .write(value);
        }
        Ok(())
    }

    fn into_buffer(self) -> Result<ReadyValues, ExactHostNumericAllocationError> {
        match self {
            Self::F32(allocation, len) => {
                HostComplexBuffer::<f32>::try_adopt_exact_fallible(allocation, len)
                    .map(ReadyValues::ComplexF32)
            }
            Self::F64(allocation, len) => {
                HostComplexBuffer::<f64>::try_adopt_exact_fallible(allocation, len)
                    .map(ReadyValues::ComplexF64)
            }
        }
    }
}

enum Values {
    Logical,
    Numeric(ExactHostNumericAllocation),
    Complex(ExactComplexAllocation),
}

enum ReadyValues {
    Logical,
    Numeric(HostNumericBuffer),
    ComplexF32(HostComplexBuffer<f32>),
    ComplexF64(HostComplexBuffer<f64>),
}

pub struct ExactHostSparseTensorAllocation {
    rows: usize,
    cols: usize,
    nnz: usize,
    col_ptrs: ExactHostIndexBufferAllocation,
    row_indices: ExactHostIndexBufferAllocation,
    values: Values,
}

impl ExactHostSparseTensorAllocation {
    pub fn requested_bytes_for_dimensions(
        class: ExactHostSparseClass,
        rows: usize,
        cols: usize,
        nnz: usize,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        class.validate()?;
        if (rows == 0 || cols == 0) && nnz != 0 {
            return Err(ExactHostNumericAllocationError::LayoutOverflow);
        }
        // CSC storage does not materialize the dense rows × cols product.
        // If that product exceeds usize, every representable nnz remains
        // within its mathematical capacity.
        if rows
            .checked_mul(cols)
            .is_some_and(|capacity| nnz > capacity)
        {
            return Err(ExactHostNumericAllocationError::LayoutOverflow);
        }
        let pointers = cols
            .checked_add(1)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let mut bytes = ExactHostIndexBufferAllocation::requested_bytes(pointers)?
            .checked_add(ExactHostIndexBufferAllocation::requested_bytes(nnz)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let values = match class {
            ExactHostSparseClass::Logical => 0,
            ExactHostSparseClass::Numeric(dtype) => {
                ExactHostNumericAllocation::requested_bytes(dtype, nnz)?
            }
            ExactHostSparseClass::Complex(dtype) => {
                ExactComplexAllocation::requested_bytes(dtype, nnz)?
            }
        };
        bytes = bytes
            .checked_add(values)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        Ok(bytes)
    }

    pub fn new(
        class: ExactHostSparseClass,
        rows: usize,
        cols: usize,
        nnz: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        Self::requested_bytes_for_dimensions(class, rows, cols, nnz)?;
        let pointers = cols
            .checked_add(1)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        let col_ptrs = ExactHostIndexBufferAllocation::new(pointers, permit)?;
        let row_indices = ExactHostIndexBufferAllocation::new(nnz, permit)?;
        let values = match class {
            ExactHostSparseClass::Logical => Values::Logical,
            ExactHostSparseClass::Numeric(dtype) => {
                Values::Numeric(ExactHostNumericAllocation::new(dtype, nnz, permit)?)
            }
            ExactHostSparseClass::Complex(dtype) => {
                Values::Complex(ExactComplexAllocation::new(dtype, nnz, permit)?)
            }
        };
        Ok(Self {
            rows,
            cols,
            nnz,
            col_ptrs,
            row_indices,
            values,
        })
    }

    pub fn set_col_ptr(&mut self, column: usize, offset: usize) -> Result<(), String> {
        let slot = self
            .col_ptrs
            .as_mut_slice()
            .get_mut(column)
            .ok_or_else(|| "sparse column pointer exceeds admitted columns".to_string())?;
        *slot = offset;
        Ok(())
    }

    pub fn set_row_index(&mut self, index: usize, row: usize) -> Result<(), String> {
        let slot = self
            .row_indices
            .as_mut_slice()
            .get_mut(index)
            .ok_or_else(|| "sparse row index exceeds admitted nnz".to_string())?;
        *slot = row;
        Ok(())
    }

    pub fn set_numeric_value(&mut self, index: usize, value: NumericScalar) -> Result<(), String> {
        match &mut self.values {
            Values::Numeric(values) => values.set_value(index, value),
            _ => Err("sparse output is not numeric".into()),
        }
    }

    /// Writes native components without a floating precision conversion,
    /// preserving their bit patterns during lossless value transport.
    pub fn set_complex_f32_value(
        &mut self,
        index: usize,
        value: ComplexElement<f32>,
    ) -> Result<(), String> {
        match &mut self.values {
            Values::Complex(values) => values.set_f32(index, value),
            _ => Err("sparse output is not complex".into()),
        }
    }

    pub fn set_complex_value(&mut self, index: usize, real: f64, imag: f64) -> Result<(), String> {
        match &mut self.values {
            Values::Complex(values) => values.set(index, real, imag),
            _ => Err("sparse output is not complex".into()),
        }
    }

    pub fn into_validation(self) -> Result<PreparedExactSparseTensor, String> {
        self.into_validation_fallible()
            .map_err(|error| error.to_string())
    }

    /// Preserve physical header/publication failures as typed causes. The
    /// original string adapter remains an ordinary compatibility boundary.
    pub fn into_validation_fallible(
        self,
    ) -> Result<PreparedExactSparseTensor, ExactHostSparsePublicationError> {
        use ExactHostSparsePublicationError::{Allocation, Invalid};
        let col_ptrs = self.col_ptrs.into_buffer_fallible().map_err(Allocation)?;
        let row_indices = self
            .row_indices
            .into_buffer_fallible()
            .map_err(Allocation)?;
        let values = match self.values {
            Values::Logical => ReadyValues::Logical,
            Values::Numeric(values) => {
                ReadyValues::Numeric(values.try_into_host_buffer().map_err(Allocation)?)
            }
            Values::Complex(values) => values.into_buffer().map_err(Allocation)?,
        };
        let validation =
            SparseCscValidation::new(self.rows, self.cols, col_ptrs, row_indices, self.nnz)
                .map_err(Invalid)?;
        Ok(PreparedExactSparseTensor { validation, values })
    }
}

pub struct PreparedExactSparseTensor {
    validation: SparseCscValidation,
    values: ReadyValues,
}

impl PreparedExactSparseTensor {
    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<bool, String> {
        self.validation.advance(maximum_work)
    }

    pub fn into_tensor(self) -> Result<SparseTensor, String> {
        let parts = self
            .validation
            .into_validated()
            .map_err(|_| "sparse CSC validation is not complete".to_string())?;
        match self.values {
            ReadyValues::Logical => SparseTensor::from_validated_logical(parts),
            ReadyValues::Numeric(values) => SparseTensor::from_validated_numeric(parts, values),
            ReadyValues::ComplexF32(values) => {
                SparseTensor::from_validated_complex_f32(parts, values)
            }
            ReadyValues::ComplexF64(values) => {
                SparseTensor::from_validated_complex_f64(parts, values)
            }
        }
    }
}
