use std::collections::BTreeMap;
use std::ffi::c_void;

use runmat_types::ClassIdentity;
use runmat_value::{
    AdoptedHostAllocation, HostComplexBuffer, HostIndexBuffer, HostLogicalBuffer,
    HostNumericBuffer, NumericDType, NumericStorage,
};

use super::{MxClassId, MxGpuArray, MxGpuLease, MxInterleavedStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxApiMode {
    SeparateComplex,
    InterleavedComplex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum MxBoundaryInterface {
    CMatrix,
    CxxData,
    FortranMatrix,
}

/// Origin-thread value identity carried through a native MEX lane.
///
/// The token is safe to move between threads because it cannot dereference the
/// underlying garbage-collected object. Resolution remains with the invocation
/// context on the originating runtime thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MxHandleToken {
    pub resource: u64,
    pub generation: u64,
    pub class_name: ClassIdentity,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MxNumeric {
    pub real: HostNumericBuffer,
    pub imag: Option<HostNumericBuffer>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MxInterleaved {
    pub values: MxInterleavedStorage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MxSparseValues {
    Numeric(HostNumericBuffer),
    InterleavedComplex(HostComplexBuffer<f64>),
    SeparateComplex {
        real: HostNumericBuffer,
        imaginary: HostNumericBuffer,
    },
    Logical(HostLogicalBuffer),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MxSparse {
    pub rows: usize,
    pub cols: usize,
    pub col_ptrs: HostIndexBuffer,
    pub row_indices: HostIndexBuffer,
    pub values: MxSparseValues,
    pub nzmax: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MxArrayData {
    Numeric(MxNumeric),
    Interleaved(MxInterleaved),
    Logical(HostLogicalBuffer),
    Char(Vec<u16>),
    String(Vec<String>),
    Cell(Vec<Option<Box<MxArray>>>),
    Struct {
        fields: Vec<String>,
        /// Field-major values: `field * numel + element`.
        values: Vec<Option<Box<MxArray>>>,
    },
    Object {
        class_name: ClassIdentity,
        /// Property-major values: `property * numel + element`.
        properties: Vec<String>,
        values: Vec<Option<Box<MxArray>>>,
    },
    Handle(MxHandleToken),
    Sparse(MxSparse),
    Gpu(MxGpuArray),
}

/// Boundary-owned C Matrix API value. C consumers see only an opaque
/// `mxArray *`; this Rust layout is never a public ABI promise.
#[derive(Debug, Clone, PartialEq)]
pub struct MxArray {
    class_id: MxClassId,
    shape: Vec<usize>,
    data: MxArrayData,
    persistent: bool,
}

impl MxArray {
    pub fn deep_duplicate(&self) -> Result<Self, String> {
        let mut duplicate = self.clone();
        duplicate.detach_shared_storage()?;
        Ok(duplicate)
    }

    fn detach_shared_storage(&mut self) -> Result<(), String> {
        match &mut self.data {
            MxArrayData::Numeric(value) => {
                value.real.make_unique();
                if let Some(imaginary) = &mut value.imag {
                    imaginary.make_unique();
                }
            }
            MxArrayData::Cell(values)
            | MxArrayData::Struct { values, .. }
            | MxArrayData::Object { values, .. } => {
                for value in values.iter_mut().filter_map(Option::as_deref_mut) {
                    value.detach_shared_storage()?;
                }
            }
            MxArrayData::Logical(values) => values.make_unique(),
            MxArrayData::Sparse(MxSparse {
                col_ptrs,
                row_indices,
                values: MxSparseValues::Logical(values),
                ..
            }) => {
                col_ptrs.make_unique();
                row_indices.make_unique();
                values.make_unique();
            }
            MxArrayData::Sparse(MxSparse {
                col_ptrs,
                row_indices,
                values: MxSparseValues::Numeric(values),
                ..
            }) => {
                col_ptrs.make_unique();
                row_indices.make_unique();
                values.make_unique();
            }
            MxArrayData::Sparse(MxSparse {
                col_ptrs,
                row_indices,
                values: MxSparseValues::InterleavedComplex(values),
                ..
            }) => {
                col_ptrs.make_unique();
                row_indices.make_unique();
                values.make_unique();
            }
            MxArrayData::Sparse(MxSparse {
                col_ptrs,
                row_indices,
                values: MxSparseValues::SeparateComplex { real, imaginary },
                ..
            }) => {
                col_ptrs.make_unique();
                row_indices.make_unique();
                real.make_unique();
                imaginary.make_unique();
            }
            MxArrayData::Interleaved(value) => match &mut value.values {
                MxInterleavedStorage::F64(values) => values.make_unique(),
                MxInterleavedStorage::F32(values) => values.make_unique(),
                MxInterleavedStorage::I8(_)
                | MxInterleavedStorage::I16(_)
                | MxInterleavedStorage::I32(_)
                | MxInterleavedStorage::I64(_)
                | MxInterleavedStorage::U8(_)
                | MxInterleavedStorage::U16(_)
                | MxInterleavedStorage::U32(_)
                | MxInterleavedStorage::U64(_) => {}
            },
            MxArrayData::Char(values) => runmat_value::record_host_copy(
                runmat_value::HostCopyReason::CopyOnWriteMutation,
                values.len().saturating_mul(std::mem::size_of::<u16>()),
            ),
            MxArrayData::String(values) => runmat_value::record_host_copy(
                runmat_value::HostCopyReason::CopyOnWriteMutation,
                values.iter().map(String::len).sum(),
            ),
            MxArrayData::Handle(_) => {}
            MxArrayData::Gpu(value) => {
                value.lease = value.lease.duplicate().map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }

    pub(crate) fn encoded_storage_bytes(&self) -> usize {
        match &self.data {
            MxArrayData::Char(values) => values.len().saturating_mul(std::mem::size_of::<u16>()),
            MxArrayData::String(values) => values.iter().map(String::len).sum(),
            MxArrayData::Cell(values)
            | MxArrayData::Struct { values, .. }
            | MxArrayData::Object { values, .. } => values
                .iter()
                .filter_map(Option::as_deref)
                .map(Self::encoded_storage_bytes)
                .fold(0usize, usize::saturating_add),
            _ => 0,
        }
    }

    pub fn numeric(
        storage: NumericStorage,
        shape: Vec<usize>,
        complexity: Option<NumericStorage>,
    ) -> Result<Self, String> {
        Self::numeric_buffer(
            HostNumericBuffer::from_numeric_storage(storage),
            shape,
            complexity.map(HostNumericBuffer::from_numeric_storage),
        )
    }

    pub fn numeric_buffer(
        storage: HostNumericBuffer,
        shape: Vec<usize>,
        complexity: Option<HostNumericBuffer>,
    ) -> Result<Self, String> {
        validate_shape(storage.len(), &shape)?;
        if let Some(imag) = &complexity {
            if imag.numeric_dtype() != storage.numeric_dtype() || imag.len() != storage.len() {
                return Err(
                    "separate complex components must have matching class and length".into(),
                );
            }
        }
        Ok(Self {
            class_id: MxClassId::from_numeric_dtype(storage.numeric_dtype()),
            shape,
            data: MxArrayData::Numeric(MxNumeric {
                real: storage,
                imag: complexity,
            }),
            persistent: false,
        })
    }

    pub fn interleaved(values: MxInterleavedStorage, shape: Vec<usize>) -> Result<Self, String> {
        validate_shape(values.len(), &shape)?;
        Ok(Self {
            class_id: MxClassId::from_numeric_dtype(values.dtype()),
            shape,
            data: MxArrayData::Interleaved(MxInterleaved { values }),
            persistent: false,
        })
    }

    pub fn zeros_numeric(
        dtype: NumericDType,
        shape: Vec<usize>,
        complex: bool,
        mode: MxApiMode,
    ) -> Result<Self, String> {
        let len = checked_numel(&shape)?;
        match (complex, mode) {
            (false, _) => Self::numeric(NumericStorage::zeros(dtype, len), shape, None),
            (true, MxApiMode::SeparateComplex) => Self::numeric(
                NumericStorage::zeros(dtype, len),
                shape,
                Some(NumericStorage::zeros(dtype, len)),
            ),
            (true, MxApiMode::InterleavedComplex) => {
                Self::interleaved(MxInterleavedStorage::zeros(dtype, len), shape)
            }
        }
    }

    pub fn logical(values: Vec<u8>, shape: Vec<usize>) -> Result<Self, String> {
        Self::logical_buffer(HostLogicalBuffer::new(values), shape)
    }

    pub fn logical_buffer(values: HostLogicalBuffer, shape: Vec<usize>) -> Result<Self, String> {
        validate_shape(values.len(), &shape)?;
        Ok(Self {
            class_id: MxClassId::Logical,
            shape,
            data: MxArrayData::Logical(values),
            persistent: false,
        })
    }

    pub fn character(values: Vec<u16>, shape: Vec<usize>) -> Result<Self, String> {
        validate_shape(values.len(), &shape)?;
        Ok(Self {
            class_id: MxClassId::Char,
            shape,
            data: MxArrayData::Char(values),
            persistent: false,
        })
    }

    pub fn string(values: Vec<String>, shape: Vec<usize>) -> Result<Self, String> {
        validate_shape(values.len(), &shape)?;
        Ok(Self {
            class_id: MxClassId::Unknown,
            shape,
            data: MxArrayData::String(values),
            persistent: false,
        })
    }

    pub fn cell(values: Vec<Option<Box<Self>>>, shape: Vec<usize>) -> Result<Self, String> {
        validate_shape(values.len(), &shape)?;
        Ok(Self {
            class_id: MxClassId::Cell,
            shape,
            data: MxArrayData::Cell(values),
            persistent: false,
        })
    }

    pub fn structure(
        fields: Vec<String>,
        values: Vec<Option<Box<Self>>>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        if fields.iter().any(|field| field.is_empty()) {
            return Err("struct field names must be non-empty".into());
        }
        let numel = checked_numel(&shape)?;
        let expected = fields
            .len()
            .checked_mul(numel)
            .ok_or_else(|| "struct field storage exceeds platform limits".to_string())?;
        if values.len() != expected {
            return Err(format!(
                "struct field storage length {} does not match {} fields x {numel} elements",
                values.len(),
                fields.len()
            ));
        }
        Ok(Self {
            class_id: MxClassId::Struct,
            shape,
            data: MxArrayData::Struct { fields, values },
            persistent: false,
        })
    }

    pub fn object(
        class_name: ClassIdentity,
        properties: Vec<String>,
        values: Vec<Option<Box<Self>>>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        if properties.iter().any(|property| property.is_empty()) {
            return Err("object property names must be non-empty".into());
        }
        let numel = checked_numel(&shape)?;
        let expected = properties
            .len()
            .checked_mul(numel)
            .ok_or_else(|| "object property storage exceeds platform limits".to_string())?;
        if values.len() != expected {
            return Err(format!(
                "object property storage length {} does not match {} properties x {numel} elements",
                values.len(),
                properties.len()
            ));
        }
        Ok(Self {
            class_id: MxClassId::Object,
            shape,
            data: MxArrayData::Object {
                class_name,
                properties,
                values,
            },
            persistent: false,
        })
    }

    pub fn handle(value: MxHandleToken) -> Self {
        Self {
            class_id: MxClassId::Object,
            shape: vec![1, 1],
            data: MxArrayData::Handle(value),
            persistent: false,
        }
    }

    pub fn gpu_borrowed(
        class_id: MxClassId,
        handle: runmat_accelerate_api::GpuTensorHandle,
    ) -> Result<Self, String> {
        Self::gpu(class_id, MxGpuLease::borrowed(handle))
    }

    pub fn gpu_owned(class_id: MxClassId, lease: MxGpuLease) -> Result<Self, String> {
        Self::gpu(class_id, lease)
    }

    fn gpu(class_id: MxClassId, lease: MxGpuLease) -> Result<Self, String> {
        if !class_id.is_numeric() && class_id != MxClassId::Logical {
            return Err("GPU arrays require a numeric or logical underlying class".into());
        }
        let shape = lease.handle().shape.clone();
        checked_numel(&shape)?;
        Ok(Self {
            class_id: MxClassId::Object,
            shape,
            data: MxArrayData::Gpu(MxGpuArray { class_id, lease }),
            persistent: false,
        })
    }

    pub fn sparse(value: MxSparse) -> Result<Self, String> {
        if value.col_ptrs.len() != value.cols.saturating_add(1)
            || value.col_ptrs.first().copied() != Some(0)
            || value.col_ptrs.last().copied().unwrap_or(usize::MAX) > value.nzmax
        {
            return Err("invalid sparse column pointers".into());
        }
        let value_count = match &value.values {
            MxSparseValues::Numeric(values) => values.len(),
            MxSparseValues::InterleavedComplex(values) => values.len(),
            MxSparseValues::SeparateComplex { real, imaginary } => {
                if real.numeric_dtype() != NumericDType::F64
                    || imaginary.numeric_dtype() != NumericDType::F64
                    || real.len() != imaginary.len()
                {
                    return Err(
                        "separate sparse complex components must be matching doubles".into(),
                    );
                }
                real.len()
            }
            MxSparseValues::Logical(values) => values.len(),
        };
        if value_count != value.nzmax || value.row_indices.len() != value.nzmax {
            return Err("sparse storage length does not match nzmax".into());
        }
        let nnz = value.col_ptrs.last().copied().unwrap_or(0);
        if value.row_indices[..nnz]
            .iter()
            .any(|row| *row >= value.rows)
        {
            return Err("invalid sparse row storage".into());
        }
        let class_id = match &value.values {
            MxSparseValues::Numeric(values) => {
                MxClassId::from_numeric_dtype(values.numeric_dtype())
            }
            MxSparseValues::InterleavedComplex(_) | MxSparseValues::SeparateComplex { .. } => {
                MxClassId::Double
            }
            MxSparseValues::Logical(_) => MxClassId::Logical,
        };
        Ok(Self {
            class_id,
            shape: vec![value.rows, value.cols],
            data: MxArrayData::Sparse(value),
            persistent: false,
        })
    }

    pub const fn class_id(&self) -> MxClassId {
        self.class_id
    }

    pub fn class_name(&self) -> &str {
        match &self.data {
            MxArrayData::Object { class_name, .. } => class_name.display_name(),
            MxArrayData::Handle(value) => value.class_name.display_name(),
            MxArrayData::Gpu(_) => "gpuArray",
            _ => super::super::libmx::class_name(self.class_id),
        }
    }

    pub(crate) fn set_class_id(&mut self, class_id: MxClassId) {
        self.class_id = class_id;
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_complex(&self) -> bool {
        match &self.data {
            MxArrayData::Numeric(MxNumeric { imag: Some(_), .. })
            | MxArrayData::Interleaved(_)
            | MxArrayData::Sparse(MxSparse {
                values:
                    MxSparseValues::InterleavedComplex(_) | MxSparseValues::SeparateComplex { .. },
                ..
            }) => true,
            MxArrayData::Gpu(MxGpuArray { lease, .. }) => {
                lease.handle().descriptor.storage
                    == Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
            }
            _ => false,
        }
    }

    pub const fn is_persistent(&self) -> bool {
        self.persistent
    }

    pub fn make_persistent(&mut self) {
        self.persistent = true;
    }

    pub fn data(&self) -> &MxArrayData {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut MxArrayData {
        &mut self.data
    }

    pub fn dimensions_pointer(&self) -> *const usize {
        self.shape.as_ptr()
    }

    pub fn set_shape(&mut self, shape: Vec<usize>) -> Result<(), String> {
        let requested = checked_numel(&shape)?;
        if matches!(self.data, MxArrayData::Gpu(_)) {
            if requested > self.numel() {
                return Err("GPU dimensions cannot exceed the allocated element count".into());
            }
        } else {
            validate_shape(self.numel(), &shape)?;
        }
        self.shape = shape;
        Ok(())
    }

    pub fn data_pointer(&mut self) -> *mut c_void {
        match &mut self.data {
            MxArrayData::Numeric(value) => numeric_pointer(&value.real),
            MxArrayData::Interleaved(value) => interleaved_pointer(&mut value.values),
            MxArrayData::Logical(values) => {
                // SAFETY: the mxArray retains the invocation-scoped buffer lease.
                unsafe { values.foreign_data_pointer() }
            }
            MxArrayData::Char(values) => values.as_mut_ptr().cast(),
            MxArrayData::Sparse(value) => match &mut value.values {
                MxSparseValues::Numeric(values) => {
                    // SAFETY: the sparse mxArray retains the invocation lease.
                    unsafe { values.foreign_data_pointer() }
                }
                MxSparseValues::InterleavedComplex(values) => {
                    // SAFETY: the sparse mxArray retains the invocation lease.
                    unsafe { values.foreign_data_pointer() }
                }
                MxSparseValues::SeparateComplex { real, .. } => {
                    // SAFETY: the sparse mxArray retains the invocation lease.
                    unsafe { real.foreign_data_pointer() }
                }
                MxSparseValues::Logical(values) => {
                    // SAFETY: the sparse mxArray retains the invocation lease.
                    unsafe { values.foreign_data_pointer() }
                }
            },
            MxArrayData::String(_)
            | MxArrayData::Cell(_)
            | MxArrayData::Struct { .. }
            | MxArrayData::Object { .. }
            | MxArrayData::Handle(_)
            | MxArrayData::Gpu(_) => std::ptr::null_mut(),
        }
    }

    pub fn data_pointer_for_write(&mut self) -> *mut c_void {
        match &mut self.data {
            MxArrayData::Numeric(value) => {
                // SAFETY: the mutable array borrow is retained for the duration
                // of the caller's synchronous write and COW has already detached.
                unsafe { value.real.foreign_data_pointer_mut() }
            }
            MxArrayData::Logical(values) => {
                // SAFETY: the mutable mxArray borrow spans the synchronous write.
                unsafe { values.foreign_data_pointer_mut() }
            }
            MxArrayData::Sparse(value) => match &mut value.values {
                MxSparseValues::Numeric(values) => {
                    // SAFETY: the mutable mxArray borrow spans the synchronous write.
                    unsafe { values.foreign_data_pointer_mut() }
                }
                MxSparseValues::InterleavedComplex(values) => {
                    // SAFETY: the mutable mxArray borrow spans the synchronous write.
                    unsafe { values.foreign_data_pointer_mut() }
                }
                MxSparseValues::SeparateComplex { real, .. } => {
                    // SAFETY: the mutable mxArray borrow spans the synchronous write.
                    unsafe { real.foreign_data_pointer_mut() }
                }
                MxSparseValues::Logical(values) => {
                    // SAFETY: the mutable mxArray borrow spans the synchronous write.
                    unsafe { values.foreign_data_pointer_mut() }
                }
            },
            _ => self.data_pointer(),
        }
    }

    pub fn imaginary_pointer(&mut self) -> *mut c_void {
        match &mut self.data {
            MxArrayData::Numeric(MxNumeric {
                imag: Some(values), ..
            }) => numeric_pointer(values),
            MxArrayData::Sparse(MxSparse {
                values: MxSparseValues::SeparateComplex { imaginary, .. },
                ..
            }) => numeric_pointer(imaginary),
            _ => std::ptr::null_mut(),
        }
    }

    pub fn imaginary_pointer_for_write(&mut self) -> *mut c_void {
        match &mut self.data {
            MxArrayData::Numeric(MxNumeric {
                imag: Some(values), ..
            }) => {
                // SAFETY: the mutable array borrow is retained for the duration
                // of the caller's synchronous write and COW has already detached.
                unsafe { values.foreign_data_pointer_mut() }
            }
            MxArrayData::Sparse(MxSparse {
                values: MxSparseValues::SeparateComplex { imaginary, .. },
                ..
            }) => {
                // SAFETY: the mutable mxArray borrow spans the synchronous write.
                unsafe { imaginary.foreign_data_pointer_mut() }
            }
            _ => std::ptr::null_mut(),
        }
    }

    pub fn data_byte_len(&self) -> Option<usize> {
        match &self.data {
            MxArrayData::Numeric(value) => value.real.checked_byte_len(),
            MxArrayData::Interleaved(value) => value
                .values
                .len()
                .checked_mul(value.values.dtype().byte_size())?
                .checked_mul(2),
            MxArrayData::Logical(values) => Some(values.len()),
            MxArrayData::Char(values) => values.len().checked_mul(std::mem::size_of::<u16>()),
            MxArrayData::Sparse(value) => match &value.values {
                MxSparseValues::Numeric(values) => values.checked_byte_len(),
                MxSparseValues::InterleavedComplex(values) => values
                    .len()
                    .checked_mul(std::mem::size_of::<runmat_value::ComplexElement<f64>>()),
                MxSparseValues::SeparateComplex { real, .. } => real.checked_byte_len(),
                MxSparseValues::Logical(values) => Some(values.len()),
            },
            MxArrayData::String(_)
            | MxArrayData::Cell(_)
            | MxArrayData::Struct { .. }
            | MxArrayData::Object { .. }
            | MxArrayData::Handle(_)
            | MxArrayData::Gpu(_) => None,
        }
    }

    pub fn imaginary_byte_len(&self) -> Option<usize> {
        match &self.data {
            MxArrayData::Numeric(MxNumeric {
                imag: Some(values), ..
            }) => values.checked_byte_len(),
            MxArrayData::Sparse(MxSparse {
                values: MxSparseValues::SeparateComplex { imaginary, .. },
                ..
            }) => imaginary.checked_byte_len(),
            _ => None,
        }
    }

    pub fn try_replace_with_adopted_data(
        &mut self,
        allocation: AdoptedHostAllocation,
        imaginary: bool,
    ) -> Result<(), (AdoptedHostAllocation, String)> {
        if imaginary {
            return match &mut self.data {
                MxArrayData::Numeric(MxNumeric {
                    real,
                    imag: Some(values),
                }) => {
                    let replacement =
                        HostNumericBuffer::try_adopt(allocation, real.numeric_dtype(), real.len())?;
                    *values = replacement;
                    Ok(())
                }
                MxArrayData::Sparse(MxSparse {
                    values: MxSparseValues::SeparateComplex { real, imaginary },
                    ..
                }) => {
                    let replacement =
                        HostNumericBuffer::try_adopt(allocation, real.numeric_dtype(), real.len())?;
                    *imaginary = replacement;
                    Ok(())
                }
                _ => Err((
                    allocation,
                    "mxArray does not expose separate imaginary storage".into(),
                )),
            };
        }

        match &mut self.data {
            MxArrayData::Numeric(value) => {
                let replacement = HostNumericBuffer::try_adopt(
                    allocation,
                    value.real.numeric_dtype(),
                    value.real.len(),
                )?;
                value.real = replacement;
                Ok(())
            }
            MxArrayData::Interleaved(value) => match &mut value.values {
                MxInterleavedStorage::F64(values) => {
                    *values = HostComplexBuffer::try_adopt(allocation, values.len())?;
                    Ok(())
                }
                MxInterleavedStorage::F32(values) => {
                    *values = HostComplexBuffer::try_adopt(allocation, values.len())?;
                    Ok(())
                }
                _ => Err((
                    allocation,
                    "integer-complex storage is not yet an adoptable canonical buffer".into(),
                )),
            },
            MxArrayData::Logical(values) => {
                let byte_length = values.len();
                *values = HostLogicalBuffer::try_adopt(allocation, byte_length)?;
                Ok(())
            }
            MxArrayData::Sparse(value) => match &mut value.values {
                MxSparseValues::Numeric(values) => {
                    *values = HostNumericBuffer::try_adopt(
                        allocation,
                        values.numeric_dtype(),
                        values.len(),
                    )?;
                    Ok(())
                }
                MxSparseValues::InterleavedComplex(values) => {
                    *values = HostComplexBuffer::try_adopt(allocation, values.len())?;
                    Ok(())
                }
                MxSparseValues::SeparateComplex { real, .. } => {
                    *real =
                        HostNumericBuffer::try_adopt(allocation, real.numeric_dtype(), real.len())?;
                    Ok(())
                }
                MxSparseValues::Logical(values) => {
                    let len = values.len();
                    *values = HostLogicalBuffer::try_adopt(allocation, len)?;
                    Ok(())
                }
            },
            MxArrayData::Char(_)
            | MxArrayData::String(_)
            | MxArrayData::Cell(_)
            | MxArrayData::Struct { .. }
            | MxArrayData::Object { .. } => Err((
                allocation,
                "mxArray data layout requires explicit conversion".into(),
            )),
            MxArrayData::Handle(_) => Err((
                allocation,
                "handle objects do not expose replaceable storage".into(),
            )),
            MxArrayData::Gpu(_) => Err((
                allocation,
                "GPU arrays do not expose replaceable host storage".into(),
            )),
        }
    }

    pub(crate) fn find(&self, pointer: *const Self) -> Option<&Self> {
        if std::ptr::eq(self, pointer) {
            return Some(self);
        }
        match &self.data {
            MxArrayData::Cell(values)
            | MxArrayData::Struct { values, .. }
            | MxArrayData::Object { values, .. } => values
                .iter()
                .filter_map(Option::as_deref)
                .find_map(|value| value.find(pointer)),
            _ => None,
        }
    }

    pub(crate) fn find_mut(&mut self, pointer: *mut Self) -> Option<&mut Self> {
        if std::ptr::eq(self, pointer) {
            return Some(self);
        }
        match &mut self.data {
            MxArrayData::Cell(values)
            | MxArrayData::Struct { values, .. }
            | MxArrayData::Object { values, .. } => values
                .iter_mut()
                .filter_map(Option::as_deref_mut)
                .find_map(|value| value.find_mut(pointer)),
            _ => None,
        }
    }

    pub(crate) fn take_descendant(&mut self, pointer: *mut Self) -> Option<Box<Self>> {
        let values = match &mut self.data {
            MxArrayData::Cell(values)
            | MxArrayData::Struct { values, .. }
            | MxArrayData::Object { values, .. } => values,
            _ => return None,
        };
        for slot in values {
            if slot
                .as_deref()
                .is_some_and(|value| std::ptr::eq(value, pointer))
            {
                return slot.take();
            }
            if let Some(value) = slot.as_deref_mut() {
                if let Some(found) = value.take_descendant(pointer) {
                    return Some(found);
                }
            }
        }
        None
    }

    pub(crate) fn drain_persistent_descendants(
        &mut self,
        retained: &mut BTreeMap<usize, Box<Self>>,
    ) {
        let values = match &mut self.data {
            MxArrayData::Cell(values) | MxArrayData::Struct { values, .. } => values,
            _ => return,
        };
        for slot in values {
            let Some(mut child) = slot.take() else {
                continue;
            };
            if child.is_persistent() {
                let pointer = std::ptr::from_mut(child.as_mut()) as usize;
                retained.insert(pointer, child);
            } else {
                child.drain_persistent_descendants(retained);
            }
        }
    }
}

fn numeric_pointer(values: &HostNumericBuffer) -> *mut c_void {
    // SAFETY: MxArray owns an invocation lease for this buffer. The C Matrix
    // API pointer must not escape the call or mutate a logically read-only input.
    unsafe { values.foreign_data_pointer() }
}

fn interleaved_pointer(values: &mut MxInterleavedStorage) -> *mut c_void {
    match values {
        // SAFETY: MxArray owns an invocation lease for this buffer. As with
        // ordinary numeric inputs, foreign code must not mutate a const input.
        MxInterleavedStorage::F64(values) => unsafe { values.foreign_data_pointer() },
        // SAFETY: same invocation lease as the double-precision case.
        MxInterleavedStorage::F32(values) => unsafe { values.foreign_data_pointer() },
        MxInterleavedStorage::I8(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::I16(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::I32(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::I64(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::U8(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::U16(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::U32(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::U64(values) => values.as_mut_ptr().cast(),
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "mxArray dimensions exceed platform limits".to_string())
    })
}

fn validate_shape(len: usize, shape: &[usize]) -> Result<(), String> {
    let expected = checked_numel(shape)?;
    if len != expected {
        return Err(format!(
            "mxArray data length {len} does not match shape {shape:?} ({expected} elements)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_arrays_are_native_lane_transferable_without_runtime_handles() {
        fn assert_send<T: Send>() {}
        assert_send::<MxArray>();
    }

    #[test]
    fn complex_mode_selects_one_boundary_storage_contract() {
        let separate = MxArray::zeros_numeric(
            NumericDType::U64,
            vec![2, 3],
            true,
            MxApiMode::SeparateComplex,
        )
        .unwrap();
        let interleaved = MxArray::zeros_numeric(
            NumericDType::U64,
            vec![2, 3],
            true,
            MxApiMode::InterleavedComplex,
        )
        .unwrap();
        assert!(matches!(separate.data(), MxArrayData::Numeric(_)));
        assert!(matches!(interleaved.data(), MxArrayData::Interleaved(_)));
        assert_eq!(separate.class_id(), MxClassId::Uint64);
        assert_eq!(interleaved.class_id(), MxClassId::Uint64);
    }

    #[test]
    fn struct_storage_is_field_major_and_shape_checked() {
        let values = (0..6)
            .map(|_| Some(Box::new(MxArray::logical(vec![1], vec![1, 1]).unwrap())))
            .collect();
        let value = MxArray::structure(vec!["x".into(), "y".into()], values, vec![1, 3]).unwrap();
        assert_eq!(value.numel(), 3);
        assert_eq!(value.class_id(), MxClassId::Struct);
    }

    #[test]
    fn deep_duplicate_detaches_shared_dense_storage_recursively() {
        let numeric = HostNumericBuffer::from_numeric_storage(NumericStorage::I32(vec![3, 4]));
        let logical = HostLogicalBuffer::new(vec![1, 0]);
        let source = MxArray::cell(
            vec![
                Some(Box::new(
                    MxArray::numeric_buffer(numeric.clone(), vec![1, 2], None).unwrap(),
                )),
                Some(Box::new(
                    MxArray::logical_buffer(logical.clone(), vec![1, 2]).unwrap(),
                )),
            ],
            vec![1, 2],
        )
        .unwrap();

        let duplicate = source.deep_duplicate().unwrap();
        let MxArrayData::Cell(values) = duplicate.data() else {
            panic!("duplicate must remain a cell array");
        };
        let MxArrayData::Numeric(duplicate_numeric) = values[0].as_deref().unwrap().data() else {
            panic!("first duplicate element must remain numeric");
        };
        let MxArrayData::Logical(duplicate_logical) = values[1].as_deref().unwrap().data() else {
            panic!("second duplicate element must remain logical");
        };
        assert!(!numeric.shares_allocation_with(&duplicate_numeric.real));
        assert!(!logical.shares_allocation_with(duplicate_logical));
    }

    #[test]
    fn deep_duplicate_detaches_sparse_indices_and_values() {
        let columns = HostIndexBuffer::new(vec![0, 1, 2]);
        let rows = HostIndexBuffer::new(vec![0, 1]);
        let values = HostNumericBuffer::from_numeric_storage(NumericStorage::F64(vec![1.0, 2.0]));
        let source = MxArray::sparse(MxSparse {
            rows: 2,
            cols: 2,
            col_ptrs: columns.clone(),
            row_indices: rows.clone(),
            values: MxSparseValues::Numeric(values.clone()),
            nzmax: 2,
        })
        .unwrap();

        let duplicate = source.deep_duplicate().unwrap();
        let MxArrayData::Sparse(duplicate) = duplicate.data() else {
            panic!("duplicate must remain sparse");
        };
        let MxSparseValues::Numeric(duplicate_values) = &duplicate.values else {
            panic!("duplicate must remain numeric");
        };
        assert!(!columns.shares_allocation_with(&duplicate.col_ptrs));
        assert!(!rows.shares_allocation_with(&duplicate.row_indices));
        assert!(!values.shares_allocation_with(duplicate_values));
    }

    #[test]
    fn sparse_complex_storage_reports_layout_and_detaches_on_duplicate() {
        let values: HostComplexBuffer<f64> = vec![(1.0, -2.0), (3.0, 4.0)].into();
        let mut source = MxArray::sparse(MxSparse {
            rows: 2,
            cols: 2,
            col_ptrs: vec![0, 1, 2].into(),
            row_indices: vec![0, 1].into(),
            values: MxSparseValues::InterleavedComplex(values.clone()),
            nzmax: 2,
        })
        .unwrap();
        assert!(source.is_complex());
        assert_eq!(source.class_id(), MxClassId::Double);
        assert_eq!(source.data_byte_len(), Some(32));
        assert_eq!(source.imaginary_byte_len(), None);
        // SAFETY: both owners remain alive and the addresses are not dereferenced.
        assert_eq!(source.data_pointer(), unsafe {
            values.foreign_data_pointer()
        });

        let duplicate = source.deep_duplicate().unwrap();
        let MxArrayData::Sparse(MxSparse {
            values: MxSparseValues::InterleavedComplex(duplicate_values),
            ..
        }) = duplicate.data()
        else {
            panic!("duplicate must remain sparse complex");
        };
        assert!(!values.shares_allocation_with(duplicate_values));
    }
}
