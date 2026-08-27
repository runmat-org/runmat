use std::collections::BTreeMap;
use std::ffi::c_void;

use runmat_value::{
    HostIndexBuffer, HostLogicalBuffer, HostNumericBuffer, NumericDType, NumericStorage,
};

use super::{MxClassId, MxInterleavedStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxApiMode {
    SeparateComplex,
    InterleavedComplex,
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
    Cell(Vec<Option<Box<MxArray>>>),
    Struct {
        fields: Vec<String>,
        /// Field-major values: `field * numel + element`.
        values: Vec<Option<Box<MxArray>>>,
    },
    Object {
        class_name: String,
        /// Property-major values: `property * numel + element`.
        properties: Vec<String>,
        values: Vec<Option<Box<MxArray>>>,
    },
    Sparse(MxSparse),
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
    pub fn deep_duplicate(&self) -> Self {
        let mut duplicate = self.clone();
        duplicate.detach_shared_storage();
        duplicate
    }

    fn detach_shared_storage(&mut self) {
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
                    value.detach_shared_storage();
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
            MxArrayData::Interleaved(_) | MxArrayData::Char(_) => {}
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
        class_name: String,
        properties: Vec<String>,
        values: Vec<Option<Box<Self>>>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        if class_name.is_empty() {
            return Err("object class name must be non-empty".into());
        }
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

    pub fn sparse(value: MxSparse) -> Result<Self, String> {
        if value.col_ptrs.len() != value.cols.saturating_add(1)
            || value.col_ptrs.first().copied() != Some(0)
            || value.col_ptrs.last().copied().unwrap_or(usize::MAX) > value.nzmax
        {
            return Err("invalid sparse column pointers".into());
        }
        let value_count = match &value.values {
            MxSparseValues::Numeric(values) => values.len(),
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
            MxArrayData::Object { class_name, .. } => class_name,
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
        matches!(
            &self.data,
            MxArrayData::Numeric(MxNumeric { imag: Some(_), .. }) | MxArrayData::Interleaved(_)
        )
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
        validate_shape(self.numel(), &shape)?;
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
                MxSparseValues::Logical(values) => {
                    // SAFETY: the sparse mxArray retains the invocation lease.
                    unsafe { values.foreign_data_pointer() }
                }
            },
            MxArrayData::Cell(_) | MxArrayData::Struct { .. } | MxArrayData::Object { .. } => {
                std::ptr::null_mut()
            }
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
                MxSparseValues::Logical(values) => Some(values.len()),
            },
            MxArrayData::Cell(_) | MxArrayData::Struct { .. } | MxArrayData::Object { .. } => None,
        }
    }

    pub fn imaginary_byte_len(&self) -> Option<usize> {
        match &self.data {
            MxArrayData::Numeric(MxNumeric {
                imag: Some(values), ..
            }) => values.checked_byte_len(),
            _ => None,
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
        MxInterleavedStorage::F64(values) => values.as_mut_ptr().cast(),
        MxInterleavedStorage::F32(values) => values.as_mut_ptr().cast(),
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

        let duplicate = source.deep_duplicate();
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

        let duplicate = source.deep_duplicate();
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
}
