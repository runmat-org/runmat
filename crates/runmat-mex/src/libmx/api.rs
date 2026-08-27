use std::ffi::c_void;

use runmat_value::{record_host_copy, HostCopyReason, HostNumericBuffer, NumericStorage};

use super::memory::MexMemoryRegistry;
use crate::mxarray::MxArrayData;
use crate::{MxApiMode, MxArena, MxArenaError, MxArray, MxClassId, MxSparse, MxSparseValues};

#[derive(Debug)]
pub struct MxApi {
    pub(super) mode: MxApiMode,
    pub(super) arena: MxArena,
    pub(super) memory: MexMemoryRegistry,
}

impl MxApi {
    pub fn new(mode: MxApiMode) -> Self {
        Self {
            mode,
            arena: MxArena::default(),
            memory: MexMemoryRegistry::default(),
        }
    }

    pub const fn mode(&self) -> MxApiMode {
        self.mode
    }

    pub fn arena(&self) -> &MxArena {
        &self.arena
    }

    pub fn arena_mut(&mut self) -> &mut MxArena {
        &mut self.arena
    }

    pub fn allocate(&mut self, value: MxArray) -> *mut MxArray {
        self.arena.allocate(value)
    }

    pub fn make_persistent(&mut self, value: *mut MxArray) -> Result<(), String> {
        self.arena
            .get_mut(value)
            .map_err(|error| error.to_string())?
            .make_persistent();
        Ok(())
    }

    pub fn finish_call(&mut self) {
        self.arena.retain_persistent();
        self.memory.finish_call();
    }

    pub fn allocate_memory(&mut self, byte_length: usize, zeroed: bool) -> *mut c_void {
        self.memory.allocate(byte_length, zeroed)
    }

    pub fn reallocate_memory(&mut self, pointer: *mut c_void, byte_length: usize) -> *mut c_void {
        self.memory.reallocate(pointer, byte_length)
    }

    pub fn free_memory(&mut self, pointer: *mut c_void) -> bool {
        self.memory.free(pointer)
    }

    pub fn make_memory_persistent(&mut self, pointer: *mut c_void) -> bool {
        self.memory.make_persistent(pointer)
    }

    pub fn create_numeric(
        &mut self,
        class_id: MxClassId,
        shape: Vec<usize>,
        complex: bool,
    ) -> Result<*mut MxArray, String> {
        let dtype = class_id
            .numeric_dtype()
            .ok_or_else(|| format!("{class_id:?} is not a numeric class"))?;
        let array = MxArray::zeros_numeric(dtype, shape, complex, self.mode)?;
        Ok(self.arena.allocate(array))
    }

    pub fn create_double_scalar(&mut self, value: f64) -> *mut MxArray {
        self.arena.allocate(
            MxArray::numeric(
                runmat_value::NumericStorage::F64(vec![value]),
                vec![1, 1],
                None,
            )
            .expect("scalar shape"),
        )
    }

    pub fn create_logical(&mut self, shape: Vec<usize>) -> Result<*mut MxArray, String> {
        let len = checked_numel(&shape)?;
        Ok(self.arena.allocate(MxArray::logical(vec![0; len], shape)?))
    }

    pub fn create_char(&mut self, shape: Vec<usize>) -> Result<*mut MxArray, String> {
        let len = checked_numel(&shape)?;
        Ok(self
            .arena
            .allocate(MxArray::character(vec![0; len], shape)?))
    }

    pub fn create_string(&mut self, value: &str) -> Result<*mut MxArray, String> {
        let values = value.encode_utf16().collect::<Vec<_>>();
        let shape = vec![1, values.len()];
        Ok(self.arena.allocate(MxArray::character(values, shape)?))
    }

    pub fn create_cell(&mut self, shape: Vec<usize>) -> Result<*mut MxArray, String> {
        let len = checked_numel(&shape)?;
        Ok(self.arena.allocate(MxArray::cell(vec![None; len], shape)?))
    }

    pub fn create_sparse(
        &mut self,
        rows: usize,
        cols: usize,
        nzmax: usize,
        logical: bool,
    ) -> Result<*mut MxArray, String> {
        let values = if logical {
            MxSparseValues::Logical(vec![0; nzmax].into())
        } else {
            MxSparseValues::Numeric(HostNumericBuffer::from_numeric_storage(
                NumericStorage::F64(vec![0.0; nzmax]),
            ))
        };
        Ok(self.arena.allocate(MxArray::sparse(MxSparse {
            rows,
            cols,
            col_ptrs: vec![0; cols.saturating_add(1)].into(),
            row_indices: vec![0; nzmax].into(),
            values,
            nzmax,
        })?))
    }

    pub fn create_struct(
        &mut self,
        shape: Vec<usize>,
        fields: Vec<String>,
    ) -> Result<*mut MxArray, String> {
        let value_count = checked_numel(&shape)?
            .checked_mul(fields.len())
            .ok_or_else(|| "struct field storage exceeds platform limits".to_string())?;
        Ok(self
            .arena
            .allocate(MxArray::structure(fields, vec![None; value_count], shape)?))
    }

    pub fn duplicate(&mut self, source: *const MxArray) -> Result<*mut MxArray, MxArenaError> {
        let duplicate = self.arena.get(source)?.deep_duplicate();
        Ok(self.arena.allocate(duplicate))
    }

    pub fn destroy(&mut self, value: *mut MxArray) -> Result<(), MxArenaError> {
        self.arena.destroy(value)
    }

    pub fn class_id(&self, value: *const MxArray) -> Result<MxClassId, MxArenaError> {
        self.arena.get(value).map(MxArray::class_id)
    }

    pub fn shape(&self, value: *const MxArray) -> Result<&[usize], MxArenaError> {
        self.arena.get(value).map(MxArray::shape)
    }

    pub fn set_shape(&mut self, value: *mut MxArray, shape: Vec<usize>) -> Result<(), String> {
        self.arena
            .get_mut(value)
            .map_err(|error| error.to_string())?
            .set_shape(shape)
    }

    pub fn numel(&self, value: *const MxArray) -> Result<usize, MxArenaError> {
        self.arena.get(value).map(MxArray::numel)
    }

    pub fn is_complex(&self, value: *const MxArray) -> Result<bool, MxArenaError> {
        self.arena.get(value).map(MxArray::is_complex)
    }

    pub fn is_sparse(&self, value: *const MxArray) -> Result<bool, MxArenaError> {
        self.arena
            .get(value)
            .map(|value| matches!(value.data(), MxArrayData::Sparse(_)))
    }

    pub fn data_pointer(
        &mut self,
        value: *mut MxArray,
        expected: Option<MxClassId>,
    ) -> Result<*mut c_void, String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        if let Some(expected) = expected {
            if value.class_id() != expected {
                return Err(format!(
                    "requested {} data from {} mxArray",
                    class_name(expected),
                    class_name(value.class_id())
                ));
            }
        }
        Ok(value.data_pointer())
    }

    pub fn imaginary_pointer(&mut self, value: *mut MxArray) -> Result<*mut c_void, String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        if self.mode != MxApiMode::SeparateComplex {
            return Err("mxGetPi is unavailable in interleaved-complex mode".into());
        }
        Ok(value.imaginary_pointer())
    }

    /// Adopt a proven compatible C data buffer, or perform one checked copy.
    ///
    /// # Safety
    ///
    /// `source` must reference at least the byte length reported for the
    /// selected real/interleaved or imaginary component and must not overlap
    /// the array's current storage.
    pub unsafe fn replace_data(
        &mut self,
        value: *mut MxArray,
        source: *const c_void,
        imaginary: bool,
    ) -> Result<(), String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let byte_len = if imaginary {
            value.imaginary_byte_len()
        } else {
            value.data_byte_len()
        }
        .ok_or_else(|| "mxArray does not expose replaceable data storage".to_string())?;
        if byte_len > 0 && source.is_null() {
            return Err("replacement data pointer is null".into());
        }
        let current = if imaginary {
            value.imaginary_pointer()
        } else {
            value.data_pointer()
        };
        if current.cast_const() == source {
            return Ok(());
        }

        if let Some(allocation) = self.memory.take_for_adoption(source.cast_mut()) {
            match value.try_replace_with_adopted_data(allocation, imaginary) {
                Ok(()) => return Ok(()),
                Err((allocation, _reason)) => {
                    if allocation.provenance().byte_length < byte_len {
                        return Err(format!(
                            "replacement allocation contains {} bytes but {byte_len} are required",
                            allocation.provenance().byte_length
                        ));
                    }
                    record_host_copy(HostCopyReason::ForeignStorageConversion, byte_len);
                    let destination = if imaginary {
                        value.imaginary_pointer_for_write()
                    } else {
                        value.data_pointer_for_write()
                    };
                    if byte_len > 0 && destination.is_null() {
                        return Err("mxArray data storage is unavailable".into());
                    }
                    if value.class_id() == MxClassId::Logical {
                        // SAFETY: the registry proves both byte ranges. Logical
                        // host storage requires canonical zero/one bytes.
                        let source = unsafe {
                            std::slice::from_raw_parts(allocation.pointer().as_ptr(), byte_len)
                        };
                        let destination = unsafe {
                            std::slice::from_raw_parts_mut(destination.cast::<u8>(), byte_len)
                        };
                        destination
                            .iter_mut()
                            .zip(source)
                            .for_each(|(output, input)| *output = u8::from(*input != 0));
                    } else {
                        // SAFETY: the registry proves the source allocation
                        // length, and the array reports the destination length.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                allocation.pointer().as_ptr(),
                                destination.cast(),
                                byte_len,
                            )
                        };
                    }
                    return Ok(());
                }
            }
        }

        Err(
            "replacement data must come from mxMalloc, mxCalloc, or mxRealloc so ownership and allocation length can be verified"
                .into(),
        )
    }

    /// Adopt a native-width sparse-index allocation, or perform one checked copy.
    ///
    /// # Safety
    ///
    /// `source` must reference at least `nzmax` row indices or `n + 1`
    /// column pointers, according to `columns`, and must not overlap the
    /// array's current index storage.
    pub unsafe fn replace_sparse_indices(
        &mut self,
        value: *mut MxArray,
        source: *const usize,
        columns: bool,
    ) -> Result<(), String> {
        let (len, current) = {
            let value = self
                .arena
                .get_mut(value)
                .map_err(|error| error.to_string())?;
            let MxArrayData::Sparse(value) = value.data_mut() else {
                return Err("mxArray is not sparse".into());
            };
            let destination = if columns {
                &mut value.col_ptrs
            } else {
                &mut value.row_indices
            };
            (destination.len(), destination.as_ptr())
        };
        if len > 0 && source.is_null() {
            return Err("replacement sparse-index pointer is null".into());
        }
        if current == source {
            return Ok(());
        }
        if let Some(allocation) = self.memory.take_for_adoption(source.cast_mut().cast()) {
            match runmat_value::HostIndexBuffer::try_adopt(allocation, len) {
                Ok(replacement) => {
                    let value = self
                        .arena
                        .get_mut(value)
                        .map_err(|error| error.to_string())?;
                    let MxArrayData::Sparse(value) = value.data_mut() else {
                        unreachable!("sparse array was validated before adoption")
                    };
                    if columns {
                        value.col_ptrs = replacement;
                    } else {
                        value.row_indices = replacement;
                    }
                    return Ok(());
                }
                Err((allocation, _reason)) => {
                    let required_bytes = len
                        .checked_mul(std::mem::size_of::<usize>())
                        .ok_or_else(|| "sparse-index byte length overflowed".to_string())?;
                    if allocation.provenance().byte_length < required_bytes {
                        return Err(format!(
                            "sparse-index allocation contains {} bytes but {required_bytes} are required",
                            allocation.provenance().byte_length
                        ));
                    }
                    record_host_copy(HostCopyReason::ForeignStorageConversion, required_bytes);
                    let value = self
                        .arena
                        .get_mut(value)
                        .map_err(|error| error.to_string())?;
                    let MxArrayData::Sparse(value) = value.data_mut() else {
                        unreachable!("sparse array was validated before replacement")
                    };
                    let destination = if columns {
                        &mut value.col_ptrs
                    } else {
                        &mut value.row_indices
                    };
                    // SAFETY: the allocation registry proves the source byte
                    // length, and `destination` has the validated typed length.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            allocation.pointer().as_ptr().cast::<usize>(),
                            destination.as_mut_ptr(),
                            len,
                        )
                    };
                    return Ok(());
                }
            }
        }
        Err(
            "replacement sparse indices must come from mxMalloc, mxCalloc, or mxRealloc so ownership and allocation length can be verified"
                .into(),
        )
    }

    pub fn get_cell(&self, value: *const MxArray, index: usize) -> Result<*mut MxArray, String> {
        let value = self.arena.get(value).map_err(|error| error.to_string())?;
        let MxArrayData::Cell(values) = value.data() else {
            return Err("mxArray is not a cell array".into());
        };
        let value = values
            .get(index)
            .ok_or_else(|| "cell index exceeds array bounds".to_string())?;
        Ok(value
            .as_deref()
            .map(|value: &MxArray| std::ptr::from_ref(value).cast_mut())
            .unwrap_or(std::ptr::null_mut()))
    }

    pub fn set_cell(
        &mut self,
        value: *mut MxArray,
        index: usize,
        child: *mut MxArray,
    ) -> Result<(), String> {
        {
            let value = self.arena.get(value).map_err(|error| error.to_string())?;
            let MxArrayData::Cell(values) = value.data() else {
                return Err("mxArray is not a cell array".into());
            };
            if index >= values.len() {
                return Err("cell index exceeds array bounds".into());
            }
        }
        let child = if child.is_null() {
            None
        } else {
            Some(self.arena.take(child).map_err(|error| error.to_string())?)
        };
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let MxArrayData::Cell(values) = value.data_mut() else {
            unreachable!("validated cell array")
        };
        values[index] = child;
        Ok(())
    }

    pub fn field_names(&self, value: *const MxArray) -> Result<&[String], String> {
        let value = self.arena.get(value).map_err(|error| error.to_string())?;
        let MxArrayData::Struct { fields, .. } = value.data() else {
            return Err("mxArray is not a struct array".into());
        };
        Ok(fields)
    }

    pub fn add_field(&mut self, value: *mut MxArray, field: String) -> Result<usize, String> {
        if field.is_empty() {
            return Err("struct field names must be non-empty".into());
        }
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let numel = value.numel();
        let MxArrayData::Struct { fields, values } = value.data_mut() else {
            return Err("mxArray is not a struct array".into());
        };
        if fields.iter().any(|existing| existing == &field) {
            return Err(format!("struct field '{field}' already exists"));
        }
        let index = fields.len();
        fields.push(field);
        values.extend((0..numel).map(|_| None));
        Ok(index)
    }

    pub fn remove_field(&mut self, value: *mut MxArray, field: usize) -> Result<(), String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let numel = value.numel();
        let MxArrayData::Struct { fields, values } = value.data_mut() else {
            return Err("mxArray is not a struct array".into());
        };
        if field >= fields.len() {
            return Err("struct field number exceeds array bounds".into());
        }
        fields.remove(field);
        let start = field * numel;
        values.drain(start..start + numel);
        Ok(())
    }

    pub fn get_field(
        &self,
        value: *const MxArray,
        element: usize,
        field: usize,
    ) -> Result<*mut MxArray, String> {
        let value = self.arena.get(value).map_err(|error| error.to_string())?;
        let numel = value.numel();
        let MxArrayData::Struct { fields, values } = value.data() else {
            return Err("mxArray is not a struct array".into());
        };
        if element >= numel || field >= fields.len() {
            return Err("struct field index exceeds array bounds".into());
        }
        Ok(values[field * numel + element]
            .as_deref()
            .map(|value: &MxArray| std::ptr::from_ref(value).cast_mut())
            .unwrap_or(std::ptr::null_mut()))
    }

    pub fn set_field(
        &mut self,
        value: *mut MxArray,
        element: usize,
        field: usize,
        child: *mut MxArray,
    ) -> Result<(), String> {
        let numel = {
            let value = self.arena.get(value).map_err(|error| error.to_string())?;
            let MxArrayData::Struct { fields, .. } = value.data() else {
                return Err("mxArray is not a struct array".into());
            };
            if element >= value.numel() || field >= fields.len() {
                return Err("struct field index exceeds array bounds".into());
            }
            value.numel()
        };
        let child = if child.is_null() {
            None
        } else {
            Some(self.arena.take(child).map_err(|error| error.to_string())?)
        };
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let MxArrayData::Struct { values, .. } = value.data_mut() else {
            unreachable!("validated struct array")
        };
        values[field * numel + element] = child;
        Ok(())
    }

    pub fn sparse_indices(
        &mut self,
        value: *mut MxArray,
    ) -> Result<(*mut usize, *mut usize, usize), String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let MxArrayData::Sparse(value) = value.data_mut() else {
            return Err("mxArray is not sparse".into());
        };
        Ok((
            // SAFETY: the mxArray retains both invocation-scoped index leases.
            // Inputs are logically read-only even though the compatibility API
            // exposes mutable pointer types. Explicit replacement and resizing
            // enter the copy-on-write mutation path separately.
            unsafe { value.row_indices.foreign_data_pointer().cast() },
            unsafe { value.col_ptrs.foreign_data_pointer().cast() },
            value.nzmax,
        ))
    }

    pub fn set_nzmax(&mut self, value: *mut MxArray, nzmax: usize) -> Result<(), String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let MxArrayData::Sparse(value) = value.data_mut() else {
            return Err("mxArray is not sparse".into());
        };
        let nnz = value.col_ptrs.last().copied().unwrap_or(0);
        if nzmax < nnz {
            return Err(format!("nzmax {nzmax} is smaller than current nnz {nnz}"));
        }
        value.row_indices.resize(nzmax, 0);
        match &mut value.values {
            MxSparseValues::Numeric(values) => values.resize_zeroed(nzmax),
            MxSparseValues::Logical(values) => values.resize(nzmax, 0),
        }
        value.nzmax = nzmax;
        Ok(())
    }
}

pub const fn class_name(class_id: MxClassId) -> &'static str {
    match class_id {
        MxClassId::Unknown => "unknown",
        MxClassId::Cell => "cell",
        MxClassId::Struct => "struct",
        MxClassId::Logical => "logical",
        MxClassId::Char => "char",
        MxClassId::Void => "void",
        MxClassId::Double => "double",
        MxClassId::Single => "single",
        MxClassId::Int8 => "int8",
        MxClassId::Uint8 => "uint8",
        MxClassId::Int16 => "int16",
        MxClassId::Uint16 => "uint16",
        MxClassId::Int32 => "int32",
        MxClassId::Uint32 => "uint32",
        MxClassId::Int64 => "int64",
        MxClassId::Uint64 => "uint64",
        MxClassId::Function => "function_handle",
        MxClassId::Opaque => "opaque",
        MxClassId::Object => "object",
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "mxArray dimensions exceed platform limits".to_string())
    })
}
