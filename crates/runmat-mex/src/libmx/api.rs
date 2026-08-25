use std::ffi::c_void;

use runmat_value::NumericDType;

use crate::{MxApiMode, MxArena, MxArenaError, MxArray, MxClassId};

#[derive(Debug)]
pub struct MxApi {
    mode: MxApiMode,
    arena: MxArena,
}

impl MxApi {
    pub fn new(mode: MxApiMode) -> Self {
        Self {
            mode,
            arena: MxArena::default(),
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

    pub fn duplicate(&mut self, source: *const MxArray) -> Result<*mut MxArray, MxArenaError> {
        let duplicate = self.arena.get(source)?.clone();
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

    pub fn data_pointer(
        &mut self,
        value: *mut MxArray,
        expected: Option<NumericDType>,
    ) -> Result<*mut c_void, String> {
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        if let Some(expected) = expected {
            if value.class_id().numeric_dtype() != Some(expected) {
                return Err(format!(
                    "requested {} data from {} mxArray",
                    expected.class_name(),
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
