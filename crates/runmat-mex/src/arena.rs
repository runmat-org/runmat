use std::collections::BTreeMap;
use std::fmt;

use crate::MxArray;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MxArenaError {
    NullArray,
    UnknownArray,
}

impl fmt::Display for MxArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullArray => write!(formatter, "mxArray pointer is null"),
            Self::UnknownArray => write!(formatter, "mxArray is not owned by this call arena"),
        }
    }
}

impl std::error::Error for MxArenaError {}

/// Owns temporary `mxArray` allocations for one MEX invocation.
///
/// The map key is the stable address of the boxed opaque object. Lookup never
/// dereferences a caller-supplied pointer before proving ownership.
#[derive(Debug, Default)]
pub struct MxArena {
    arrays: BTreeMap<usize, Box<MxArray>>,
}

impl MxArena {
    pub fn allocate(&mut self, value: MxArray) -> *mut MxArray {
        let mut value = Box::new(value);
        let pointer = std::ptr::from_mut(value.as_mut());
        self.arrays.insert(pointer as usize, value);
        pointer
    }

    pub fn get(&self, pointer: *const MxArray) -> Result<&MxArray, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        self.arrays
            .get(&(pointer as usize))
            .map(Box::as_ref)
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn get_mut(&mut self, pointer: *mut MxArray) -> Result<&mut MxArray, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        self.arrays
            .get_mut(&(pointer as usize))
            .map(Box::as_mut)
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn destroy(&mut self, pointer: *mut MxArray) -> Result<(), MxArenaError> {
        if pointer.is_null() {
            return Ok(());
        }
        self.arrays
            .remove(&(pointer as usize))
            .map(drop)
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn take(&mut self, pointer: *mut MxArray) -> Result<Box<MxArray>, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        self.arrays
            .remove(&(pointer as usize))
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_rejects_foreign_and_released_pointers_without_dereferencing() {
        let mut arena = MxArena::default();
        let pointer = arena.allocate(MxArray::logical(vec![1], vec![1, 1]).unwrap());
        assert!(arena.get(pointer).is_ok());
        arena.destroy(pointer).unwrap();
        assert_eq!(arena.get(pointer), Err(MxArenaError::UnknownArray));
        assert_eq!(
            arena.get(std::ptr::dangling::<MxArray>()),
            Err(MxArenaError::UnknownArray)
        );
    }
}
