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
    data_api_leases: BTreeMap<usize, usize>,
    call_active: bool,
}

impl MxArena {
    pub fn allocate(&mut self, value: MxArray) -> *mut MxArray {
        let mut value = Box::new(value);
        let pointer = std::ptr::from_mut(value.as_mut());
        self.arrays.insert(pointer as usize, value);
        pointer
    }

    pub fn begin_call(&mut self) {
        self.call_active = true;
    }

    pub fn retain_data_api(&mut self, pointer: *const MxArray) -> Result<(), MxArenaError> {
        let key = self.root_key(pointer)?;
        *self.data_api_leases.entry(key).or_default() += 1;
        Ok(())
    }

    pub fn release_data_api(
        &mut self,
        pointer: *mut MxArray,
        owned: bool,
    ) -> Result<(), MxArenaError> {
        let key = self.root_key(pointer)?;
        let leases = self
            .data_api_leases
            .get_mut(&key)
            .ok_or(MxArenaError::UnknownArray)?;
        *leases = leases.saturating_sub(1);
        if *leases == 0 {
            self.data_api_leases.remove(&key);
            let persistent = self
                .arrays
                .get(&key)
                .is_some_and(|value| value.is_persistent());
            if owned || (!self.call_active && !persistent) {
                self.arrays.remove(&key);
            }
        }
        Ok(())
    }

    pub fn get(&self, pointer: *const MxArray) -> Result<&MxArray, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        self.arrays
            .get(&(pointer as usize))
            .map(Box::as_ref)
            .or_else(|| self.arrays.values().find_map(|value| value.find(pointer)))
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn get_mut(&mut self, pointer: *mut MxArray) -> Result<&mut MxArray, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        let key = pointer as usize;
        if self.arrays.contains_key(&key) {
            return self
                .arrays
                .get_mut(&key)
                .map(Box::as_mut)
                .ok_or(MxArenaError::UnknownArray);
        }
        self.arrays
            .values_mut()
            .find_map(|value| value.find_mut(pointer))
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn destroy(&mut self, pointer: *mut MxArray) -> Result<(), MxArenaError> {
        if pointer.is_null() {
            return Ok(());
        }
        self.take(pointer).map(drop)
    }

    pub fn take(&mut self, pointer: *mut MxArray) -> Result<Box<MxArray>, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        let key = pointer as usize;
        if let Some(value) = self.arrays.remove(&key) {
            self.data_api_leases.remove(&key);
            return Ok(value);
        }
        self.arrays
            .values_mut()
            .find_map(|value| value.take_descendant(pointer))
            .ok_or(MxArenaError::UnknownArray)
    }

    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    pub fn retain_persistent(&mut self) {
        self.call_active = false;
        let arrays = std::mem::take(&mut self.arrays);
        let mut retained = BTreeMap::new();
        for (pointer, mut value) in arrays {
            if value.is_persistent() || self.data_api_leases.contains_key(&pointer) {
                retained.insert(pointer, value);
            } else {
                value.drain_persistent_descendants(&mut retained);
            }
        }
        self.arrays = retained;
        self.data_api_leases
            .retain(|pointer, _| self.arrays.contains_key(pointer));
    }

    fn root_key(&self, pointer: *const MxArray) -> Result<usize, MxArenaError> {
        if pointer.is_null() {
            return Err(MxArenaError::NullArray);
        }
        let direct = pointer as usize;
        if self.arrays.contains_key(&direct) {
            return Ok(direct);
        }
        self.arrays
            .iter()
            .find_map(|(key, value)| value.find(pointer).map(|_| *key))
            .ok_or(MxArenaError::UnknownArray)
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

    #[test]
    fn data_api_lease_keeps_a_transient_array_alive_past_the_call() {
        let mut arena = MxArena::default();
        arena.begin_call();
        let pointer = arena.allocate(MxArray::logical(vec![1], vec![1, 1]).unwrap());
        arena.retain_data_api(pointer).unwrap();
        arena.retain_persistent();
        assert!(arena.get(pointer).is_ok());
        arena.release_data_api(pointer, false).unwrap();
        assert_eq!(arena.get(pointer), Err(MxArenaError::UnknownArray));
    }

    #[test]
    fn owned_data_api_control_releases_during_an_active_call() {
        let mut arena = MxArena::default();
        arena.begin_call();
        let pointer = arena.allocate(MxArray::logical(vec![1], vec![1, 1]).unwrap());
        arena.retain_data_api(pointer).unwrap();
        arena.release_data_api(pointer, true).unwrap();
        assert_eq!(arena.get(pointer), Err(MxArenaError::UnknownArray));
    }
}
