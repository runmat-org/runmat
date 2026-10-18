use super::allocator::{empty_map, table_layout, RegistryAllocator, RootMap};
use super::RootAdmissionError;
use crate::{GcError, GcHandle, ROOT_ADMISSION};

pub(crate) struct RootRegistry {
    map: RootMap<GcHandle, usize>,
    revision: u64,
    exhausted: bool,
}

/// Opaque retired backing returned to the GC owner for destruction after its
/// registry and admission locks have been released.
pub(crate) struct RetiredRootRegistry {
    _map: RootMap<GcHandle, usize>,
}

impl RootRegistry {
    pub(crate) fn new() -> Self {
        let layout = table_layout::<GcHandle, usize>(1).expect("root table layout");
        let map =
            empty_map(1, RegistryAllocator::unbounded(layout)).expect("root table allocation");
        Self {
            map,
            revision: 0,
            exhausted: false,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
    pub(crate) fn capacity(&self) -> usize {
        self.map.capacity()
    }
    pub(crate) fn keys(&self) -> impl Iterator<Item = &GcHandle> {
        self.map.keys()
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&GcHandle, &usize)> {
        self.map.iter()
    }
    #[cfg(test)]
    pub(crate) fn contains_key(&self, handle: &GcHandle) -> bool {
        self.map.contains_key(handle)
    }
    #[cfg(test)]
    pub(crate) fn get(&self, handle: &GcHandle) -> Option<&usize> {
        self.map.get(handle)
    }
    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.advance_revision();
    }
    pub(super) fn plan_revision(&self) -> Result<u64, RootAdmissionError> {
        (!self.exhausted)
            .then_some(self.revision)
            .ok_or(RootAdmissionError::RevisionExhausted)
    }
    pub(super) fn validate_plan(&self, revision: u64) -> Result<(), RootAdmissionError> {
        if self.exhausted {
            return Err(RootAdmissionError::RevisionExhausted);
        }
        if self.revision != revision {
            return Err(RootAdmissionError::StalePlan);
        }
        self.revision
            .checked_add(1)
            .ok_or(RootAdmissionError::RevisionExhausted)
            .map(|_| ())
    }
    pub(super) fn validate_increment(&self, handle: GcHandle) -> Result<(), RootAdmissionError> {
        self.map
            .get(&handle)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(RootAdmissionError::RegistrationCountOverflow)
            .map(|_| ())
    }
    pub(super) fn commit_prepared(
        &mut self,
        handles: &[GcHandle],
        replacement: Option<RootMap<GcHandle, usize>>,
    ) -> Option<RootMap<GcHandle, usize>> {
        let retired = if let Some(map) = replacement {
            Some(std::mem::replace(&mut self.map, map))
        } else {
            for handle in handles {
                increment_map(&mut self.map, *handle)
                    .expect("prepared root increments were validated before publication");
            }
            None
        };
        self.advance_revision();
        retired
    }
    pub(crate) fn add_unbounded(
        &mut self,
        handle: GcHandle,
    ) -> crate::Result<Option<RetiredRootRegistry>> {
        self.validate_increment(handle).map_err(root_error)?;
        let mut retired = None;
        if self.map.len() == self.map.capacity() && !self.map.contains_key(&handle) {
            let target = self
                .map
                .len()
                .checked_add(1)
                .ok_or_else(|| root_error(RootAdmissionError::CountOverflow))?;
            let layout = table_layout::<GcHandle, usize>(target)
                .map_err(|_| root_error(RootAdmissionError::LayoutOverflow))?;
            let mut replacement = empty_map(target, RegistryAllocator::unbounded(layout))
                .map_err(|_| root_error(RootAdmissionError::AllocationFailed))?;
            for (old, count) in self.map.iter() {
                replacement.insert(*old, *count);
            }
            retired = Some(std::mem::replace(&mut self.map, replacement));
        }
        increment_map(&mut self.map, handle).map_err(root_error)?;
        self.advance_revision();
        Ok(retired.map(|map| RetiredRootRegistry { _map: map }))
    }
    pub(crate) fn remove(&mut self, handle: GcHandle) {
        match self.map.get_mut(&handle) {
            Some(count) if *count > 1 => *count -= 1,
            Some(_) => {
                self.map.remove(&handle);
            }
            None => {}
        }
        self.advance_revision();
    }
    fn advance_revision(&mut self) {
        match self.revision.checked_add(1) {
            Some(next) => self.revision = next,
            None => self.exhausted = true,
        }
    }
}

impl crate::GarbageCollector {
    pub(super) fn remove_root_batch(&self, handles: &[GcHandle]) {
        {
            let _admission = ROOT_ADMISSION.write();
            let mut registry = self.root_ptrs.lock();
            for handle in handles {
                registry.remove(*handle);
            }
        }
        for handle in handles {
            crate::root_retirement::notify(crate::RootRetirement::Explicit(*handle));
        }
    }
}

pub(super) fn increment_map(
    map: &mut RootMap<GcHandle, usize>,
    handle: GcHandle,
) -> Result<(), RootAdmissionError> {
    let count = map.entry(handle).or_insert(0);
    *count = count
        .checked_add(1)
        .ok_or(RootAdmissionError::RegistrationCountOverflow)?;
    Ok(())
}
fn root_error(error: RootAdmissionError) -> GcError {
    GcError::RootRegistrationFailed(error.to_string())
}
