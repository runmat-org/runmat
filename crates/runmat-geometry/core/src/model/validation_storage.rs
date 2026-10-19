//! Typed validation scratch: admitted before allocation, released with its buffer.
use crate::{
    GeometryAllocationReservation, GeometryContractError, GeometryEvaluationControl,
    PersistentEntityId,
};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy)]
pub(crate) struct ValidationWork<'a>(pub Option<&'a dyn GeometryEvaluationControl>);
impl<'a> ValidationWork<'a> {
    pub fn checkpoint(self) -> Result<(), GeometryContractError> {
        if let Some(control) = self.0 {
            control.checkpoint().map_err(control_error)?;
        }
        Ok(())
    }
    pub fn buffer<T>(
        self,
        capacity: usize,
    ) -> Result<ValidationBuffer<'a, T>, GeometryContractError> {
        self.checkpoint()?;
        let bytes = capacity
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|v| u64::try_from(v).ok())
            .filter(|v| *v <= isize::MAX as u64)
            .ok_or_else(|| {
                GeometryContractError::invalid("validation scratch", "typed capacity overflow")
            })?;
        let reservation = self
            .0
            .map(|control| GeometryAllocationReservation::new(control, bytes))
            .transpose()
            .map_err(control_error)?;
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(|_| {
            GeometryContractError::invalid("validation scratch", "allocation failed")
        })?;
        Ok(ValidationBuffer {
            values,
            _reservation: reservation,
        })
    }
}
pub(crate) struct ValidationBuffer<'a, T> {
    // Fields drop in this order: owned values before their reservation.
    values: Vec<T>,
    _reservation: Option<GeometryAllocationReservation<'a>>,
}
impl<T> ValidationBuffer<'_, T> {
    pub fn push(&mut self, value: T) {
        assert!(
            self.values.len() < self.values.capacity(),
            "validation exceeded its typed inventory"
        );
        self.values.push(value);
    }
    pub fn pop(&mut self) -> Option<T> {
        self.values.pop()
    }
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.values.dedup();
    }
}
impl<T> Deref for ValidationBuffer<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T> DerefMut for ValidationBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

pub(crate) struct IdIndex<'a>(ValidationBuffer<'a, &'a PersistentEntityId>);
impl<'a> IdIndex<'a> {
    pub fn collect(
        ids: impl ExactSizeIterator<Item = &'a PersistentEntityId>,
        work: ValidationWork<'a>,
    ) -> Result<Self, GeometryContractError> {
        let mut values = work.buffer(ids.len())?;
        for id in ids {
            work.checkpoint()?;
            values.push(id);
        }
        Ok(Self(values))
    }
    pub fn index(&self, id: &PersistentEntityId) -> Option<usize> {
        self.0.binary_search_by(|entry| (*entry).cmp(id)).ok()
    }
    pub fn contains(&self, id: &PersistentEntityId) -> bool {
        self.index(id).is_some()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &'a PersistentEntityId> + '_ {
        self.0.iter().copied()
    }
}
pub(crate) struct Claims<'index, 'data> {
    index: &'index IdIndex<'data>,
    flags: ValidationBuffer<'data, u8>,
    count: usize,
}
impl<'index, 'data> Claims<'index, 'data> {
    pub fn new(
        index: &'index IdIndex<'data>,
        work: ValidationWork<'data>,
    ) -> Result<Self, GeometryContractError> {
        let mut flags = work.buffer(index.len())?;
        for _ in 0..index.len() {
            flags.push(0);
        }
        Ok(Self {
            index,
            flags,
            count: 0,
        })
    }
    pub fn insert(&mut self, id: &PersistentEntityId) -> bool {
        let Some(slot) = self.index.index(id) else {
            return false;
        };
        if std::mem::replace(&mut self.flags[slot], 1) != 0 {
            return false;
        }
        self.count += 1;
        true
    }
    pub fn complete(&self) -> bool {
        self.count == self.index.len()
    }
    pub fn count(&self) -> usize {
        self.count
    }
}
pub(crate) fn lookup<'a, T, K: Ord>(
    records: &'a [T],
    target: &K,
    key: impl Fn(&T) -> &K,
) -> Option<&'a T> {
    records
        .binary_search_by(|record| key(record).cmp(target))
        .ok()
        .map(|slot| &records[slot])
}
fn control_error(error: crate::GeometryEvaluationError) -> GeometryContractError {
    GeometryContractError::invalid("geometry validation control", error.to_string())
}

#[cfg(test)]
pub(crate) mod test_control;
#[cfg(test)]
mod tests;
