//! Exact-capacity scratch which carries its charge through returned inventories.
use super::*;
use runmat_geometry_core::GeometryAllocationReservation;
use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
};
pub(in crate::cdt::solver_topology) struct ReservedBuffer<'a, T> {
    values: Vec<T>,
    _storage: Option<GeometryAllocationReservation<'a>>,
}
impl<'a, T> ReservedBuffer<'a, T> {
    pub fn new(count: usize, storage: Storage<'a>) -> Result<Self, DelaunaySolverTopologyError> {
        let reservation = storage
            .0
            .map(|control| {
                GeometryAllocationReservation::new(control, bytes::<T>(count)?)
                    .map_err(error::geometry)
            })
            .transpose()?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| overflow())?;
        Ok(Self {
            values,
            _storage: reservation,
        })
    }
    pub fn push(&mut self, value: T) -> Result<(), DelaunaySolverTopologyError> {
        if self.values.len() == self.values.capacity() {
            return Err(overflow());
        }
        self.values.push(value);
        Ok(())
    }
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
}
impl<T> Deref for ReservedBuffer<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T> DerefMut for ReservedBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T: PartialEq> PartialEq for ReservedBuffer<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}
impl<T: Eq> Eq for ReservedBuffer<'_, T> {}
impl<T: Ord> PartialOrd for ReservedBuffer<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Ord> Ord for ReservedBuffer<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.values.cmp(&other.values)
    }
}
