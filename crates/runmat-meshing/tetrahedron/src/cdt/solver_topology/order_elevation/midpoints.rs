//! Immutable sorted edge lookup with explicit storage admission and scoped release.
use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
type Entry = ([u64; 2], u64);
pub(super) struct MidpointMap<'a> {
    entries: Vec<Entry>,
    maximum_entries: usize,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> MidpointMap<'a> {
    pub fn new(
        count: usize,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let bytes = count
            .checked_mul(std::mem::size_of::<Entry>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                error::failure(
                    DelaunaySolverTopologyErrorKind::ResourceLimit,
                    "midpoint lookup storage overflow",
                )
            })?;
        let storage =
            GeometryAllocationReservation::new(control, bytes).map_err(error::geometry)?;
        Ok(Self {
            entries: Vec::with_capacity(count),
            maximum_entries: count,
            _storage: storage,
        })
    }
    pub fn push(&mut self, edge: [u64; 2], node: u64) -> Result<(), DelaunaySolverTopologyError> {
        if self.entries.len() == self.maximum_entries
            || self.entries.last().is_some_and(|last| last.0 >= edge)
        {
            return Err(error::failure(
                DelaunaySolverTopologyErrorKind::InvalidMesh,
                "midpoint preparation exceeds its inventory or repeats an unordered edge",
            ));
        }
        self.entries.push((edge, node));
        Ok(())
    }
    pub fn get(&self, edge: &[u64; 2]) -> Option<&u64> {
        self.entries
            .binary_search_by_key(edge, |entry| entry.0)
            .ok()
            .map(|index| &self.entries[index].1)
    }
}
fn borrowed(entry: &Entry) -> (&[u64; 2], &u64) {
    (&entry.0, &entry.1)
}
impl<'a, 'b> IntoIterator for &'b MidpointMap<'a> {
    type Item = (&'b [u64; 2], &'b u64);
    type IntoIter = std::iter::Map<std::slice::Iter<'b, Entry>, fn(&'b Entry) -> Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(borrowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flat_lookup_matches_ordered_reference_and_refuses_unplanned_growth() {
        let reference =
            std::collections::BTreeMap::from([([1, 2], 11), ([1, 3], 12), ([2, 3], 13)]);
        let mut map = MidpointMap::new(
            reference.len(),
            &crate::cdt::solver_topology::test_evaluator::CONTROL,
        )
        .unwrap();
        for (&edge, &node) in &reference {
            map.push(edge, node).unwrap();
        }
        assert_eq!(
            (&map).into_iter().collect::<Vec<_>>(),
            reference.iter().collect::<Vec<_>>()
        );
        for edge in [[1, 2], [1, 3], [2, 3], [3, 4]] {
            assert_eq!(map.get(&edge), reference.get(&edge));
        }
        assert!(map.push([3, 4], 14).is_err());
    }
}
