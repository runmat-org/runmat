//! Read-only Tet10 node incidence in contiguous buffers, shared by all candidates.
use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
pub(super) struct Incidence<'a> {
    offsets: Vec<usize>,
    elements: Vec<usize>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> Incidence<'a> {
    pub fn prepare(
        topology: &SolverMeshTopology,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let nodes = topology.nodes.len();
        let count = topology
            .volume_elements
            .iter()
            .try_fold(0usize, |count, element| {
                count
                    .checked_add(element.node_ids.len())
                    .ok_or_else(|| invalid("node incidence count overflow"))
            })?;
        let slots = nodes
            .checked_add(1)
            .and_then(|slots| slots.checked_add(count))
            .ok_or_else(|| invalid("node incidence storage overflow"))?;
        let bytes = slots
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or_else(|| invalid("node incidence storage overflow"))?;
        let storage =
            GeometryAllocationReservation::new(control, bytes as u64).map_err(error::geometry)?;
        let mut offsets = vec![0usize; nodes + 1];
        for element in &topology.volume_elements {
            control.checkpoint().map_err(error::geometry)?;
            for &id in element.node_ids.iter() {
                let slot = geometry::node_slot(&topology.nodes, id)?;
                offsets[slot + 1] = offsets[slot + 1]
                    .checked_add(1)
                    .ok_or_else(|| invalid("node incidence degree overflow"))?;
            }
        }
        for index in 1..offsets.len() {
            control.checkpoint().map_err(error::geometry)?;
            offsets[index] += offsets[index - 1];
        }
        let cursor_storage = GeometryAllocationReservation::new(
            control,
            (nodes * std::mem::size_of::<usize>()) as u64,
        )
        .map_err(error::geometry)?;
        let mut cursor = offsets[..nodes].to_vec();
        let mut elements = vec![0; count];
        for (index, element) in topology.volume_elements.iter().enumerate() {
            control.checkpoint().map_err(error::geometry)?;
            for &id in element.node_ids.iter() {
                let slot = geometry::node_slot(&topology.nodes, id)?;
                elements[cursor[slot]] = index;
                cursor[slot] += 1;
            }
        }
        drop(cursor);
        drop(cursor_storage);
        Ok(Self {
            offsets,
            elements,
            _storage: storage,
        })
    }
    pub fn elements(&self, id: u64) -> Result<&[usize], DelaunaySolverTopologyError> {
        let slot = usize::try_from(
            id.checked_sub(1)
                .ok_or_else(|| invalid("node incidence node is absent"))?,
        )
        .map_err(|_| invalid("node incidence node is absent"))?;
        let end = *self
            .offsets
            .get(slot + 1)
            .ok_or_else(|| invalid("node incidence node is absent"))?;
        Ok(&self.elements[self.offsets[slot]..end])
    }
}

#[cfg(test)]
mod tests;
