//! Local edits preserve original nodes by move. Temporary candidate storage dies
//! after each attempt; only an accepted node's actual retained buffers transfer.
use super::*;
use crate::cdt::solver_topology::storage::Storage;
use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};

pub(super) fn edit_nodes(
    topology: &mut SolverMeshTopology,
    slots: &[usize],
    parent: &dyn GeometryEvaluationControl,
    operation: impl FnOnce(
        &mut SolverMeshTopology,
        &dyn GeometryEvaluationControl,
    ) -> Result<bool, DelaunaySolverTopologyError>,
) -> Result<u64, DelaunaySolverTopologyError> {
    let scope = GeometryAllocationScope::new(parent);
    let storage = Storage(Some(&scope));
    let mut originals = storage.vector(slots.len())?;
    let mut replacements = storage.vector(slots.len())?;
    let mut original_bytes = 0u64;
    for (ordinal, &slot) in slots.iter().enumerate() {
        if slots[..ordinal].contains(&slot) {
            return Err(invalid("local repair references the same node twice"));
        }
        let node = topology
            .nodes
            .get(slot)
            .ok_or_else(|| invalid("local repair node is absent"))?;
        original_bytes = original_bytes
            .checked_add(node.owned_buffer_bytes().map_err(error::solver)?)
            .ok_or_else(|| resource("local repair storage overflow"))?;
        replacements.push(storage.node(node)?);
    }
    for (&slot, replacement) in slots.iter().zip(replacements) {
        originals.push((
            slot,
            std::mem::replace(&mut topology.nodes[slot], replacement),
        ));
    }
    let result = operation(topology, &scope).and_then(|accepted| {
        if !accepted {
            return Ok(None);
        }
        let mut retained = 0u64;
        let mut changed = 0;
        for (slot, original) in &originals {
            let node = &topology.nodes[*slot];
            changed += u64::from(node != original);
            retained = retained
                .checked_add(node.owned_buffer_bytes().map_err(error::solver)?)
                .ok_or_else(|| resource("local repair retained storage overflow"))?;
        }
        if changed == 0 {
            return Ok(None);
        }
        let ownership = scope
            .transfer_reservation(retained)
            .map_err(error::geometry)?;
        Ok(Some((changed, retained, ownership)))
    });
    match result {
        Ok(Some((changed, retained, mut ownership))) => {
            ownership.retain_bytes(retained);
            drop(ownership);
            drop(originals);
            parent.release_allocation_bytes(original_bytes);
            Ok(changed)
        }
        result => {
            for (slot, original) in originals {
                let discarded = std::mem::replace(&mut topology.nodes[slot], original);
                drop(discarded);
            }
            result.map(|_| 0)
        }
    }
}

#[cfg(test)]
mod tests;
