//! Allocation estimates owned alongside the artifact validators, not by numerical consumers.
use std::collections::BTreeSet;
use std::mem::size_of;

use runmat_canonical_codec::{serialized_materialization_resident_bytes_estimate, CanonicalLimits};

use super::{MeshingContractError, SolverMeshArtifact, StableDigest};

impl SolverMeshArtifact {
    /// Conservative additional requested-heap estimate for `validate_canonical`, excluding the
    /// borrowed artifact, allocator bookkeeping and native/RSS usage. Does not validate or encode.
    /// Payload validation releases its indexes before canonical identity encoding. The validator's
    /// index phases are summed conservatively; the encoding phase uses the shared codec profiler.
    /// Profiling can reject malformed serialization (e.g. nonfinite coordinates) before admission.
    pub fn validation_resident_bytes_estimate(&self) -> Result<u64, MeshingContractError> {
        // The identity projection only zeroes two fixed-width digests. Its shape/serialized widths
        // equal this artifact, so no second identity schema is needed for resource accounting.
        let encoding = serialized_materialization_resident_bytes_estimate(
            self,
            CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
        )
        .map_err(|error| {
            MeshingContractError::invalid("mesh validation resources", error.to_string())
        })?;
        let t = &self.topology;
        let mut scratch = 64 * 1024_u64;
        // Ordered ID vectors/sets, node identity vector/set/map, element identity set.
        for count in [
            t.nodes.len(),
            t.volume_elements.len(),
            t.boundary_faces.len(),
            t.boundary_edges.len(),
        ] {
            add(&mut scratch, index::<u64>(count)?)?;
            add(&mut scratch, slots::<u64>(count, 2)?)?;
        }
        add(&mut scratch, index::<(u64, StableDigest)>(t.nodes.len())?)?;
        add(&mut scratch, index::<StableDigest>(t.nodes.len())?)?;
        add(&mut scratch, slots::<StableDigest>(t.nodes.len(), 2)?)?;
        add(
            &mut scratch,
            index::<StableDigest>(t.volume_elements.len())?,
        )?;
        // Boundary validation retains a per-element connectivity set and face slice index.
        add(
            &mut scratch,
            index::<(u64, BTreeSet<u64>)>(t.volume_elements.len())?,
        )?;
        for element in &t.volume_elements {
            add(&mut scratch, index::<u64>(element.node_ids.len())?)?;
            add(&mut scratch, index::<&u64>(element.node_ids.len())?)?;
        }
        add(
            &mut scratch,
            index::<(u64, &[u64])>(t.boundary_faces.len())?,
        )?;
        add(&mut scratch, index::<StableDigest>(t.boundary_faces.len())?)?;
        add(&mut scratch, index::<StableDigest>(t.boundary_edges.len())?)?;
        for count in t
            .boundary_faces
            .iter()
            .map(|face| face.node_ids.len())
            .chain(t.boundary_edges.iter().map(|edge| edge.node_ids.len()))
        {
            add(&mut scratch, index::<&u64>(count)?)?;
        }
        // Tet4 referenced nodes; Tet10 corner/midpoint/union sets, identity/node maps, six
        // volume edges per tetrahedron, and boundary-face corner lookup. Count both orders.
        for _ in 0..4 {
            add(&mut scratch, index::<u64>(t.nodes.len())?)?;
        }
        add(&mut scratch, index::<(u64, StableDigest)>(t.nodes.len())?)?;
        add(
            &mut scratch,
            index::<(u64, &super::SolverMeshNode)>(t.nodes.len())?,
        )?;
        let edges = t
            .volume_elements
            .len()
            .checked_mul(6)
            .ok_or_else(overflow)?;
        add(&mut scratch, index::<([u64; 2], u64)>(edges)?)?;
        add(
            &mut scratch,
            index::<(u64, &[u64])>(t.volume_elements.len())?,
        )?;
        for count in t
            .boundary_faces
            .iter()
            .map(|face| face.provenance.len())
            .chain(t.boundary_edges.iter().map(|edge| edge.provenance.len()))
        {
            add(&mut scratch, index::<&super::PersistentEntityId>(count)?)?;
        }
        // Region/contact/interface classification, neighbors and field topology comparisons.
        add(
            &mut scratch,
            index::<&super::PersistentEntityId>(t.regions.len())?,
        )?;
        add(&mut scratch, index::<u64>(t.volume_elements.len())?)?;
        add(&mut scratch, index::<u64>(t.boundary_faces.len())?)?;
        add(
            &mut scratch,
            index::<(u64, &super::SolverBoundaryFace)>(t.boundary_faces.len())?,
        )?;
        add(&mut scratch, index::<(u64, u64)>(t.neighbors.len())?)?;
        for field in &t.field_topologies {
            add(&mut scratch, index::<u64>(field.ordered_entity_ids.len())?)?;
        }
        // Fixed four-domain lookup/location sets and bounded diagnostics fit the fixed allowance.
        Ok(scratch.max(encoding))
    }
}

fn index<T>(count: usize) -> Result<u64, MeshingContractError> {
    let width = size_of::<T>() + 4 * size_of::<usize>();
    u64::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(3 * width as u64))
        .ok_or_else(overflow)
}
fn slots<T>(count: usize, growth: u64) -> Result<u64, MeshingContractError> {
    u64::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(size_of::<T>() as u64))
        .and_then(|n| n.checked_mul(growth))
        .ok_or_else(overflow)
}
fn add(total: &mut u64, bytes: u64) -> Result<(), MeshingContractError> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}
fn overflow() -> MeshingContractError {
    MeshingContractError::invalid(
        "mesh validation resources",
        "resident byte estimate overflow",
    )
}

#[cfg(test)]
mod tests;
