//! Actual requested compact source capacities, excluding allocator bookkeeping/RSS.
#[cfg(test)]
mod tests;
use crate::cdt::{DelaunayVolumeProvenance, DelaunayVolumeTopology};
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    PersistentEntityId,
};

pub(super) fn retained_bytes(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    control: &dyn GeometryEvaluationControl,
) -> Result<u64, GeometryEvaluationError> {
    let mut inventory = Inventory {
        seen: Default::default(),
        bytes: 0,
        control,
    };
    let DelaunayVolumeTopology {
        nodes,
        tetrahedra,
        incidence,
    } = topology;
    inventory.inline_vector(nodes)?;
    inventory.vector(tetrahedra)?;
    for cell in tetrahedra {
        let crate::cdt::DelaunayVolumeTetrahedron {
            vertex_indices: _,
            neighbors: _,
            region_id,
        } = cell;
        if let Some(region) = region_id {
            inventory.entity(region)?;
        }
    }
    let crate::cdt::DelaunayVolumeIncidence {
        vertex_stars,
        boundary_facets,
        regions,
        unassigned_tetrahedron_indices,
    } = incidence;
    inventory.vector(vertex_stars)?;
    for star in vertex_stars {
        inventory.inline_vector(star)?;
    }
    inventory.vector(boundary_facets)?;
    inventory.vector(regions)?;
    for region in regions {
        let crate::cdt::DelaunayRegionIncidence {
            region_id,
            tetrahedron_indices,
        } = region;
        inventory.entity(region_id)?;
        inventory.inline_vector(tetrahedron_indices)?;
    }
    inventory.inline_vector(unassigned_tetrahedron_indices)?;
    let DelaunayVolumeProvenance {
        nodes,
        segments,
        facets,
    } = provenance;
    inventory.vector(nodes)?;
    for node in nodes {
        let crate::cdt::DelaunayNodeProvenance {
            node_identity: _,
            entity_ids,
        } = node;
        inventory.entities(entity_ids)?;
    }
    inventory.vector(segments)?;
    for segment in segments {
        let crate::cdt::DelaunaySegmentProvenance {
            node_identities: _,
            entity_ids,
            edge_parameters: _,
        } = segment;
        inventory.entities(entity_ids)?;
    }
    inventory.vector(facets)?;
    for facet in facets {
        let crate::cdt::DelaunayFacetProvenance {
            node_identities: _,
            chart_id: _,
            entity_ids,
            region_ids,
        } = facet;
        inventory.entities(entity_ids)?;
        inventory.entities(region_ids)?;
    }
    Ok(inventory.bytes)
}
struct Inventory<'a> {
    seen: std::collections::HashSet<usize>,
    bytes: u64,
    control: &'a dyn GeometryEvaluationControl,
}
impl Inventory<'_> {
    fn add(&mut self, bytes: u64) -> Result<(), GeometryEvaluationError> {
        self.control.checkpoint()?;
        self.bytes = self.bytes.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }
    fn vector<T>(&mut self, values: &Vec<T>) -> Result<(), GeometryEvaluationError> {
        self.add(
            (values.capacity() as u64)
                .checked_mul(std::mem::size_of::<T>() as u64)
                .ok_or_else(overflow)?,
        )
    }
    fn inline_vector<T: Copy>(&mut self, values: &Vec<T>) -> Result<(), GeometryEvaluationError> {
        self.vector(values)
    }
    fn entities(
        &mut self,
        values: &Vec<PersistentEntityId>,
    ) -> Result<(), GeometryEvaluationError> {
        self.vector(values)?;
        for entity in values {
            self.entity(entity)?;
        }
        Ok(())
    }
    /// Records hold handles (in their slots); each distinct shared identity
    /// allocation is charged once per inventory.
    fn entity(&mut self, entity: &PersistentEntityId) -> Result<(), GeometryEvaluationError> {
        if self.seen.insert(entity.shared_address()) {
            self.add(entity.shared_heap_bytes())?;
        }
        Ok(())
    }
}
fn overflow() -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::AllocationBudgetExceeded,
        "compact source storage inventory overflow",
    )
}
