//! Requested typed retention, distinct from transient canonical decoder workspace.
use super::types::*;
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    PersistentEntityId,
};

struct Inventory<'a> {
    seen: std::collections::HashSet<usize>,
    bytes: u64,
    control: &'a dyn GeometryEvaluationControl,
}
impl Inventory<'_> {
    fn add(&mut self, bytes: usize) -> Result<(), GeometryEvaluationError> {
        self.bytes = self.bytes.checked_add(bytes as u64).ok_or_else(overflow)?;
        Ok(())
    }
    fn vector<T>(&mut self, capacity: usize) -> Result<(), GeometryEvaluationError> {
        self.add(
            capacity
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(overflow)?,
        )
    }
    /// Records hold handles (in their slots); each distinct shared identity
    /// allocation is charged once per inventory.
    fn entity(&mut self, entity: &PersistentEntityId) -> Result<(), GeometryEvaluationError> {
        self.control.checkpoint()?;
        if self.seen.insert(entity.shared_address()) {
            self.add(entity.shared_heap_bytes() as usize)?;
        }
        Ok(())
    }
}
fn overflow() -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::AllocationBudgetExceeded,
        "typed face storage inventory overflow",
    )
}

impl ExactFaceMesh {
    /// Root, vector capacities and nested owned strings. This is requested typed
    /// storage, not allocator overhead or process RSS; it grants no mesh authority.
    pub fn retained_storage_bytes(
        &self,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<u64, GeometryEvaluationError> {
        let mut inventory = Inventory {
            seen: Default::default(),
            bytes: std::mem::size_of::<Self>() as u64,
            control,
        };
        inventory.entity(&self.source_face_id)?;
        inventory.vector::<ExactFaceMeshNode>(self.nodes.capacity())?;
        inventory.vector::<ExactFaceMeshTriangle>(self.triangles.capacity())?;
        inventory.vector::<ExactFaceMeshBoundarySegment>(self.boundary_segments.capacity())?;
        inventory.vector::<ExactFaceMeshJoinedCut>(self.joined_chart_cuts.capacity())?;
        inventory.rows(&self.nodes, &self.triangles, &self.boundary_segments)?;
        Ok(inventory.bytes)
    }
}

impl Inventory<'_> {
    fn rows(
        &mut self,
        nodes: &[ExactFaceMeshNode],
        triangles: &[ExactFaceMeshTriangle],
        boundary_segments: &[ExactFaceMeshBoundarySegment],
    ) -> Result<(), GeometryEvaluationError> {
        for node in nodes {
            self.control.checkpoint()?;
            self.vector::<ExactFaceMeshNodeUse>(node.uses.capacity())?;
            for usage in &node.uses {
                self.entity(&usage.source_face_id)?;
                self.vector::<ExactFaceMeshEdgeParameter>(usage.exact_edge_parameters.capacity())?;
                for parameter in &usage.exact_edge_parameters {
                    self.entity(&parameter.source_coedge_id)?;
                    self.entity(&parameter.source_edge_id)?;
                }
            }
        }
        for triangle in triangles {
            self.entity(&triangle.source_face_id)?;
        }
        for segment in boundary_segments {
            self.entity(&segment.source_coedge_id)?;
            self.entity(&segment.source_edge_id)?;
        }
        Ok(())
    }
}
impl crate::ExactSurfaceMesh {
    pub fn retained_storage_bytes(
        &self,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<u64, GeometryEvaluationError> {
        let mut inventory = Inventory {
            seen: Default::default(),
            bytes: std::mem::size_of::<Self>() as u64,
            control,
        };
        inventory.vector::<PersistentEntityId>(self.face_ids.capacity())?;
        inventory.vector::<ExactFaceMeshNode>(self.nodes.capacity())?;
        inventory.vector::<ExactFaceMeshTriangle>(self.triangles.capacity())?;
        inventory.vector::<ExactFaceMeshBoundarySegment>(self.boundary_segments.capacity())?;
        inventory.vector::<crate::ExactSurfaceShellEvidence>(self.shells.capacity())?;
        for face in &self.face_ids {
            inventory.entity(face)?;
        }
        for shell in &self.shells {
            inventory.entity(&shell.source_shell_id)?;
        }
        inventory.rows(&self.nodes, &self.triangles, &self.boundary_segments)?;
        Ok(inventory.bytes)
    }
}
