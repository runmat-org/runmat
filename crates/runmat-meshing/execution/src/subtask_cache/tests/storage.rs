//! Exercise the real accepted-face encoder repeatedly without rerunning geometry.
use super::*;
pub(super) struct RepeatedRetain<'a, 'b, 'c> {
    pub reuse: &'a mut SurfaceFaceReuse<'b>,
    pub control: &'a MeshingGeometryEvaluationControl<'c>,
}
impl ExactFaceMeshReuse for RepeatedRetain<'_, '_, '_> {
    fn restore(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
    ) -> Result<Option<ExactFaceMesh>, ExactFacePartitionError> {
        let before = self.control.usage().live_allocation_bytes;
        let restored = self.reuse.restore(boundary, coordinates)?;
        if let Some(mesh) = &restored {
            assert_eq!(
                self.control.usage().live_allocation_bytes - before,
                mesh.retained_storage_bytes(self.control).unwrap(),
                "cold restoration retains typed storage, not decoder workspace"
            );
        }
        Ok(restored)
    }
    fn retain(
        &mut self,
        boundary: &ExactFaceBoundary,
        coordinates: &ExactBoundaryNodeCoordinates<'_>,
        accepted: AcceptedExactFace<'_>,
    ) -> Result<(), ExactFacePartitionError> {
        let before = self.control.usage().live_allocation_bytes;
        for _ in 0..8 {
            // Retain takes its producer capability by value; borrow it through
            // a helper below so the same original accepted mesh is never copied.
            self.reuse
                .retain_borrowed(boundary, coordinates, &accepted)?;
            assert_eq!(
                self.control.usage().live_allocation_bytes,
                before,
                "completed face encoding must release all temporary storage"
            );
        }
        Ok(())
    }
}
