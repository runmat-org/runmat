//! Requested contiguous geometry buffers; shared provenance is charged once by its field owner.
use super::{ExactFaceGeometryVertex, ExactFaceTriangleGeometry};
pub(crate) fn buffer_bytes(vertices: usize, triangles: usize) -> Option<u64> {
    let vertices = vertices.checked_mul(std::mem::size_of::<ExactFaceGeometryVertex>())?;
    let triangles = triangles.checked_mul(std::mem::size_of::<ExactFaceTriangleGeometry>())?;
    u64::try_from(vertices.checked_add(triangles)?).ok()
}
