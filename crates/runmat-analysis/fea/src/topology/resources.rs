//! Count-only admission estimate beside the authoritative projection constructor.
use runmat_canonical_codec::{profile_serialized_shape, CanonicalLimits};
use runmat_meshing_core::{BoundaryTriangleOrder, PersistentEntityId, SolverMeshArtifact};

use super::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy, BoundaryQuadraturePoint,
    ProjectedBoundaryFace, ProjectedElement, ProjectedNode,
};

mod retained;

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

impl AnalysisProjection {
    /// Shared curved-boundary moment construction scratch, excluding source and allocator overhead.
    pub fn boundary_quadrature_scratch_resident_bytes(
        order: BoundaryTriangleOrder,
    ) -> Result<u64, AnalysisProjectionError> {
        let mut total = 0;
        if order == BoundaryTriangleOrder::Tri6 {
            for size in [
                super::quadrature::GAUSS_LEGENDRE_EIGHT_UNIT.len(),
                super::quadrature::GAUSS_LEGENDRE_TWELVE_UNIT.len(),
                super::quadrature::GAUSS_LEGENDRE_SIXTEEN_UNIT.len(),
            ] {
                add(
                    &mut total,
                    slots::<BoundaryQuadraturePoint>(
                        size.checked_mul(size).ok_or_else(overflow)?,
                        1,
                    )?,
                )?;
            }
        }
        Ok(total)
    }
    /// Analytic requested-heap bound for an owned clone, including the actual
    /// private lookup inventories and nested provenance/connectivity payloads.
    /// This is not allocator bookkeeping or measured RSS.
    pub fn clone_resident_bytes_estimate(&self) -> Result<u64, AnalysisProjectionError> {
        let mut total = 0;
        add(&mut total, vector::<ProjectedNode>(self.nodes.len())?)?;
        add(&mut total, vector::<ProjectedElement>(self.elements.len())?)?;
        add(
            &mut total,
            vector::<ProjectedBoundaryFace>(self.boundary_faces.len())?,
        )?;
        add(&mut total, vector::<Vec<usize>>(self.node_neighbors.len())?)?;
        add(&mut total, slots::<(u64, usize)>(self.node_by_id.len(), 1)?)?;
        add(
            &mut total,
            slots::<(u64, usize)>(self.element_by_id.len(), 1)?,
        )?;
        for neighbors in &self.node_neighbors {
            add(&mut total, vector::<usize>(neighbors.len())?)?;
        }
        for element in &self.elements {
            add(&mut total, vector::<usize>(element.node_indices.len())?)?;
            add(
                &mut total,
                vector::<PersistentEntityId>(element.provenance.len())?,
            )?;
        }
        for face in &self.boundary_faces {
            add(&mut total, vector::<usize>(face.node_indices.len())?)?;
            add(
                &mut total,
                vector::<usize>(face.adjacent_element_indices.len())?,
            )?;
            add(
                &mut total,
                vector::<PersistentEntityId>(face.provenance.len())?,
            )?;
        }
        Ok(total)
    }

    /// Additional requested-heap estimate for `derive`, excluding the borrowed mesh, allocator
    /// bookkeeping and RSS. Counts malformed raw connectivity without trusting canonical limits.
    /// No validation, geometric projection, or canonical encoding is performed by this method.
    /// Sequential construction phases are summed conservatively, including mesh validation.
    pub fn derivation_resident_bytes_estimate(
        mesh: &SolverMeshArtifact,
    ) -> Result<u64, AnalysisProjectionError> {
        let topology = &mesh.topology;
        let mut total = mesh
            .validation_resident_bytes_estimate()
            .map_err(resource)?;
        add(&mut total, Self::retained_resident_bytes_upper_bound(mesh)?)?;
        let neighbor_references = super::neighbors::occurrence_count(
            topology
                .volume_elements
                .iter()
                .map(|element| element.node_ids.len()),
        )?;
        // Sorted edge candidates coexist with retained adjacency; duplicate occurrences remain
        // in this exact-reserved scratch capacity until construction finishes.
        add(&mut total, slots::<(usize, usize)>(neighbor_references, 1)?)?;
        // Element/face coordinate arrays are bounded stack storage. Curved Tri6 moments
        // retain and refine the original shared quadrature rules below.
        if topology
            .boundary_faces
            .iter()
            .any(|face| face.order == BoundaryTriangleOrder::Tri6)
        {
            add(
                &mut total,
                Self::boundary_quadrature_scratch_resident_bytes(BoundaryTriangleOrder::Tri6)?,
            )?;
        }
        // Projection identity is a small mesh-digest/policy contract; profile the same borrowed
        // shape rather than encoding it merely to count its temporary tree and byte buffer.
        add(
            &mut total,
            profile_serialized_shape(
                &(mesh.canonical_digest, AnalysisProjectionPolicy::default()),
                PROFILE_LIMITS,
            )
            .map_err(resource)?
            .materialization_resident_bytes_estimate,
        )?;
        Ok(total)
    }
}

// Records contain the scalar kind and string/vector headers already. Count only the
// borrowed original UTF-8 and path buffers copied by their standard Clone implementations.
/// Owned identity text, for retained-storage bounds. Clones share interned
/// identities, so clone estimates charge only the handle slots.
fn entity_payload_bytes(entity: &PersistentEntityId) -> Result<u64, AnalysisProjectionError> {
    let mut bytes = vector::<String>(entity.assembly_path.len())?;
    add(&mut bytes, vector::<u8>(entity.source_topology_id.len())?)?;
    for path in &entity.assembly_path {
        add(&mut bytes, vector::<u8>(path.len())?)?;
    }
    Ok(bytes)
}

fn vector<T>(count: usize) -> Result<u64, AnalysisProjectionError> {
    slots::<T>(count, 1)
}
fn slots<T>(count: usize, copies: u64) -> Result<u64, AnalysisProjectionError> {
    let bytes = super::storage::slots::<T>(count)?;
    scale(bytes, copies)
}
fn scale(count: u64, width: u64) -> Result<u64, AnalysisProjectionError> {
    count.checked_mul(width).ok_or_else(overflow)
}
fn add(total: &mut u64, bytes: u64) -> Result<(), AnalysisProjectionError> {
    *total = total.checked_add(bytes).ok_or_else(overflow)?;
    Ok(())
}
fn overflow() -> AnalysisProjectionError {
    resource("resident byte estimate overflow")
}
fn resource(error: impl std::fmt::Display) -> AnalysisProjectionError {
    AnalysisProjectionError::ResourceEstimate(error.to_string())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tests::allocation::measured;
