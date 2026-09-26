//! Count-only admission estimate beside the authoritative projection constructor.
use std::{collections::BTreeSet, mem::size_of};

use runmat_canonical_codec::{serialized_materialization_resident_bytes_estimate, CanonicalLimits};
use runmat_meshing_core::{BoundaryTriangleOrder, SolverMeshArtifact};
use serde::{Serialize, Serializer};

use super::{
    AnalysisProjection, AnalysisProjectionError, AnalysisProjectionPolicy, BoundaryQuadraturePoint,
    ProjectedBoundaryFace, ProjectedElement, ProjectedNode,
};

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

impl AnalysisProjection {
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
        // Retained projected records and lookup indexes.
        add(&mut total, vector::<ProjectedNode>(topology.nodes.len())?)?;
        add(
            &mut total,
            vector::<ProjectedElement>(topology.volume_elements.len())?,
        )?;
        add(
            &mut total,
            vector::<ProjectedBoundaryFace>(topology.boundary_faces.len())?,
        )?;
        add(&mut total, index::<(u64, usize)>(topology.nodes.len())?)?;
        add(
            &mut total,
            index::<(u64, usize)>(topology.volume_elements.len())?,
        )?;

        let mut neighbor_references = 0_u64;
        let mut largest_element = 0_usize;
        for element in &topology.volume_elements {
            let count = element.node_ids.len();
            add(&mut total, vector::<usize>(count)?)?;
            largest_element = largest_element.max(count);
            let count = u64::try_from(count).map_err(resource)?;
            add(
                &mut neighbor_references,
                count
                    .checked_mul(count.saturating_sub(1))
                    .ok_or_else(overflow)?,
            )?;
        }
        let mut largest_face = 0_usize;
        for face in &topology.boundary_faces {
            add(&mut total, vector::<usize>(face.node_ids.len())?)?;
            add(
                &mut total,
                vector::<usize>(face.adjacent_volume_element_ids.len())?,
            )?;
            add(
                &mut total,
                vector::<runmat_meshing_core::PersistentEntityId>(face.provenance.len())?,
            )?;
            largest_face = largest_face.max(face.node_ids.len());
        }
        // Construction neighbor sets coexist with already-converted neighbor vectors. Each
        // element contributes at most k*(k-1) directed references; sharing can only reduce this.
        add(&mut total, vector::<BTreeSet<usize>>(topology.nodes.len())?)?;
        add(&mut total, index_count::<usize>(neighbor_references)?)?;
        add(&mut total, vector::<Vec<usize>>(topology.nodes.len())?)?;
        add(
            &mut total,
            scale(neighbor_references, 2 * size_of::<usize>() as u64)?,
        )?;
        add(&mut total, slots::<usize>(topology.nodes.len(), 4)?)?;

        // Coordinate lists are local to one element/face. Quadrature values themselves use
        // stack arrays except curved Tri6 boundaries, which retain/refine product-rule vectors.
        add(&mut total, vector::<[f64; 3]>(largest_element)?)?;
        add(&mut total, vector::<[f64; 3]>(largest_face)?)?;
        if topology
            .boundary_faces
            .iter()
            .any(|face| face.order == BoundaryTriangleOrder::Tri6)
        {
            // Actual peak retains the 12x12 rule while building 16x16. Include the preceding
            // 8x8 rule too so this remains conservative over the convergence branch sequence.
            for rule_size in [
                super::quadrature::GAUSS_LEGENDRE_EIGHT_UNIT.len(),
                super::quadrature::GAUSS_LEGENDRE_TWELVE_UNIT.len(),
                super::quadrature::GAUSS_LEGENDRE_SIXTEEN_UNIT.len(),
            ] {
                add(
                    &mut total,
                    slots::<BoundaryQuadraturePoint>(
                        rule_size.checked_mul(rule_size).ok_or_else(overflow)?,
                        1,
                    )?,
                )?;
            }
        }
        // These are exactly the authored metadata families cloned by derive/project_boundary_face.
        add(
            &mut total,
            serialized_materialization_resident_bytes_estimate(
                &(ElementRegions(mesh), FaceProvenance(mesh)),
                PROFILE_LIMITS,
            )
            .map_err(resource)?,
        )?;
        // Projection identity is a small mesh-digest/policy contract; profile the same borrowed
        // shape rather than encoding it merely to count its temporary tree and byte buffer.
        add(
            &mut total,
            serialized_materialization_resident_bytes_estimate(
                &(mesh.canonical_digest, AnalysisProjectionPolicy::default()),
                PROFILE_LIMITS,
            )
            .map_err(resource)?,
        )?;
        Ok(total)
    }
}

struct ElementRegions<'a>(&'a SolverMeshArtifact);
impl Serialize for ElementRegions<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.0
                .topology
                .volume_elements
                .iter()
                .map(|element| &element.region_id),
        )
    }
}
struct FaceProvenance<'a>(&'a SolverMeshArtifact);
impl Serialize for FaceProvenance<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.0
                .topology
                .boundary_faces
                .iter()
                .map(|face| &face.provenance),
        )
    }
}

fn vector<T>(count: usize) -> Result<u64, AnalysisProjectionError> {
    if count == 0 {
        return Ok(0);
    }
    let count = u64::try_from(count).map_err(resource)?;
    scale(
        count
            .checked_mul(2)
            .and_then(|count| count.checked_add(4))
            .ok_or_else(overflow)?,
        size_of::<T>() as u64,
    )
}
fn slots<T>(count: usize, copies: u64) -> Result<u64, AnalysisProjectionError> {
    scale(
        u64::try_from(count).map_err(resource)?,
        (size_of::<T>() as u64)
            .checked_mul(copies)
            .ok_or_else(overflow)?,
    )
}
fn index<T>(count: usize) -> Result<u64, AnalysisProjectionError> {
    index_count::<T>(u64::try_from(count).map_err(resource)?)
}
fn index_count<T>(count: u64) -> Result<u64, AnalysisProjectionError> {
    // Partial BTree nodes and construction scratch; fixed profiler allowances cover tiny roots.
    scale(count, 3 * (size_of::<T>() + 4 * size_of::<usize>()) as u64)
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
