use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;

use super::{
    error, validate_option_consistency, DelaunayVolumeMesh, DelaunayVolumeMeshError,
    DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshOptions, DelaunayVolumeMeshStage,
};
use crate::cdt::{
    build_delaunay_constraints, validate_delaunay_volume_provenance,
    validate_delaunay_volume_quality,
};

mod evidence;
mod provenance;

pub fn validate_delaunay_volume_mesh(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    result: &DelaunayVolumeMesh,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeMeshError> {
    validate_charged(
        topology,
        surface,
        metric_request,
        result,
        options,
        cancellation,
        None,
    )
}

/// Validates `result` as [`validate_delaunay_volume_mesh`] does, charging the rebuilt
/// constraints and the bound on every check's scratch to `memory` (the run's memory authority)
/// before the checks allocate it.
pub(in crate::cdt) fn validate_charged(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    result: &DelaunayVolumeMesh,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    memory: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<(), DelaunayVolumeMeshError> {
    validate_option_consistency(options)?;
    let constraints =
        build_delaunay_constraints(topology, surface, options.constraints, cancellation)
            .map_err(super::constraint_error)?;
    let mut charge = super::residency::ResidentCharge::new(memory);
    charge.hold(
        super::residency::constraints_bytes(&constraints).saturating_add(
            super::residency::validation_scratch_bytes(
                result,
                &constraints,
                topology.edges.len(),
                !metric_request.contributions.is_empty(),
            ),
        ),
    )?;
    let protected_faces = result
        .provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    crate::cdt::insertion::validate_conforming_volume_topology(
        &result.topology,
        &protected_faces,
        options.refinement.step.insertion,
        cancellation,
    )
    .map_err(|failure| {
        error(
            DelaunayVolumeMeshStage::Refinement,
            match failure.kind {
                crate::cdt::DelaunayInsertionErrorKind::InvalidOptions => {
                    DelaunayVolumeMeshErrorKind::InvalidOptions
                }
                crate::cdt::DelaunayInsertionErrorKind::ResourceLimit => {
                    DelaunayVolumeMeshErrorKind::ResourceLimit
                }
                crate::cdt::DelaunayInsertionErrorKind::Cancelled => {
                    DelaunayVolumeMeshErrorKind::Cancelled
                }
                crate::cdt::DelaunayInsertionErrorKind::InvalidNode => {
                    DelaunayVolumeMeshErrorKind::InvalidGeometry
                }
                crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
                | crate::cdt::DelaunayInsertionErrorKind::PointOutsideTopology => {
                    DelaunayVolumeMeshErrorKind::InvalidTopology
                }
            },
            failure.to_string(),
        )
    })?;
    validate_delaunay_volume_provenance(
        &result.topology,
        &result.provenance,
        options.provenance,
        cancellation,
    )
    .map_err(super::provenance_error)?;
    provenance::validate_exact_sources(topology, &constraints, result)?;
    evidence::validate_facet_insertions(&constraints, result, options)?;
    evidence::validate_mutations(&constraints, result, options)?;
    validate_delaunay_volume_quality(
        &result.topology,
        metric_request,
        &result.provenance,
        &result.quality,
        options.quality,
        cancellation,
    )
    .map_err(super::quality_error)?;
    if result.quality.worst_refinement_tetrahedron.is_some() {
        return Err(error(
            DelaunayVolumeMeshStage::Refinement,
            DelaunayVolumeMeshErrorKind::InvalidQuality,
            "published volume mesh does not satisfy the configured refinement bounds",
        ));
    }
    if result
        .quality
        .tetrahedra
        .iter()
        .any(|tetrahedron| tetrahedron.requires_optimization())
    {
        return Err(error(
            DelaunayVolumeMeshStage::Optimization,
            DelaunayVolumeMeshErrorKind::InvalidQuality,
            "published volume mesh retains a shape-quality violation",
        ));
    }
    Ok(())
}
