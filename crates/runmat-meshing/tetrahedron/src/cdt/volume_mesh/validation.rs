use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;

use super::{
    carving_error, constraint_error, facet_error, point_set_error, point_set_options,
    provenance_error, quality_error, refinement_error, segment_error, validate_option_consistency,
    DelaunayVolumeMesh, DelaunayVolumeMeshError, DelaunayVolumeMeshOptions,
};
use crate::cdt::{
    build_delaunay_constraints, build_delaunay_volume_point_set, build_delaunay_volume_provenance,
    carve_delaunay_volume, evaluate_delaunay_volume_quality, recover_delaunay_facets,
    recover_delaunay_segments, validate_delaunay_volume_refinement, DelaunayVolumeRefinement,
    DelaunayVolumeRefinementInput,
};

pub fn validate_delaunay_volume_mesh(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    result: &DelaunayVolumeMesh,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeMeshError> {
    validate_option_consistency(options)?;
    let constraints =
        build_delaunay_constraints(topology, surface, options.constraints, cancellation)
            .map_err(constraint_error)?;
    let initial = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        point_set_options(options),
        cancellation,
    )
    .map_err(point_set_error)?;
    let segments = recover_delaunay_segments(
        initial,
        &constraints,
        options.carving.facet_recovery.segment_recovery,
        cancellation,
    )
    .map_err(segment_error)?;
    let facets = recover_delaunay_facets(
        segments,
        &constraints,
        options.carving.facet_recovery,
        cancellation,
    )
    .map_err(facet_error)?;
    let carving = carve_delaunay_volume(&facets, &constraints, options.carving, cancellation)
        .map_err(carving_error)?;
    let expected_provenance = build_delaunay_volume_provenance(
        &facets,
        &constraints,
        &carving,
        options.carving,
        options.provenance,
        cancellation,
    )
    .map_err(provenance_error)?;
    if result.provenance != expected_provenance {
        return Err(super::error(
            super::DelaunayVolumeMeshStage::Provenance,
            super::DelaunayVolumeMeshErrorKind::InvalidGeometry,
            "final volume provenance differs from canonical exact-constraint lineage",
        ));
    }
    let initial_quality = evaluate_delaunay_volume_quality(
        &carving.topology,
        metric_request,
        &result.provenance,
        options.quality,
        cancellation,
    )
    .map_err(quality_error)?;
    validate_delaunay_volume_refinement(
        DelaunayVolumeRefinementInput {
            topology: &carving.topology,
            metric_request,
            provenance: &result.provenance,
            quality: &initial_quality,
            quality_options: options.quality,
        },
        &DelaunayVolumeRefinement {
            topology: result.topology.clone(),
            quality: result.quality.clone(),
            mutations: result.mutations.clone(),
        },
        options.refinement,
        cancellation,
    )
    .map_err(refinement_error)
}
