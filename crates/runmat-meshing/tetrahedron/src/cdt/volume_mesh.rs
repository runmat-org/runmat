use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;
use serde::{Deserialize, Serialize};

use super::{
    build_delaunay_constraints, build_delaunay_volume_point_set, build_delaunay_volume_provenance,
    carve_delaunay_volume, evaluate_delaunay_volume_quality, recover_delaunay_facets,
    recover_delaunay_segments, refine_delaunay_volume, DelaunayCarvingOptions,
    DelaunayConstraintOptions, DelaunayFacetSteinerInsertion, DelaunayPointSetOptions,
    DelaunayVolumeProvenance, DelaunayVolumeProvenanceOptions, DelaunayVolumeQuality,
    DelaunayVolumeQualityOptions, DelaunayVolumeRefinementInput, DelaunayVolumeRefinementMutation,
    DelaunayVolumeRefinementOptions, DelaunayVolumeTopology,
};

mod codec;
mod error;
mod validation;

pub use codec::{
    decode_delaunay_volume_mesh, encode_delaunay_volume_mesh, DelaunayVolumeMeshCodecError,
    DelaunayVolumeMeshCodecErrorKind, DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};
use error::{
    carving_error, constraint_error, facet_error, point_set_error, provenance_error, quality_error,
    refinement_error, segment_error,
};
pub use validation::validate_delaunay_volume_mesh;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaunayVolumeMeshOptions {
    pub constraints: DelaunayConstraintOptions,
    pub point_set_validation_check_interval: u64,
    pub carving: DelaunayCarvingOptions,
    pub provenance: DelaunayVolumeProvenanceOptions,
    pub quality: DelaunayVolumeQualityOptions,
    pub refinement: DelaunayVolumeRefinementOptions,
}

impl Default for DelaunayVolumeMeshOptions {
    fn default() -> Self {
        Self {
            constraints: DelaunayConstraintOptions::default(),
            point_set_validation_check_interval: 256,
            carving: DelaunayCarvingOptions::default(),
            provenance: DelaunayVolumeProvenanceOptions::default(),
            quality: DelaunayVolumeQualityOptions::default(),
            refinement: DelaunayVolumeRefinementOptions::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeMesh {
    pub topology: DelaunayVolumeTopology,
    pub provenance: DelaunayVolumeProvenance,
    pub quality: DelaunayVolumeQuality,
    pub facet_recovery_insertions: Vec<DelaunayFacetSteinerInsertion>,
    pub mutations: Vec<DelaunayVolumeRefinementMutation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeMeshStage {
    Constraints,
    PointSet,
    SegmentRecovery,
    FacetRecovery,
    Carving,
    Provenance,
    Quality,
    Refinement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeMeshErrorKind {
    InvalidOptions,
    InvalidGeometry,
    InvalidTopology,
    UnsatisfiableConstraint,
    InvalidMetric,
    InvalidQuality,
    ResourceLimit,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeMeshError {
    pub stage: DelaunayVolumeMeshStage,
    pub kind: DelaunayVolumeMeshErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeMeshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "3D Delaunay volume mesh {:?} {:?}: {}",
            self.stage, self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeMeshError {}

/// Constructs one checked general CDT volume from the authoritative exact surface closure.
/// Scheduling and artifact publication remain outside this geometry-kernel boundary.
pub fn construct_delaunay_volume_mesh(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshError> {
    validate_option_consistency(options)?;
    let constraints =
        build_delaunay_constraints(topology, surface, options.constraints, cancellation)
            .map_err(constraint_error)?;
    let point_set_options = point_set_options(options);
    let initial = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        point_set_options,
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
    let provenance = build_delaunay_volume_provenance(
        &facets,
        &constraints,
        &carving,
        options.carving,
        options.provenance,
        cancellation,
    )
    .map_err(provenance_error)?;
    let quality = evaluate_delaunay_volume_quality(
        &carving.topology,
        metric_request,
        &provenance,
        options.quality,
        cancellation,
    )
    .map_err(quality_error)?;
    let refinement = refine_delaunay_volume(
        DelaunayVolumeRefinementInput {
            topology: &carving.topology,
            metric_request,
            provenance: &provenance,
            quality: &quality,
            quality_options: options.quality,
        },
        options.refinement,
        cancellation,
    )
    .map_err(refinement_error)?;
    let result = DelaunayVolumeMesh {
        topology: refinement.topology,
        provenance,
        quality: refinement.quality,
        facet_recovery_insertions: facets.steiner_insertions,
        mutations: refinement.mutations,
    };
    validate_delaunay_volume_mesh(
        topology,
        surface,
        metric_request,
        &result,
        options,
        cancellation,
    )?;
    Ok(result)
}

pub(super) fn point_set_options(options: DelaunayVolumeMeshOptions) -> DelaunayPointSetOptions {
    DelaunayPointSetOptions {
        insertion: options.carving.facet_recovery.segment_recovery.insertion,
        validation_check_interval: options.point_set_validation_check_interval,
    }
}

pub(super) fn validate_option_consistency(
    options: DelaunayVolumeMeshOptions,
) -> Result<(), DelaunayVolumeMeshError> {
    if options.quality.provenance != options.provenance {
        return Err(error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::InvalidOptions,
            "volume quality and construction must share one provenance policy",
        ));
    }
    Ok(())
}

pub(super) fn error(
    stage: DelaunayVolumeMeshStage,
    kind: DelaunayVolumeMeshErrorKind,
    reason: impl Into<String>,
) -> DelaunayVolumeMeshError {
    DelaunayVolumeMeshError {
        stage,
        kind,
        reason: reason.into(),
    }
}

#[cfg(test)]
#[path = "volume_mesh/tests.rs"]
mod tests;
