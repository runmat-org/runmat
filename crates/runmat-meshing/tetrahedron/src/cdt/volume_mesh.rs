use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;
use serde::{Deserialize, Serialize};

use super::segment_recovery::recover_delaunay_segments_from_point_set;
use super::topology::build_delaunay_volume_topology_with_regions;
use super::{
    build_delaunay_constraints, build_delaunay_volume_provenance, carve_delaunay_volume,
    evaluate_delaunay_volume_quality, recover_delaunay_facets, DelaunayCarvingOptions,
    DelaunayConstraintOptions, DelaunayFacetSteinerInsertion, DelaunayPointSetOptions,
    DelaunayVolumeNode, DelaunayVolumeProvenance, DelaunayVolumeProvenanceOptions,
    DelaunayVolumeQuality, DelaunayVolumeQualityOptions, DelaunayVolumeRefinementOptions,
    DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions, DelaunayVolumeTopology,
};

#[cfg(test)]
use super::build_delaunay_volume_point_set;

mod centroid;
mod codec;
pub(in crate::cdt) use codec::admission::decode as decode_delaunay_volume_mesh_admitted;
mod components;
#[cfg(test)]
pub(in crate::cdt) use components::assemble as assemble_completed_components;
mod constraint_star;
mod construction;
mod quality_replay;
mod representation;
pub use construction::{
    construct_delaunay_volume_mesh, construct_delaunay_volume_mesh_with_quality_observer,
};
pub use quality_replay::{
    encode_volume_quality_snapshot, prepare_volume_quality_snapshot,
    profile_volume_quality_snapshot, replay_volume_quality, replay_volume_quality_observed,
    volume_quality_snapshot_preparation_storage_bytes, PreparedVolumeQualitySnapshot,
    VolumeQualityBinding, VolumeQualityObserver, VolumeQualityReplayResult, VolumeQualitySnapshot,
};
mod error;
mod frame;
mod quality_convergence;
mod region_seed;
mod validation;

pub use codec::{
    decode_delaunay_volume_mesh, encode_delaunay_volume_mesh, DelaunayVolumeMeshCodecError,
    DelaunayVolumeMeshCodecErrorKind, ValidatedDelaunayVolumeMesh, VolumeMeshEncoding,
    DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};
use error::{
    carving_error, constraint_error, facet_error, point_set_error, provenance_error, quality_error,
    segment_error, sliver_error,
};
pub use validation::validate_delaunay_volume_mesh;

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeMeshOptions {
    pub constraints: DelaunayConstraintOptions,
    pub carving: DelaunayCarvingOptions,
    pub provenance: DelaunayVolumeProvenanceOptions,
    pub quality: DelaunayVolumeQualityOptions,
    pub refinement: DelaunayVolumeRefinementOptions,
    pub sliver: DelaunayVolumeSliverOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelaunayVolumeMesh {
    pub topology: DelaunayVolumeTopology,
    pub provenance: DelaunayVolumeProvenance,
    pub quality: DelaunayVolumeQuality,
    pub facet_recovery_insertions: Vec<DelaunayFacetSteinerInsertion>,
    /// Refinement insertions and optimization mutations in the order they were applied.
    /// Coupled convergence interleaves them, and identity replay depends on that order.
    pub quality_mutations: Vec<DelaunayVolumeQualityMutation>,
}

/// One applied step of volume quality convergence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "mutation",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum DelaunayVolumeQualityMutation {
    Refined(DelaunayVolumeNode),
    Optimized(DelaunayVolumeSliverMutation),
}

impl DelaunayVolumeMesh {
    pub fn refinement_insertions(&self) -> impl Iterator<Item = &DelaunayVolumeNode> {
        self.quality_mutations
            .iter()
            .filter_map(|mutation| match mutation {
                DelaunayVolumeQualityMutation::Refined(node) => Some(node),
                DelaunayVolumeQualityMutation::Optimized(_) => None,
            })
    }
    pub fn optimization_mutations(&self) -> impl Iterator<Item = &DelaunayVolumeSliverMutation> {
        self.quality_mutations
            .iter()
            .filter_map(|mutation| match mutation {
                DelaunayVolumeQualityMutation::Refined(_) => None,
                DelaunayVolumeQualityMutation::Optimized(mutation) => Some(mutation),
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeMeshStage {
    Constraints,
    PointSet,
    SegmentRecovery,
    FacetRecovery,
    Carving,
    Provenance,
    CoordinateMaterialization,
    Quality,
    Refinement,
    Optimization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeMeshErrorKind {
    Accounting(runmat_simulation_usage::LedgerError),
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

fn construct_component_mesh(
    // The one region of a single-region component; joined regions have no constraint star.
    component_region: Option<&runmat_geometry_core::PersistentEntityId>,
    owned_constraints: super::DelaunayConstraints,
    metric_request: &MetricFieldRequest,
    frame: &frame::CanonicalVolumeFrame,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    observer: &dyn quality_replay::VolumeQualityObserver,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshError> {
    let constraints = &owned_constraints;
    let component_span = tracing::info_span!(
        "meshing.tetrahedron.component",
        constraint_nodes = constraints.nodes.len(),
    );
    let _component_guard = component_span.enter();
    let point_set_options = point_set_options(options);
    let (initial, point_set_barriers) = traced_substage("point_set", || {
        let star = match component_region {
            Some(region) => constraint_star::try_build_constraint_star(
                region,
                constraints,
                options.refinement.step.insertion,
                cancellation,
            )?,
            None => None,
        };
        if let Some(star) = star {
            let mut barriers = constraints
                .facets
                .iter()
                .map(|facet| {
                    let mut face = facet
                        .vertex_indices
                        .map(|index| constraints.nodes[index as usize].identity);
                    face.sort_unstable();
                    face
                })
                .collect::<Vec<_>>();
            barriers.sort_unstable();
            barriers.dedup();
            return Ok((star, barriers));
        }
        super::point_set::build_delaunay_volume_point_set_with_order(
            constraints.volume_nodes(),
            point_set_options,
            cancellation,
            |left, right| frame.compare_nodes(left, right),
        )
        .map_err(point_set_error)
        .map(|topology| (topology, Vec::new()))
    })?;
    let initial = traced_substage("point_set_coordinate_normalization", || {
        normalize_point_set_coordinates(
            initial,
            &point_set_barriers,
            options.refinement.step.insertion,
            cancellation,
        )
    })?;
    drop(point_set_barriers);
    let segments = traced_substage("segment_recovery", || {
        recover_delaunay_segments_from_point_set(
            initial,
            constraints,
            options.carving.facet_recovery.segment_recovery,
            cancellation,
        )
        .map_err(segment_error)
    })?;
    let mut facets = traced_substage("facet_recovery", || {
        recover_delaunay_facets(
            segments,
            constraints,
            options.carving.facet_recovery,
            cancellation,
        )
        .map_err(facet_error)
    })?;
    let carving = traced_substage("carving", || {
        carve_delaunay_volume(&facets, constraints, options.carving, cancellation)
            .map_err(carving_error)
    })?;
    let provenance = traced_substage("provenance", || {
        build_delaunay_volume_provenance(
            &facets,
            constraints,
            &carving,
            options.carving,
            options.provenance,
            cancellation,
        )
        .map_err(provenance_error)
    })?;
    // Provenance owns the durable boundary evidence. Recovery's prerequisite and
    // working topologies have no remaining consumer during volume optimization.
    let facet_recovery_insertions = std::mem::take(&mut facets.steiner_insertions);
    drop(facets);
    let protected_faces = provenance
        .facets
        .iter()
        .map(|facet| facet.node_identities)
        .collect::<Vec<_>>();
    let seeded = traced_substage("region_seed", || {
        region_seed::seed_carved_regions(
            carving.topology,
            &protected_faces,
            options.refinement.step.insertion,
            cancellation,
        )
    })?;
    let materialized = traced_substage("coordinate_materialization", || {
        materialize_recovered_coordinates(
            seeded.topology,
            &protected_faces,
            options.refinement.step.insertion,
            cancellation,
        )
    })?;
    drop(protected_faces);
    let quality = traced_substage("quality", || {
        evaluate_delaunay_volume_quality(
            &materialized,
            metric_request,
            &provenance,
            options.quality,
            cancellation,
        )
        .map_err(quality_error)
    })?;
    if let Some(source) = quality.worst_optimization_tetrahedron.and_then(|identity| {
        quality
            .tetrahedra
            .iter()
            .find(|tetrahedron| tetrahedron.node_identities == identity)
    }) {
        let constraint_node_count = source
            .node_identities
            .iter()
            .filter(|identity| {
                constraints
                    .nodes
                    .binary_search_by_key(identity, |node| &node.identity)
                    .is_ok()
            })
            .count();
        let facet_steiner_node_count = source
            .node_identities
            .iter()
            .filter(|identity| {
                facet_recovery_insertions
                    .iter()
                    .any(|insertion| insertion.node_identity == **identity)
            })
            .count();
        tracing::info!(
            constraint_node_count,
            facet_steiner_node_count,
            boundary_node_count = source.boundary_node_count,
            protected_facet_count = source.protected_facet_count,
            metric_radius_edge_ratio = source.metric_radius_edge_ratio,
            metric_scaled_jacobian = source.metric_scaled_jacobian,
            "initial volume defect lineage"
        );
    }
    // Durable provenance and seeded identities now carry every remaining consumer.
    drop(owned_constraints);
    observer.observe(quality_replay::VolumeQualitySnapshot {
        topology: &materialized,
        provenance: &provenance,
        metric: metric_request,
        fixed_nodes: &seeded.node_identities,
        options,
    })?;
    let optimized = traced_substage("quality_convergence", || {
        quality_convergence::converge_volume_quality(
            materialized,
            quality,
            metric_request,
            provenance,
            &seeded.node_identities,
            options,
            quality_convergence::QualityExecution {
                cancellation,
                observer,
            },
        )
    })?;
    Ok(DelaunayVolumeMesh {
        topology: optimized.topology,
        provenance: optimized.provenance,
        quality: optimized.quality,
        facet_recovery_insertions,
        quality_mutations: optimized.quality_mutations,
    })
}

/// Constraint recovery uses exact constructions to prove edge and facet
/// incidence. Once those constraints are recovered and protected, refinement
/// must operate on the floating-point coordinates that the solver artifact
/// actually stores. Rebuilding once at this boundary proves that the recovered
/// topology is valid under those materialized coordinates.
fn materialize_recovered_coordinates(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: super::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayVolumeMeshError> {
    // The recovered incidence is obsolete; release it before building the
    // independently checked incidence at materialized solver coordinates.
    drop(topology.incidence);
    let nodes = topology
        .nodes
        .into_iter()
        .map(|node| {
            DelaunayVolumeNode::explicit(node.identity, node.coordinates_m.canonical_value())
        })
        .collect();
    let tetrahedra = topology
        .tetrahedra
        .into_iter()
        .map(|tetrahedron| (tetrahedron.vertex_indices, tetrahedron.region_id))
        .collect();
    let topology = build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        options.topology,
        cancellation,
    )
    .map_err(|failure| {
        let kind = match failure.kind {
            super::DelaunayTopologyErrorKind::ResourceLimit => {
                DelaunayVolumeMeshErrorKind::ResourceLimit
            }
            super::DelaunayTopologyErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
            super::DelaunayTopologyErrorKind::InvalidNode
            | super::DelaunayTopologyErrorKind::DegenerateTetrahedron => {
                DelaunayVolumeMeshErrorKind::InvalidGeometry
            }
            _ => DelaunayVolumeMeshErrorKind::InvalidTopology,
        };
        error(
            DelaunayVolumeMeshStage::CoordinateMaterialization,
            kind,
            format!("recovered topology is invalid at materialized solver coordinates: {failure}"),
        )
    })?;
    super::insertion::restore_constrained_delaunay_topology(
        topology,
        protected_faces,
        options,
        cancellation,
    )
    .map_err(|failure| coordinate_topology_error(failure, "materialized solver topology"))
}

fn normalize_point_set_coordinates(
    topology: DelaunayVolumeTopology,
    protected_faces: &[[runmat_meshing_core::StableDigest; 3]],
    options: super::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayVolumeMeshError> {
    super::insertion::restore_constrained_delaunay_topology(
        topology,
        protected_faces,
        options,
        cancellation,
    )
    .map_err(|failure| coordinate_topology_error(failure, "initial solver-coordinate topology"))
}

fn coordinate_topology_error(
    failure: super::DelaunayInsertionError,
    context: &str,
) -> DelaunayVolumeMeshError {
    let kind = match failure.kind {
        super::DelaunayInsertionErrorKind::InvalidOptions => {
            DelaunayVolumeMeshErrorKind::InvalidOptions
        }
        super::DelaunayInsertionErrorKind::InvalidNode => {
            DelaunayVolumeMeshErrorKind::InvalidGeometry
        }
        super::DelaunayInsertionErrorKind::ResourceLimit => {
            DelaunayVolumeMeshErrorKind::ResourceLimit
        }
        super::DelaunayInsertionErrorKind::Cancelled => DelaunayVolumeMeshErrorKind::Cancelled,
        super::DelaunayInsertionErrorKind::InvalidTopology
        | super::DelaunayInsertionErrorKind::PointOutsideTopology => {
            DelaunayVolumeMeshErrorKind::InvalidTopology
        }
    };
    error(
        DelaunayVolumeMeshStage::CoordinateMaterialization,
        kind,
        format!("{context} could not be legalized: {failure}"),
    )
}

fn traced_substage<T>(
    stage: &'static str,
    operation: impl FnOnce() -> Result<T, DelaunayVolumeMeshError>,
) -> Result<T, DelaunayVolumeMeshError> {
    let span = tracing::info_span!("meshing.tetrahedron.substage", stage);
    let _guard = span.enter();
    let started = runmat_time::Instant::now();
    tracing::info!(stage, state = "started", "tetrahedral meshing substage");
    let result = operation();
    tracing::info!(
        stage,
        state = if result.is_ok() {
            "completed"
        } else {
            "failed"
        },
        elapsed_milliseconds = started.elapsed().as_millis() as u64,
        "tetrahedral meshing substage"
    );
    result
}

pub(super) fn point_set_options(options: DelaunayVolumeMeshOptions) -> DelaunayPointSetOptions {
    DelaunayPointSetOptions {
        insertion: options.carving.facet_recovery.segment_recovery.insertion,
    }
}

pub(super) fn validate_option_consistency(
    options: DelaunayVolumeMeshOptions,
) -> Result<(), DelaunayVolumeMeshError> {
    super::volume_sliver::validate_options(options.sliver).map_err(sliver_error)?;
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
