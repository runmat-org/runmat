use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;
use serde::{Deserialize, Serialize};

use super::segment_recovery::recover_delaunay_segments_from_point_set;
use super::topology::build_delaunay_volume_topology_with_regions;
use super::{
    build_delaunay_constraints, build_delaunay_volume_point_set, build_delaunay_volume_provenance,
    carve_delaunay_volume, evaluate_delaunay_volume_quality, recover_delaunay_facets,
    DelaunayCarvingOptions, DelaunayConstraintOptions, DelaunayFacetSteinerInsertion,
    DelaunayPointSetOptions, DelaunayVolumeNode, DelaunayVolumeProvenance,
    DelaunayVolumeProvenanceOptions, DelaunayVolumeQuality, DelaunayVolumeQualityOptions,
    DelaunayVolumeRefinementOptions, DelaunayVolumeSliverMutation, DelaunayVolumeSliverOptions,
    DelaunayVolumeTopology,
};

mod centroid;
mod codec;
mod components;
mod constraint_star;
mod error;
mod frame;
mod quality_convergence;
mod region_seed;
mod validation;

pub use codec::{
    decode_delaunay_volume_mesh, encode_delaunay_volume_mesh, DelaunayVolumeMeshCodecError,
    DelaunayVolumeMeshCodecErrorKind, DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};
use error::{
    carving_error, constraint_error, facet_error, point_set_error, provenance_error, quality_error,
    segment_error, sliver_error,
};
pub use validation::validate_delaunay_volume_mesh;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    pub refinement_insertions: Vec<DelaunayVolumeNode>,
    pub sliver_mutations: Vec<DelaunayVolumeSliverMutation>,
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
    let result = construct_delaunay_volume_mesh_unvalidated(
        topology,
        surface,
        metric_request,
        options,
        cancellation,
    )?;
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

pub(super) fn construct_delaunay_volume_mesh_unvalidated(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshError> {
    validate_option_consistency(options)?;
    trace_surface_contract(surface);
    let physical_constraints =
        build_delaunay_constraints(topology, surface, options.constraints, cancellation)
            .map_err(constraint_error)?;
    let frame =
        frame::CanonicalVolumeFrame::from_constraints(&physical_constraints).ok_or_else(|| {
            error(
                DelaunayVolumeMeshStage::PointSet,
                DelaunayVolumeMeshErrorKind::InvalidGeometry,
                "solid boundary cannot define a nondegenerate canonical volume frame",
            )
        })?;
    let constraints = frame.canonicalize_constraints(&physical_constraints);
    let canonical_metric_request =
        frame
            .canonicalize_metric_request(metric_request)
            .map_err(|failure| {
                error(
                    DelaunayVolumeMeshStage::Quality,
                    DelaunayVolumeMeshErrorKind::InvalidMetric,
                    failure.to_string(),
                )
            })?;
    let components = components::partition_constraints(&constraints)?;
    let mut meshes = Vec::with_capacity(components.len());
    let mut remaining_insertions = options.refinement.maximum_insertions;
    for component in components {
        let mut component_options = options;
        component_options.refinement.maximum_insertions = remaining_insertions;
        let mut mesh = construct_component_mesh(
            &component.region,
            &component.constraints,
            &canonical_metric_request,
            component_options,
            cancellation,
        )?;
        let insertion_count = mesh.refinement_insertions.len() as u64;
        remaining_insertions = remaining_insertions.saturating_sub(insertion_count);
        components::remap_facet_insertions(
            &mut mesh.facet_recovery_insertions,
            &component.global_facet_indices,
        )?;
        meshes.push(mesh);
    }
    let assembled_topology = components::assemble_topology(
        &meshes,
        options.refinement.step.insertion.topology,
        cancellation,
    )?;
    let provenance = components::assemble_provenance(&meshes);
    let quality = evaluate_delaunay_volume_quality(
        &assembled_topology,
        &canonical_metric_request,
        &provenance,
        options.quality,
        cancellation,
    )
    .map_err(quality_error)?;
    let mut result = DelaunayVolumeMesh {
        topology: assembled_topology,
        provenance,
        quality,
        facet_recovery_insertions: components::assemble_insertions(&meshes),
        refinement_insertions: components::assemble_refinement_insertions(&meshes),
        sliver_mutations: components::assemble_sliver_mutations(&meshes),
    };
    frame.restore_mesh(&mut result, &physical_constraints);
    let restored_topology = result.topology;
    result.topology = build_delaunay_volume_topology_with_regions(
        restored_topology.nodes,
        restored_topology
            .tetrahedra
            .into_iter()
            .map(|tetrahedron| (tetrahedron.vertex_indices, tetrahedron.region_id))
            .collect(),
        options.refinement.step.insertion.topology,
        cancellation,
    )
    .map_err(|failure| {
        error(
            DelaunayVolumeMeshStage::CoordinateMaterialization,
            DelaunayVolumeMeshErrorKind::InvalidTopology,
            format!("canonical volume frame could not restore solver coordinates: {failure}"),
        )
    })?;
    result.quality = evaluate_delaunay_volume_quality(
        &result.topology,
        metric_request,
        &result.provenance,
        options.quality,
        cancellation,
    )
    .map_err(quality_error)?;
    Ok(result)
}

fn trace_surface_contract(surface: &ExactSurfaceMesh) {
    let minimum_metric_edge = surface
        .triangles
        .iter()
        .flat_map(|triangle| triangle.metric_edge_lengths)
        .fold(f64::INFINITY, f64::min);
    let minimum_metric_angle_degrees = surface
        .triangles
        .iter()
        .map(|triangle| triangle.minimum_metric_angle_rad.to_degrees())
        .fold(f64::INFINITY, f64::min);
    let maximum_physical_aspect_ratio = surface
        .triangles
        .iter()
        .map(|triangle| triangle.physical_aspect_ratio)
        .fold(0.0_f64, f64::max);
    let span = tracing::info_span!(
        "meshing.tetrahedron.surface_contract",
        surface_node_count = surface.nodes.len(),
        surface_triangle_count = surface.triangles.len(),
        minimum_metric_edge,
        minimum_metric_angle_degrees,
        maximum_physical_aspect_ratio,
    );
    span.in_scope(|| {});
}

fn construct_component_mesh(
    component_region: &runmat_geometry_core::PersistentEntityId,
    constraints: &super::DelaunayConstraints,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshError> {
    let component_span = tracing::info_span!(
        "meshing.tetrahedron.component",
        constraint_nodes = constraints.volume_nodes().len(),
    );
    let _component_guard = component_span.enter();
    let point_set_options = point_set_options(options);
    let (initial, point_set_barriers) = traced_substage("point_set", || {
        if let Some(star) = constraint_star::try_build_constraint_star(
            component_region,
            constraints,
            options.refinement.step.insertion.topology,
            cancellation,
        )? {
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
        build_delaunay_volume_point_set(constraints.volume_nodes(), point_set_options, cancellation)
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
    let segments = traced_substage("segment_recovery", || {
        recover_delaunay_segments_from_point_set(
            initial,
            constraints,
            options.carving.facet_recovery.segment_recovery,
            cancellation,
        )
        .map_err(segment_error)
    })?;
    let facets = traced_substage("facet_recovery", || {
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
                facets
                    .steiner_insertions
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
    let optimized = traced_substage("quality_convergence", || {
        quality_convergence::converge_volume_quality(
            materialized,
            quality,
            metric_request,
            provenance,
            &seeded.node_identities,
            options,
            cancellation,
        )
    })?;
    Ok(DelaunayVolumeMesh {
        topology: optimized.topology,
        provenance: optimized.provenance,
        quality: optimized.quality,
        facet_recovery_insertions: facets.steiner_insertions,
        refinement_insertions: optimized.refinement_insertions,
        sliver_mutations: optimized.optimization_mutations,
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
    let nodes = topology
        .nodes
        .into_iter()
        .map(|node| DelaunayVolumeNode::explicit(node.identity, node.coordinates_m))
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
    super::insertion::legalize_delaunay_volume_topology_completely_with_barriers(
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
    super::insertion::legalize_delaunay_volume_topology_completely_with_barriers(
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
    let started = std::time::Instant::now();
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
