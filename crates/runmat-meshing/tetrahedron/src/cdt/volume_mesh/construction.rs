//! Construction entry points and component scheduling.
use super::quality_replay::{NoopQualityObserver, VolumeQualityObserver};
use super::*;
use runmat_geometry_core::GeometryEvaluationControl;

/// Constructs one checked general CDT volume from the authoritative exact surface closure.
/// Scheduling and artifact publication remain outside this geometry-kernel boundary.
pub fn construct_delaunay_volume_mesh(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshError> {
    construct_charged(
        topology,
        surface,
        metric_request,
        options,
        cancellation,
        &NoopQualityObserver,
        None,
    )
    .map(ValidatedDelaunayVolumeMesh::into_mesh)
}

/// Constructs and validates one volume mesh, charging the heap the construction holds to
/// `memory` (the run's memory authority) before its working mesh grows. A mesh that needs more
/// than the authority admits stops with [`DelaunayVolumeMeshErrorKind::MemoryBudgetExceeded`].
pub fn construct_delaunay_volume_mesh_with_quality_observer(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    observer: &dyn VolumeQualityObserver,
    memory: &dyn GeometryEvaluationControl,
) -> Result<ValidatedDelaunayVolumeMesh, DelaunayVolumeMeshError> {
    construct_charged(
        topology,
        surface,
        metric_request,
        options,
        cancellation,
        observer,
        Some(memory),
    )
}

fn construct_charged(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    observer: &dyn VolumeQualityObserver,
    memory: Option<&dyn GeometryEvaluationControl>,
) -> Result<ValidatedDelaunayVolumeMesh, DelaunayVolumeMeshError> {
    validate_option_consistency(options)?;
    let result = construct_with_observer(
        topology,
        surface,
        metric_request,
        options,
        cancellation,
        observer,
        memory,
    )?;
    // The finished mesh stays charged while validation rebuilds its constraints beside it.
    let mut charge = super::residency::ResidentCharge::new(memory);
    charge.hold(result.resident_bytes())?;
    super::validation::validate_charged(
        topology,
        surface,
        metric_request,
        &result,
        options,
        cancellation,
        memory,
    )?;
    Ok(ValidatedDelaunayVolumeMesh::new(result))
}

fn construct_with_observer(
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    observer: &dyn VolumeQualityObserver,
    memory: Option<&dyn GeometryEvaluationControl>,
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
    // The ledger charges what construction holds beside the component it is meshing: the
    // partitioned constraints still to mesh and the components already meshed.
    let mut ledger = super::residency::ResidentCharge::new(memory);
    let physical_bytes = super::residency::constraints_bytes(&physical_constraints);
    // Partitioning copies each region's constraints out of the aggregate source.
    ledger.hold(physical_bytes.saturating_mul(2))?;
    let components = components::partition_constraints(&physical_constraints)?;
    // Each partition owns its constraints; the aggregate source is no longer used.
    drop(physical_constraints);
    let mut pending = components
        .iter()
        .map(|component| super::residency::constraints_bytes(&component.constraints))
        .fold(0, u64::saturating_add);
    let mut completed = 0_u64;
    let mut meshes = Vec::with_capacity(components.len());
    let mut remaining_insertions = options.refinement.maximum_insertions;
    for component in components {
        // The component's own constraints are charged by its construction.
        pending =
            pending.saturating_sub(super::residency::constraints_bytes(&component.constraints));
        ledger.hold(pending.saturating_add(completed))?;
        let mut component_options = options;
        component_options.refinement.maximum_insertions = remaining_insertions;
        let mut mesh = construct_component_mesh(
            (!component.joined_regions).then_some(&component.region),
            component.constraints,
            metric_request,
            &frame,
            component_options,
            super::quality_convergence::QualityExecution {
                cancellation,
                observer,
                memory,
            },
        )?;
        let insertion_count = mesh.refinement_insertions().count() as u64;
        remaining_insertions = remaining_insertions.saturating_sub(insertion_count);
        components::remap_facet_insertions(
            &mut mesh.facet_recovery_insertions,
            &component.global_facet_indices,
        )?;
        completed = completed.saturating_add(mesh.resident_bytes());
        ledger.hold(pending.saturating_add(completed))?;
        meshes.push(mesh);
    }
    if meshes.len() > 1 {
        ledger.hold(completed.saturating_add(super::residency::assembly_forecast(&meshes)))?;
    }
    components::assemble(meshes, metric_request, options, cancellation)
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
