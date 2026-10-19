use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingPartitionDescriptor, MetricFieldRequest,
    SurfaceQualityTargets,
};
use runmat_meshing_curve::{canonicalize_shared_curve_splits, SharedCurveMesh};

mod reuse;
mod streaming;
pub use reuse::{AcceptedExactFace, ExactFaceMeshReuse};
pub use streaming::{
    visit_exact_face_partition, visit_exact_face_partition_with_reuse, ExactFacePartitionCompletion,
};

use crate::{
    accept_exact_face_chart_mesh, build_exact_face_charts,
    build_exact_surface_boundary_with_cancellation, join_exact_face_charts,
    recover_exact_face_chart_domains, refine_exact_face_chart_until_blocked,
    triangulate_exact_face_charts, ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions,
    ExactFaceChartDelaunayContext, ExactFaceChartErrorKind, ExactFaceChartRefinedMesh,
    ExactFaceChartRefinementOptions, ExactFaceChartRefinementOutcome, ExactFaceDelaunayOptions,
    ExactFaceJoinContext, ExactFaceJoinErrorKind, ExactFaceJoinOptions, ExactFaceMesh,
    ExactFacePartitionOutcome, ExactFacePartitionResult, ExactFaceRefinementContext,
    ExactFaceRefinementErrorKind, ExactFaceRefinementPolicy, ExactSurfaceBoundaryErrorKind,
    ExactSurfaceMeshErrorKind,
};

#[derive(Clone, Copy)]
pub struct ExactFacePartitionContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub curves: &'a SharedCurveMesh,
    pub metric_request: &'a MetricFieldRequest,
    pub quality: SurfaceQualityTargets,
    pub evaluator: &'a dyn ExactSurfaceEvaluator,
    pub geometry_control: &'a dyn GeometryEvaluationControl,
    pub cancellation: &'a dyn MeshingCancellationSignal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExactFacePartitionOptions {
    pub charts: crate::ExactFaceChartOptions,
    pub delaunay: ExactFaceDelaunayOptions,
    pub refinement: crate::ExactFaceRefinementOptions,
    pub chart_refinement: ExactFaceChartRefinementOptions,
    pub acceptance: ExactFaceAcceptanceOptions,
    pub face_join: ExactFaceJoinOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFacePartitionErrorKind {
    InvalidPartition,
    Reuse,
    Coordinates(runmat_geometry_core::GeometryEvaluationErrorKind),
    Boundary(ExactSurfaceBoundaryErrorKind),
    Chart(ExactFaceChartErrorKind),
    Refinement(ExactFaceRefinementErrorKind),
    Acceptance(ExactFaceAcceptanceErrorKind),
    FaceJoin(ExactFaceJoinErrorKind),
    Batch(ExactSurfaceMeshErrorKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFacePartitionError {
    pub kind: ExactFacePartitionErrorKind,
    pub reason: String,
}

impl std::fmt::Display for ExactFacePartitionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face partition {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for ExactFacePartitionError {}

pub fn mesh_exact_face_partition(
    partition: MeshingPartitionDescriptor,
    context: ExactFacePartitionContext<'_>,
    options: ExactFacePartitionOptions,
) -> Result<ExactFacePartitionResult, ExactFacePartitionError> {
    mesh_exact_face_partition_with_reuse(partition, context, options, &mut reuse::NoFaceReuse)
}

/// Reuse is face-local even when execution schedules several faces in one batch.
/// A converged face is retained before a different face can request curve refinement.
pub fn mesh_exact_face_partition_with_reuse(
    partition: MeshingPartitionDescriptor,
    context: ExactFacePartitionContext<'_>,
    options: ExactFacePartitionOptions,
    reuse: &mut dyn ExactFaceMeshReuse,
) -> Result<ExactFacePartitionResult, ExactFacePartitionError> {
    let mut meshes = Vec::new();
    let completion = process_partition(partition, context, options, reuse, false, &mut |mesh| {
        meshes.push(mesh);
        Ok(())
    })?;
    let outcome = if completion.curve_splits.is_empty() {
        ExactFacePartitionOutcome::Converged { faces: meshes }
    } else {
        ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: completion.curve_splits,
        }
    };
    let boundary = build_exact_surface_boundary_with_cancellation(
        context.topology,
        context.curves,
        context.cancellation,
    )
    .map_err(|error| failure(ExactFacePartitionErrorKind::Boundary(error.kind), error))?;
    crate::surface_mesh::build_exact_face_partition_result_with_boundary(
        context.topology,
        context.curves,
        &boundary,
        completion.partition,
        outcome,
        Some(context.cancellation),
    )
    .map_err(|error| failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
}

fn process_partition(
    partition: MeshingPartitionDescriptor,
    context: ExactFacePartitionContext<'_>,
    options: ExactFacePartitionOptions,
    reuse: &mut dyn ExactFaceMeshReuse,
    release_faces: bool,
    consume: &mut dyn FnMut(ExactFaceMesh) -> Result<(), ExactFacePartitionError>,
) -> Result<ExactFacePartitionCompletion, ExactFacePartitionError> {
    validate_partition(&partition)?;
    let boundary = build_exact_surface_boundary_with_cancellation(
        context.topology,
        context.curves,
        context.cancellation,
    )
    .map_err(|error| failure(ExactFacePartitionErrorKind::Boundary(error.kind), error))?;
    let range = partition
        .entity_range
        .as_ref()
        .expect("partition validated");
    let start = boundary
        .faces
        .partition_point(|face| face.source_face_id < range.first);
    let end = boundary
        .faces
        .partition_point(|face| face.source_face_id <= range.last);
    let faces = &boundary.faces[start..end];
    if faces.len() as u64 != range.entity_count
        || faces
            .first()
            .is_none_or(|face| face.source_face_id != range.first)
        || faces
            .last()
            .is_none_or(|face| face.source_face_id != range.last)
    {
        return Err(invalid_partition(
            "face partition does not cover its declared canonical range",
        ));
    }

    let delaunay_context = ExactFaceChartDelaunayContext {
        topology: context.topology,
        evaluator: context.evaluator,
        geometry_control: context.geometry_control,
        cancellation: context.cancellation,
    };
    let boundary_coordinates =
        crate::ExactBoundaryNodeCoordinates::prepare(context.curves, context.geometry_control)
            .map_err(|error| {
                failure(ExactFacePartitionErrorKind::Coordinates(error.kind), error)
            })?;
    let metric_storage =
        runmat_geometry_core::GeometryAllocationScope::new(context.geometry_control);
    let mut metric_field = None;
    let refinement_context = ExactFaceRefinementContext::new(
        context.topology,
        context.metric_request,
        context.evaluator,
        context.geometry_control,
        context.cancellation,
    )
    .with_boundary_coordinates(&boundary_coordinates);
    let refinement_policy = ExactFaceRefinementPolicy {
        quality: context.quality,
        acceptance: options.acceptance,
        delaunay: options.delaunay,
        refinement: options.refinement,
    };
    let join_context =
        ExactFaceJoinContext::new(refinement_context, context.quality, options.acceptance);
    let mut accepted_face_count = 0;
    let mut splits = Vec::new();
    for face in faces {
        context.geometry_control.checkpoint().map_err(|error| {
            failure(ExactFacePartitionErrorKind::Coordinates(error.kind), error)
        })?;
        if let Some(mesh) = reuse.restore(face, &boundary_coordinates)? {
            if mesh.source_face_id != face.source_face_id {
                return Err(invalid_partition(
                    "cached face has a different source owner",
                ));
            }
            validate_consumed_face(&mesh, face, context)?;
            let _retention =
                streaming::FaceRetention::new(&mesh, context.geometry_control, release_faces)?;
            consume(mesh)?;
            accepted_face_count += 1;
            continue;
        }
        let mesh = {
            if metric_field.is_none() {
                metric_field = Some(
                    crate::ResolvedFaceMetricField::new_with_control(
                        context.topology,
                        context.metric_request,
                        &metric_storage,
                    )
                    .map_err(|error| {
                        failure(
                            ExactFacePartitionErrorKind::Refinement(
                                ExactFaceRefinementErrorKind::Metric(error.kind),
                            ),
                            error,
                        )
                    })?,
                );
            }

            let face_storage =
                runmat_geometry_core::GeometryAllocationScope::new(context.geometry_control);
            let context = ExactFacePartitionContext {
                geometry_control: &face_storage,
                ..context
            };
            let delaunay_context = ExactFaceChartDelaunayContext {
                geometry_control: &face_storage,
                ..delaunay_context
            };
            let refinement_context = ExactFaceRefinementContext {
                metric_field: metric_field.as_ref(),
                geometry_control: &face_storage,
                ..refinement_context
            };
            let join_context = ExactFaceJoinContext {
                refinement: refinement_context,
                ..join_context
            };
            let charts = build_exact_face_charts(
                face,
                context.topology,
                context.evaluator,
                context.geometry_control,
                options.charts,
            )
            .map_err(|error| failure(ExactFacePartitionErrorKind::Chart(error.kind), error))?;
            let triangulations = triangulate_exact_face_charts(
                &charts,
                face,
                delaunay_context,
                options.charts,
                options.delaunay,
            )
            .map_err(|error| failure(ExactFacePartitionErrorKind::Chart(error.kind), error))?;
            let domains = recover_exact_face_chart_domains(
                &triangulations,
                &charts,
                face,
                delaunay_context,
                options.charts,
                options.delaunay,
            )
            .map_err(|error| failure(ExactFacePartitionErrorKind::Chart(error.kind), error))?;
            let mut refined = Vec::<ExactFaceChartRefinedMesh>::with_capacity(charts.charts.len());
            for (chart, domain) in charts.charts.iter().zip(&domains) {
                match refine_exact_face_chart_until_blocked(
                    chart,
                    domain,
                    refinement_context,
                    refinement_policy,
                    options.chart_refinement,
                )
                .map_err(|error| {
                    failure(ExactFacePartitionErrorKind::Refinement(error.kind), error)
                })? {
                    ExactFaceChartRefinementOutcome::Converged(mesh) => refined.push(*mesh),
                    ExactFaceChartRefinementOutcome::RequiresCurveSplits {
                        splits: chart_splits,
                        ..
                    } => {
                        splits.extend(chart_splits.into_iter().map(|split| split.curve_split));
                    }
                }
            }
            if refined.len() != charts.charts.len() {
                continue;
            }
            let acceptance = charts
                .charts
                .iter()
                .zip(&refined)
                .map(|(chart, mesh)| {
                    accept_exact_face_chart_mesh(
                        chart,
                        mesh,
                        refinement_context,
                        context.quality,
                        options.acceptance,
                    )
                    .map_err(|error| {
                        failure(ExactFacePartitionErrorKind::Acceptance(error.kind), error)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mesh = join_exact_face_charts(
                &charts,
                &refined,
                &acceptance,
                join_context,
                options.face_join,
            )
            .map_err(|error| failure(ExactFacePartitionErrorKind::FaceJoin(error.kind), error))?;
            let retained = mesh
                .retained_storage_bytes(&face_storage)
                .map_err(|error| {
                    failure(ExactFacePartitionErrorKind::Coordinates(error.kind), error)
                })?;
            // The accepted mesh has its own storage; it coexists with the working data
            // until this block ends. Transfer only this result to partition ownership.
            face_storage
                .consume_allocation_bytes(retained)
                .map_err(|error| {
                    failure(ExactFacePartitionErrorKind::Coordinates(error.kind), error)
                })?;
            face_storage.retain_bytes(retained);
            mesh
        };
        let _retention =
            streaming::FaceRetention::new(&mesh, context.geometry_control, release_faces)?;
        validate_consumed_face(&mesh, face, context)?;
        reuse.retain(face, &boundary_coordinates, AcceptedExactFace::new(&mesh))?;
        consume(mesh)?;
        accepted_face_count += 1;
    }
    canonicalize_shared_curve_splits(&mut splits);
    if !splits.is_empty() {
        runmat_meshing_curve::validate_shared_curve_split_set_with_cancellation(
            context.curves,
            context.topology,
            &splits,
            context.cancellation,
        )
        .map_err(|error| {
            failure(
                ExactFacePartitionErrorKind::Batch(
                    if error.kind
                        == runmat_meshing_curve::SharedCurveErrorKind::GeometryEvaluation(
                            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
                        )
                    {
                        ExactSurfaceMeshErrorKind::Cancelled
                    } else {
                        ExactSurfaceMeshErrorKind::InvalidInput
                    },
                ),
                error,
            )
        })?;
    }
    Ok(ExactFacePartitionCompletion {
        partition,
        accepted_face_count,
        curve_splits: splits,
    })
}

fn validate_consumed_face(
    mesh: &ExactFaceMesh,
    boundary: &crate::ExactFaceBoundary,
    context: ExactFacePartitionContext<'_>,
) -> Result<(), ExactFacePartitionError> {
    crate::face_mesh::validate_exact_face_mesh_contract_with_cancellation(
        mesh,
        context.topology,
        Some(context.cancellation),
    )
    .map_err(|error| failure(ExactFacePartitionErrorKind::FaceJoin(error.kind), error))?;
    crate::surface_mesh::partition_result::boundary::validate_face_curve_boundary(
        mesh,
        boundary,
        context.topology,
        context.curves,
        Some(context.cancellation),
    )
    .map_err(|error| failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
}

fn validate_partition(
    partition: &MeshingPartitionDescriptor,
) -> Result<(), ExactFacePartitionError> {
    crate::surface_mesh::validate_face_partition_descriptor(partition)
        .map_err(|error| invalid_partition(error.reason))
}

fn invalid_partition(reason: impl Into<String>) -> ExactFacePartitionError {
    ExactFacePartitionError {
        kind: ExactFacePartitionErrorKind::InvalidPartition,
        reason: reason.into(),
    }
}

fn failure(
    kind: ExactFacePartitionErrorKind,
    error: impl std::fmt::Display,
) -> ExactFacePartitionError {
    ExactFacePartitionError {
        kind,
        reason: error.to_string(),
    }
}
