use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingPartitionDescriptor, MeshingPartitionKind,
    MetricFieldRequest, SurfaceQualityTargets,
};
use runmat_meshing_curve::{SharedCurveMesh, SharedCurveSegmentSplit};

use crate::{
    accept_exact_face_chart_mesh, build_exact_face_charts, build_exact_face_mesh_batch,
    build_exact_surface_boundary, join_exact_face_charts, recover_exact_face_chart_domains,
    refine_exact_face_chart_until_blocked, triangulate_exact_face_charts,
    ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions, ExactFaceChartDelaunayContext,
    ExactFaceChartErrorKind, ExactFaceChartRefinedMesh, ExactFaceChartRefinementOptions,
    ExactFaceChartRefinementOutcome, ExactFaceDelaunayOptions, ExactFaceJoinContext,
    ExactFaceJoinErrorKind, ExactFaceJoinOptions, ExactFaceMesh, ExactFaceMeshBatch,
    ExactFaceRefinementContext, ExactFaceRefinementErrorKind, ExactFaceRefinementPolicy,
    ExactSurfaceBoundaryErrorKind, ExactSurfaceMeshErrorKind, MAX_EXACT_FACE_PARTITIONS,
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

#[derive(Clone, Debug, PartialEq)]
pub enum ExactFacePartitionOutcome {
    Converged(ExactFaceMeshBatch),
    RequiresCurveSplits(Vec<SharedCurveSegmentSplit>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFacePartitionErrorKind {
    InvalidPartition,
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
) -> Result<ExactFacePartitionOutcome, ExactFacePartitionError> {
    validate_partition(&partition)?;
    let boundary = build_exact_surface_boundary(context.topology, context.curves)
        .map_err(|error| failure(ExactFacePartitionErrorKind::Boundary(error.kind), error))?;
    let range = partition
        .entity_range
        .as_ref()
        .expect("partition validated");
    let faces = boundary
        .faces
        .iter()
        .filter(|face| face.source_face_id >= range.first && face.source_face_id <= range.last)
        .collect::<Vec<_>>();
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
    let refinement_context = ExactFaceRefinementContext::new(
        context.topology,
        context.metric_request,
        context.evaluator,
        context.geometry_control,
        context.cancellation,
    );
    let refinement_policy = ExactFaceRefinementPolicy {
        quality: context.quality,
        delaunay: options.delaunay,
        refinement: options.refinement,
    };
    let join_context =
        ExactFaceJoinContext::new(refinement_context, context.quality, options.acceptance);
    let mut meshes = Vec::<ExactFaceMesh>::with_capacity(faces.len());
    let mut splits = Vec::new();
    for face in faces {
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
            .map_err(|error| failure(ExactFacePartitionErrorKind::Refinement(error.kind), error))?
            {
                ExactFaceChartRefinementOutcome::Converged(mesh) => refined.push(*mesh),
                ExactFaceChartRefinementOutcome::RequiresCurveSplit { split, .. } => {
                    splits.push(split.curve_split);
                }
            }
        }
        if refined.len() != charts.charts.len() {
            continue;
        }
        let acceptance = refined
            .iter()
            .map(|mesh| {
                accept_exact_face_chart_mesh(
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
        meshes.push(
            join_exact_face_charts(
                &charts,
                &refined,
                &acceptance,
                join_context,
                options.face_join,
            )
            .map_err(|error| failure(ExactFacePartitionErrorKind::FaceJoin(error.kind), error))?,
        );
    }
    if !splits.is_empty() {
        canonicalize_splits(&mut splits);
        return Ok(ExactFacePartitionOutcome::RequiresCurveSplits(splits));
    }
    build_exact_face_mesh_batch(context.topology, partition, meshes)
        .map(ExactFacePartitionOutcome::Converged)
        .map_err(|error| failure(ExactFacePartitionErrorKind::Batch(error.kind), error))
}

fn validate_partition(
    partition: &MeshingPartitionDescriptor,
) -> Result<(), ExactFacePartitionError> {
    partition
        .validate()
        .map_err(|error| invalid_partition(error.to_string()))?;
    if partition.kind != MeshingPartitionKind::CanonicalEntityBatch
        || partition.entity_range.is_none()
        || partition.partition_count as usize > MAX_EXACT_FACE_PARTITIONS
    {
        return Err(invalid_partition(
            "face meshing requires a canonical entity-range partition",
        ));
    }
    Ok(())
}

pub(crate) fn canonicalize_splits(splits: &mut Vec<SharedCurveSegmentSplit>) {
    splits.sort_by(|left, right| {
        left.source_edge_id
            .cmp(&right.source_edge_id)
            .then_with(|| left.edge_parameters[0].total_cmp(&right.edge_parameters[0]))
            .then_with(|| left.edge_parameters[1].total_cmp(&right.edge_parameters[1]))
            .then_with(|| left.split_parameter.total_cmp(&right.split_parameter))
            .then_with(|| left.endpoint_node_ids.cmp(&right.endpoint_node_ids))
    });
    splits.dedup();
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
