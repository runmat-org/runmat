use std::f64::consts::PI;

use runmat_execution::Digest;
use runmat_geometry_core::{
    ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator, GeometryContractError,
    GeometryModel, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingDiagnosticEntry, MeshingDiagnosticValue,
    MeshingFailure, MeshingFailureCategory, MeshingOperation, MeshingRequest, MeshingStageKind,
    StableDigest, MESHING_FAILURE_SCHEMA_VERSION,
};
use runmat_meshing_curve::{
    derive_curve_geometry_metric, discretize_shared_curve_partition, encode_shared_curve_batch,
    CurveResolutionPolicy, ResolvedCurveMetricField, SharedCurveDiscretizationOptions,
    SharedCurveError, SharedCurveErrorKind, SHARED_CURVE_BATCH_SCHEMA_VERSION,
};

use crate::{
    MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel,
    PreparedExactGeometryObjects, PreparedMeshingInput, ValidatedMeshingStageOutput,
};

pub trait ExactCurveGeometryEvaluation:
    ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator
{
}

impl<T> ExactCurveGeometryEvaluation for T where
    T: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator
{
}

pub trait ExactCurveEvaluatorProvider: Send + Sync {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
    ) -> Result<Box<dyn ExactCurveGeometryEvaluation + 'a>, GeometryContractError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PortableCurveEvaluatorProvider;

impl ExactCurveEvaluatorProvider for PortableCurveEvaluatorProvider {
    fn evaluator<'a>(
        &self,
        geometry: &'a PreparedExactGeometryObjects,
    ) -> Result<Box<dyn ExactCurveGeometryEvaluation + 'a>, GeometryContractError> {
        let GeometryModel::ExactBRep { model } = &geometry.document.model else {
            return Err(GeometryContractError::invalid(
                "curve stage geometry",
                "portable curve evaluation requires exact B-rep geometry",
            ));
        };
        Ok(Box::new(PortableExactEvaluator::new(
            &geometry.evaluators,
            &geometry.topology,
            model,
        )?))
    }
}

#[derive(Clone, Debug)]
pub struct ExactCurveStageKernel<P = PortableCurveEvaluatorProvider> {
    evaluator_provider: P,
}

impl Default for ExactCurveStageKernel<PortableCurveEvaluatorProvider> {
    fn default() -> Self {
        Self {
            evaluator_provider: PortableCurveEvaluatorProvider,
        }
    }
}

impl<P> ExactCurveStageKernel<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self { evaluator_provider }
    }
}

impl<P: ExactCurveEvaluatorProvider> MeshingStageKernel for ExactCurveStageKernel<P> {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::CurveMesh {
            return Err(curve_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                None,
                "use the exact curve kernel only for curve-mesh stages",
                "curve stage",
            ));
        }
        let geometry = exact_geometry(invocation.inputs)?;
        let evaluator = self
            .evaluator_provider
            .evaluator(geometry)
            .map_err(|error| {
                curve_failure(
                    MeshingFailureCategory::InvalidGeometry,
                    None,
                    "regenerate an evaluator-complete exact geometry closure",
                    &error.to_string(),
                )
            })?;
        let control = invocation.control.geometry_evaluation_control();
        let metric = resolved_curve_metric(
            geometry,
            evaluator.as_ref(),
            &control,
            &invocation.host.resolved_request,
        )?;
        if invocation.host.resolved_request.resources.maximum_nodes < 2 {
            return Err(curve_failure(
                MeshingFailureCategory::NodeBudgetExceeded,
                None,
                "increase the node budget to at least two nodes per ordinary exact edge",
                "curve node budget is below the constructive minimum",
            ));
        }
        let options = curve_options(invocation.host);
        let batch = discretize_shared_curve_partition(
            &geometry.topology,
            evaluator.as_ref(),
            evaluator.as_ref(),
            &metric,
            &control,
            options,
            invocation.host.workload.partition.clone(),
        )
        .map_err(map_curve_error)?;
        let encoded =
            encode_shared_curve_batch(&batch, &geometry.topology).map_err(map_curve_error)?;
        let usage = control.usage();
        drop(control);

        let node_count = batch
            .edges
            .iter()
            .try_fold(0_u64, |count, edge| {
                count.checked_add(edge.nodes.len() as u64)
            })
            .ok_or_else(|| {
                curve_failure(
                    MeshingFailureCategory::NodeBudgetExceeded,
                    None,
                    "increase the node budget or relax curve quality targets",
                    "curve node count overflowed",
                )
            })?;
        let edge_count = batch.edges.len() as u64;
        let memory_bytes = usage.allocation_bytes.saturating_add(encoded.len() as u64);
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("curve_edges".into(), edge_count);
        entity_counts.insert("curve_nodes".into(), node_count);
        let checkpoint = MeshingStageCheckpoint {
            completed_work: edge_count,
            estimated_work: edge_count,
            node_count,
            peak_memory_bytes: memory_bytes,
            search_work: usage.search_work,
            iterations: usage.iterations,
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: validation_digest(&encoded),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::CurvePartitions,
                schema_version: SHARED_CURVE_BATCH_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

fn exact_geometry(
    inputs: &[PreparedMeshingInput],
) -> Result<&PreparedExactGeometryObjects, Box<MeshingFailure>> {
    match inputs {
        [PreparedMeshingInput::ExactGeometry(input)] => Ok(input.geometry_objects()),
        _ => Err(curve_failure(
            MeshingFailureCategory::InternalInvariantViolation,
            None,
            "submit exactly one admitted exact geometry closure to the curve stage",
            "curve input inventory",
        )),
    }
}

pub(super) fn curve_options(host: &crate::MeshingHostWorkload) -> SharedCurveDiscretizationOptions {
    let request = &host.resolved_request;
    let curve = request.quality.curve;
    let numerical_error_m = request
        .tolerance
        .absolute_floor_m
        .max(request.tolerance.requested_deviation_m * 1.0e-6)
        .max(f64::EPSILON);
    SharedCurveDiscretizationOptions {
        resolution: CurveResolutionPolicy {
            maximum_chordal_deviation_m: curve.maximum_chordal_deviation_m,
            maximum_tangent_change_rad: curve.maximum_tangent_change_degrees * PI / 180.0,
            minimum_metric_edge_length: curve.minimum_metric_edge_length,
            maximum_metric_edge_length: curve.maximum_metric_edge_length,
        },
        maximum_nodes_per_edge: request.resources.maximum_nodes.min(u64::from(u32::MAX)) as u32,
        maximum_subdivision_depth: request
            .resources
            .maximum_recursion_depth
            .min(u32::from(u16::MAX)) as u16,
        geometry_absolute_error_m: numerical_error_m,
        pcurve_absolute_error: 1.0e-10,
        arc_length_absolute_error_m: numerical_error_m,
    }
}

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-exact-curve-partition-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}

pub(super) fn map_curve_error(error: SharedCurveError) -> Box<MeshingFailure> {
    let category = shared_curve_failure_category(error.kind);
    curve_failure(
        category,
        error.edge_id,
        "repair the named geometry or relax the resolved curve constraints",
        &format!("{}: {}", error.field, error.reason),
    )
}

pub(crate) fn shared_curve_failure_category(kind: SharedCurveErrorKind) -> MeshingFailureCategory {
    match kind {
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
        ) => MeshingFailureCategory::Cancelled,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::TimeBudgetExceeded,
        ) => MeshingFailureCategory::TimeBudgetExceeded,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::AllocationBudgetExceeded,
        ) => MeshingFailureCategory::MemoryBudgetExceeded,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
        ) => MeshingFailureCategory::SearchWorkBudgetExceeded,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::IterationBudgetExceeded,
        ) => MeshingFailureCategory::IterationBudgetExceeded,
        SharedCurveErrorKind::MetricEvaluation | SharedCurveErrorKind::UnsatisfiedConstraint => {
            MeshingFailureCategory::SizingConflict
        }
        SharedCurveErrorKind::ResourceLimit => MeshingFailureCategory::NodeBudgetExceeded,
        SharedCurveErrorKind::InvalidContract
        | SharedCurveErrorKind::InvalidEncoding
        | SharedCurveErrorKind::InvalidRequest
        | SharedCurveErrorKind::GeometricMismatch => MeshingFailureCategory::InvalidGeometry,
        SharedCurveErrorKind::GeometryEvaluation(_) => MeshingFailureCategory::NumericalFailure,
    }
}

pub(crate) fn resolved_curve_metric(
    geometry: &PreparedExactGeometryObjects,
    evaluator: &dyn ExactCurveGeometryEvaluation,
    control: &crate::MeshingGeometryEvaluationControl<'_>,
    request: &MeshingRequest,
) -> Result<ResolvedCurveMetricField, Box<MeshingFailure>> {
    let metric_request = derive_curve_geometry_metric(
        &geometry.topology,
        evaluator,
        control,
        &request.metric,
        request.quality.curve,
        request.quality.surface,
    )
    .map_err(map_curve_error)?;
    ResolvedCurveMetricField::new(&geometry.topology, &metric_request).map_err(map_curve_error)
}

pub(super) fn curve_failure(
    category: MeshingFailureCategory,
    edge_id: Option<runmat_geometry_core::PersistentEntityId>,
    remediation: &str,
    detail: &str,
) -> Box<MeshingFailure> {
    let detail = crate::diagnostic::bounded_diagnostic_text(detail, "curve stage failure");
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category,
        stage: MeshingStageKind::CurveMesh,
        operation: MeshingOperation::DiscretizeCurve,
        entity_ids: edge_id.into_iter().collect(),
        witnesses: Vec::new(),
        request_values: Vec::new(),
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "curve_failure".into(),
            value: MeshingDiagnosticValue::Text(detail),
            unit: None,
        }],
        remediation: remediation.into(),
    })
}
