mod encoding;
pub use encoding::PreparedExactSurfacePassEncoding;
mod owned;
pub use owned::{
    decide_exact_surface_pass_owned, decide_exact_surface_pass_owned_admitted,
    AcceptedExactSurfacePass,
};
pub(super) mod validation;
use super::control::checkpoint;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_curve::{
    apply_shared_curve_splits, canonicalize_shared_curve_splits, validate_shared_curve_split_set,
    SharedCurveDiscretizationOptions, SharedCurveErrorKind, SharedCurveEvaluationContext,
    SharedCurveMesh,
};

use super::{
    ExactFaceMeshBatch, ExactFacePartitionOutcome, ExactFacePartitionResult,
    ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshErrorKind, ExactSurfacePassOutcome,
    ExactSurfacePassResult, EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
    EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
};

#[derive(Clone, Debug, PartialEq)]
pub enum ExactSurfaceConvergenceOutcome {
    RefinedCurves(SharedCurveMesh),
    Converged(ExactSurfaceMesh),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactSurfaceConvergenceErrorKind {
    InvalidResultSet,
    Curve(SharedCurveErrorKind),
    Surface(ExactSurfaceMeshErrorKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactSurfaceConvergenceError {
    pub kind: ExactSurfaceConvergenceErrorKind,
    pub reason: String,
}

impl std::fmt::Display for ExactSurfaceConvergenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact surface convergence {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for ExactSurfaceConvergenceError {}

pub fn resolve_exact_surface_pass(
    current_curves: &SharedCurveMesh,
    results: &[ExactFacePartitionResult],
    curve_context: SharedCurveEvaluationContext<'_>,
    curve_options: SharedCurveDiscretizationOptions,
    surface_options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceConvergenceOutcome, ExactSurfaceConvergenceError> {
    let decision = decide_exact_surface_pass(
        current_curves,
        results,
        curve_context.topology,
        surface_options,
    )?;
    match decision.outcome {
        ExactSurfacePassOutcome::RequiresCurveSplits { splits } => {
            apply_shared_curve_splits(current_curves, curve_context, curve_options, &splits)
                .map(ExactSurfaceConvergenceOutcome::RefinedCurves)
                .map_err(curve_error)
        }
        ExactSurfacePassOutcome::Converged { surface } => {
            Ok(ExactSurfaceConvergenceOutcome::Converged(surface))
        }
    }
}

pub fn decide_exact_surface_pass(
    current_curves: &SharedCurveMesh,
    results: &[ExactFacePartitionResult],
    topology: &ExactBRepTopology,
    surface_options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfacePassResult, ExactSurfaceConvergenceError> {
    decide_with_control(current_curves, results, topology, surface_options, None)
}

pub fn decide_exact_surface_pass_with_cancellation(
    current_curves: &SharedCurveMesh,
    results: &[ExactFacePartitionResult],
    topology: &ExactBRepTopology,
    surface_options: ExactSurfaceJoinOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ExactSurfacePassResult, ExactSurfaceConvergenceError> {
    decide_with_control(
        current_curves,
        results,
        topology,
        surface_options,
        Some(cancellation),
    )
}

pub(super) fn decide_with_control(
    current_curves: &SharedCurveMesh,
    results: &[ExactFacePartitionResult],
    topology: &ExactBRepTopology,
    surface_options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<ExactSurfacePassResult, ExactSurfaceConvergenceError> {
    checkpoint(control).map_err(surface_error)?;
    let ordered = validate_result_set(current_curves, results, topology, control)?;
    let mut splits = Vec::new();
    for result in &ordered {
        checkpoint(control).map_err(surface_error)?;
        if let ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: result_splits,
        } = &result.outcome
        {
            for split in result_splits {
                checkpoint(control).map_err(surface_error)?;
                splits.push(split.clone());
            }
        }
    }
    if !splits.is_empty() {
        canonicalize_shared_curve_splits(&mut splits);
        checkpoint(control).map_err(surface_error)?;
        match control {
            Some(control) => {
                runmat_meshing_curve::validate_shared_curve_split_set_with_cancellation(
                    current_curves,
                    topology,
                    &splits,
                    control,
                )
            }
            None => validate_shared_curve_split_set(current_curves, topology, &splits),
        }
        .map_err(curve_error)?;
        return Ok(ExactSurfacePassResult {
            schema_version: EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
            outcome: ExactSurfacePassOutcome::RequiresCurveSplits { splits },
        });
    }
    let batches = ordered
        .into_iter()
        .map(|result| match &result.outcome {
            ExactFacePartitionOutcome::Converged { faces } => Ok(ExactFaceMeshBatch {
                schema_version: EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
                partition: result.partition.clone(),
                faces: super::copy::clone_faces(faces, control).map_err(surface_error)?,
            }),
            ExactFacePartitionOutcome::RequiresCurveSplits { .. } => Err(invalid(
                "surface convergence lost a validated curve-restart outcome",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    super::join::join_with_control(topology, batches, surface_options, control)
        .map(|surface| ExactSurfacePassResult {
            schema_version: EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
            outcome: ExactSurfacePassOutcome::Converged { surface },
        })
        .map_err(surface_error)
}

fn validate_result_set<'a>(
    curves: &SharedCurveMesh,
    results: &'a [ExactFacePartitionResult],
    topology: &ExactBRepTopology,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<Vec<&'a ExactFacePartitionResult>, ExactSurfaceConvergenceError> {
    if results.is_empty() || results.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid(
            "surface convergence requires a bounded nonempty result set",
        ));
    }
    let boundary = match control {
        Some(control) => {
            crate::build_exact_surface_boundary_with_cancellation(topology, curves, control)
        }
        None => crate::build_exact_surface_boundary(topology, curves),
    }
    .map_err(|error| {
        if error.kind == crate::ExactSurfaceBoundaryErrorKind::Cancelled {
            surface_error(super::ExactSurfaceMeshError::new(
                ExactSurfaceMeshErrorKind::Cancelled,
                error.to_string(),
            ))
        } else {
            invalid(error.to_string())
        }
    })?;
    let mut ordered = results.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|result| result.partition.partition_index);
    let mut covered_faces = Vec::new();
    for (index, result) in ordered.iter().enumerate() {
        checkpoint(control).map_err(surface_error)?;
        super::partition_result::validate_with_boundary(
            result, topology, curves, &boundary, control,
        )
        .map_err(surface_error)?;
        if result.partition.partition_index != index as u32
            || result.partition.partition_count != ordered.len() as u32
        {
            return Err(invalid(
                "surface results do not form one complete canonical partition set",
            ));
        }
        let range = result
            .partition
            .entity_range
            .as_ref()
            .expect("partition result validated");
        for face in &topology.faces {
            checkpoint(control).map_err(surface_error)?;
            if face.id >= range.first && face.id <= range.last {
                covered_faces.push(&face.id);
            }
        }
    }
    if covered_faces.len() != topology.faces.len()
        || covered_faces
            .into_iter()
            .zip(&topology.faces)
            .any(|(actual, expected)| actual != &expected.id)
    {
        return Err(invalid(
            "surface results do not exactly cover the canonical face inventory",
        ));
    }
    Ok(ordered)
}

fn curve_error(error: runmat_meshing_curve::SharedCurveError) -> ExactSurfaceConvergenceError {
    ExactSurfaceConvergenceError {
        kind: ExactSurfaceConvergenceErrorKind::Curve(error.kind),
        reason: error.to_string(),
    }
}

fn surface_error(error: super::ExactSurfaceMeshError) -> ExactSurfaceConvergenceError {
    ExactSurfaceConvergenceError {
        kind: ExactSurfaceConvergenceErrorKind::Surface(error.kind),
        reason: error.to_string(),
    }
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceConvergenceError {
    ExactSurfaceConvergenceError {
        kind: ExactSurfaceConvergenceErrorKind::InvalidResultSet,
        reason: reason.into(),
    }
}
