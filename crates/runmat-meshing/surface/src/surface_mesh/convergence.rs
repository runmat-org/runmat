use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_curve::{
    apply_shared_curve_splits, canonicalize_shared_curve_splits, validate_shared_curve_split_set,
    SharedCurveDiscretizationOptions, SharedCurveErrorKind, SharedCurveEvaluationContext,
    SharedCurveMesh,
};

use super::{
    join_exact_face_mesh_batches, validate_exact_face_partition_result_with_boundary,
    ExactFaceMeshBatch, ExactFacePartitionOutcome, ExactFacePartitionResult,
    ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshErrorKind,
    EXACT_FACE_MESH_BATCH_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
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
    mut results: Vec<ExactFacePartitionResult>,
    curve_context: SharedCurveEvaluationContext<'_>,
    curve_options: SharedCurveDiscretizationOptions,
    surface_options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceConvergenceOutcome, ExactSurfaceConvergenceError> {
    validate_result_set(current_curves, &mut results, curve_context.topology)?;
    let mut splits = Vec::new();
    for result in &results {
        if let ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: result_splits,
        } = &result.outcome
        {
            splits.extend(result_splits.iter().cloned());
        }
    }
    if !splits.is_empty() {
        canonicalize_shared_curve_splits(&mut splits);
        validate_shared_curve_split_set(current_curves, curve_context.topology, &splits)
            .map_err(curve_error)?;
        return apply_shared_curve_splits(current_curves, curve_context, curve_options, &splits)
            .map(ExactSurfaceConvergenceOutcome::RefinedCurves)
            .map_err(curve_error);
    }
    let batches = results
        .into_iter()
        .map(|result| match result.outcome {
            ExactFacePartitionOutcome::Converged { faces } => Ok(ExactFaceMeshBatch {
                schema_version: EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
                partition: result.partition,
                faces,
            }),
            ExactFacePartitionOutcome::RequiresCurveSplits { .. } => Err(invalid(
                "surface convergence lost a validated curve-restart outcome",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    join_exact_face_mesh_batches(curve_context.topology, batches, surface_options)
        .map(ExactSurfaceConvergenceOutcome::Converged)
        .map_err(surface_error)
}

fn validate_result_set(
    curves: &SharedCurveMesh,
    results: &mut [ExactFacePartitionResult],
    topology: &ExactBRepTopology,
) -> Result<(), ExactSurfaceConvergenceError> {
    if results.is_empty() || results.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid(
            "surface convergence requires a bounded nonempty result set",
        ));
    }
    let boundary = crate::build_exact_surface_boundary(topology, curves)
        .map_err(|error| invalid(error.to_string()))?;
    results.sort_by_key(|result| result.partition.partition_index);
    let mut covered_faces = Vec::new();
    for (index, result) in results.iter().enumerate() {
        validate_exact_face_partition_result_with_boundary(result, topology, curves, &boundary)
            .map_err(surface_error)?;
        if result.partition.partition_index != index as u32
            || result.partition.partition_count != results.len() as u32
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
        covered_faces.extend(
            topology
                .faces
                .iter()
                .filter(|face| face.id >= range.first && face.id <= range.last)
                .map(|face| &face.id),
        );
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
    Ok(())
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
