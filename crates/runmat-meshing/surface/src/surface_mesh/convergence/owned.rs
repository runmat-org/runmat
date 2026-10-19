//! Construction-only producer result. Original partitions are validated before
//! their mesh buffers move into the join; no rich input inventory is cloned.
use super::*;

pub struct AcceptedExactSurfacePass {
    result: ExactSurfacePassResult,
}
impl AcceptedExactSurfacePass {
    pub fn result(&self) -> &ExactSurfacePassResult {
        &self.result
    }
    pub fn encoding_preparation_storage_bytes(
        &self,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<u64, super::super::ExactSurfaceMeshError> {
        super::encoding::preparation_bytes(self, signal)
    }
    pub fn prepare_encoding(
        &self,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<super::PreparedExactSurfacePassEncoding<'_>, super::super::ExactSurfaceMeshError>
    {
        super::encoding::prepare(self, signal)
    }
    pub fn encode_with_cancellation(
        &self,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, super::super::ExactSurfaceMeshError> {
        match &self.result.outcome {
            ExactSurfacePassOutcome::Converged { surface } => super::super::surface_codec::encode_accepted_surface(surface, control),
            ExactSurfacePassOutcome::RequiresCurveSplits { .. } => super::super::pass_result::encode_decided_exact_surface_pass_result_with_cancellation(&self.result, control),
        }
    }
}

pub fn decide_exact_surface_pass_owned(
    current_curves: &SharedCurveMesh,
    results: Vec<ExactFacePartitionResult>,
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    control: &dyn MeshingCancellationSignal,
) -> Result<AcceptedExactSurfacePass, ExactSurfaceConvergenceError> {
    decide_exact_surface_pass_owned_admitted(
        current_curves,
        results,
        topology,
        options,
        control,
        None,
    )
}

/// Move validated partitions into the canonical join with optional storage/work
/// admission. Retained output charges belong to the caller's allocation scope;
/// the caller must keep them until the accepted value is dropped.
pub fn decide_exact_surface_pass_owned_admitted(
    current_curves: &SharedCurveMesh,
    mut results: Vec<ExactFacePartitionResult>,
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    control: &dyn MeshingCancellationSignal,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<AcceptedExactSurfacePass, ExactSurfaceConvergenceError> {
    // Admission remains independent of producer claims and current shared curves.
    validate_result_set(current_curves, &results, topology, Some(control))?;
    results.sort_by_key(|result| result.partition.partition_index);
    let has_splits = results.iter().any(|result| {
        matches!(
            result.outcome,
            ExactFacePartitionOutcome::RequiresCurveSplits { .. }
        )
    });
    let outcome = if has_splits {
        let mut splits = Vec::new();
        for result in results {
            checkpoint(Some(control)).map_err(surface_error)?;
            if let ExactFacePartitionOutcome::RequiresCurveSplits {
                splits: mut demands,
            } = result.outcome
            {
                splits.append(&mut demands);
            }
        }
        canonicalize_shared_curve_splits(&mut splits);
        runmat_meshing_curve::validate_shared_curve_split_set_with_cancellation(
            current_curves,
            topology,
            &splits,
            control,
        )
        .map_err(curve_error)?;
        ExactSurfacePassOutcome::RequiresCurveSplits { splits }
    } else {
        let _batches_storage =
            super::super::join::nodes::reserve::<ExactFaceMeshBatch>(results.len(), allocations)
                .map_err(surface_error)?;
        let mut batches = Vec::with_capacity(results.len());
        for result in results {
            checkpoint(Some(control)).map_err(surface_error)?;
            let ExactFacePartitionOutcome::Converged { faces } = result.outcome else {
                unreachable!("all outcomes checked");
            };
            batches.push(ExactFaceMeshBatch {
                schema_version: EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
                partition: result.partition,
                faces,
            });
        }
        let surface = super::super::join::join_with_admission(
            topology,
            batches,
            options,
            Some(control),
            allocations,
        )
        .map_err(surface_error)?;
        super::super::artifact::validate_admitted(
            &surface,
            topology,
            options,
            Some(control),
            allocations,
        )
        .map_err(surface_error)?;
        ExactSurfacePassOutcome::Converged { surface }
    };
    Ok(AcceptedExactSurfacePass {
        result: ExactSurfacePassResult {
            schema_version: EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
            outcome,
        },
    })
}
