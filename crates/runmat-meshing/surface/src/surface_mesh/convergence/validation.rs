//! Revalidate the decision from immutable prerequisites without cloning a rich mesh.
use super::*;

pub(in crate::surface_mesh) fn validate_decision(
    decision: &ExactSurfacePassResult,
    current_curves: &SharedCurveMesh,
    results: &[ExactFacePartitionResult],
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceConvergenceError> {
    let ordered = validate_result_set(current_curves, results, topology, control)?;
    let mut splits = Vec::new();
    for result in &ordered {
        checkpoint(control).map_err(surface_error)?;
        if let ExactFacePartitionOutcome::RequiresCurveSplits { splits: original } = &result.outcome
        {
            for split in original {
                checkpoint(control).map_err(surface_error)?;
                splits.push(split.clone());
            }
        }
    }
    if !splits.is_empty() {
        canonicalize_shared_curve_splits(&mut splits);
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
        let ExactSurfacePassOutcome::RequiresCurveSplits { splits: actual } = &decision.outcome
        else {
            return Err(invalid(
                "converged decision omits original curve split demands",
            ));
        };
        if actual != &splits {
            return Err(invalid(
                "surface decision differs from original curve splits",
            ));
        }
        return Ok(());
    }
    let ExactSurfacePassOutcome::Converged { surface } = &decision.outcome else {
        return Err(invalid("curve restart has no original split demand"));
    };
    let batches = ordered
        .iter()
        .map(|result| {
            let ExactFacePartitionOutcome::Converged { faces } = &result.outcome else {
                unreachable!("checked no split outcomes");
            };
            super::super::join::validation::Batch {
                schema_version: EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
                partition: &result.partition,
                faces,
            }
        })
        .collect();
    super::super::join::validation::validate(surface, topology, batches, options, control)
        .map_err(surface_error)
}
