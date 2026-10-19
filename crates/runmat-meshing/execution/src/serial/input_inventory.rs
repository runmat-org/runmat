//! Compare the immutable union through one admitted reference buffer. Shared
//! objects count once; digest collisions with different lengths still reject.
use super::*;
use runmat_execution_artifact::LogicalObject;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};

pub(super) fn validate(
    inputs: &[PreparedMeshingInput],
    limits: ObjectInventoryLimits,
    control: &dyn GeometryEvaluationControl,
    stage: MeshingStageKind,
) -> Result<(), MeshingSerialExecutionError> {
    let count = inputs
        .iter()
        .try_fold(0usize, |sum, input| sum.checked_add(input.objects().len()))
        .ok_or_else(|| {
            MeshingExecutionError::Invalid("combined input object inventory count overflow".into())
        })?;
    let bytes = crate::object_support::import_slots::<&LogicalObject>(count)?;
    let map_control = |error: runmat_geometry_core::GeometryEvaluationError| {
        MeshingSerialExecutionError::Stage(failure(
            stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        ))
    };
    let _storage = GeometryAllocationReservation::new(control, bytes).map_err(map_control)?;
    let mut objects = Vec::with_capacity(count);
    for input in inputs {
        for object in input.objects() {
            control.checkpoint().map_err(map_control)?;
            objects.push(object);
        }
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(
        &mut objects,
        |left, right| left.descriptor.digest.cmp(&right.descriptor.digest),
        || {
            control.consume_search_work(1)?;
            control.checkpoint()
        },
    )
    .map_err(map_control)?;
    let mut previous = None;
    let mut unique = 0usize;
    let mut total = 0u64;
    for object in objects {
        control.checkpoint().map_err(map_control)?;
        let key = (object.descriptor.digest, object.descriptor.encoded_length);
        if let Some((digest, length)) = previous {
            if digest == key.0 {
                if length != key.1 {
                    return Err(MeshingExecutionError::Identity(
                        "input object digest was reused with a different length",
                    )
                    .into());
                }
                continue;
            }
        }
        previous = Some(key);
        unique += 1;
        total = total.checked_add(key.1).ok_or_else(|| {
            MeshingExecutionError::Invalid("input artifact byte total overflow".into())
        })?;
        if unique > limits.max_objects || total > limits.max_total_bytes {
            return Err(MeshingExecutionError::Invalid(
                "combined meshing input artifact inventory exceeds its hard bound".into(),
            )
            .into());
        }
    }
    Ok(())
}
