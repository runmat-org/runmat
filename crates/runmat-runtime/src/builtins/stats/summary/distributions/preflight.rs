//! Host-visible distribution admission before any provider download.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum NumericShapePolicy {
    Broadcast,
    SameSizeOrScalar,
}

/// Check only facts available from the value or resident descriptor. Numeric
/// values and fitted parameters are evaluated after this effect-free boundary.
pub(super) fn numeric_inputs(
    name: &str,
    values: &[&Value],
    policy: NumericShapePolicy,
) -> BuiltinResult<()> {
    let mut shape: Option<&[usize]> = None;
    let mut broadcast_shape: Option<Vec<usize>> = None;
    for value in values {
        let current = match value {
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => &[1, 1][..],
            Value::Tensor(tensor) => tensor.shape(),
            Value::LogicalArray(array) => array.shape.as_slice(),
            Value::GpuTensor(handle)
                if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
            {
                handle.shape()
            }
            _ => {
                return Err(normal_error(
                    name,
                    format!("{name}: expected real numeric or logical inputs"),
                ));
            }
        };
        if crate::builtins::common::spec::checked_reduction_elements(current).is_none() {
            return Err(normal_error(
                name,
                format!("{name}: input shape exceeds structural range"),
            ));
        }
        match policy {
            NumericShapePolicy::Broadcast => {
                broadcast_shape = Some(match broadcast_shape {
                    Some(previous) => broadcast::broadcast_shapes(name, &previous, current)
                        .map_err(|error| normal_error(name, error))?,
                    None => current.to_vec(),
                });
            }
            NumericShapePolicy::SameSizeOrScalar => {
                if let Some(previous) = shape {
                    let previous_scalar =
                        crate::builtins::common::spec::checked_reduction_elements(previous)
                            == Some(1);
                    let current_scalar =
                        crate::builtins::common::spec::checked_reduction_elements(current)
                            == Some(1);
                    if !previous_scalar && !current_scalar && previous != current {
                        return Err(normal_error(
                            name,
                            format!("{name}: x and nu must have the same size unless one input is scalar"),
                        ));
                    }
                } else {
                    shape = Some(current);
                }
            }
        }
    }
    if broadcast_shape.as_ref().is_some_and(|shape| {
        crate::builtins::common::spec::checked_reduction_elements(shape).is_none()
    }) {
        return Err(normal_error(
            name,
            format!("{name}: output shape exceeds structural range"),
        ));
    }
    Ok(())
}
