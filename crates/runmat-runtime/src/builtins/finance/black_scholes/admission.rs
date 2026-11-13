//! Shape and class checks that can run before a pricing provider is touched.

use super::*;

pub(super) fn preflight_numeric_shapes<'a>(
    builtin: &'static str,
    values: impl IntoIterator<Item = &'a Value>,
) -> BuiltinResult<()> {
    let mut result_shape = vec![1, 1];
    for value in values {
        let shape = match value {
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => vec![1, 1],
            Value::Tensor(tensor) => canonical_shape(tensor.shape()),
            Value::LogicalArray(logical) => logical_shape(&logical.shape),
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real =>
            {
                canonical_shape(handle.shape())
            }
            other => {
                return Err(error(
                    builtin,
                    format!("{builtin}: expected real numeric input, got {other:?}"),
                    &ERROR_INVALID_INPUT,
                ));
            }
        };
        shape_len_checked(builtin, &shape)?;
        result_shape = broadcast::broadcast_shapes(builtin, &result_shape, &shape)
            .map_err(|cause| error(builtin, format!("{builtin}: {cause}"), &ERROR_INVALID_INPUT))?;
    }
    shape_len_checked(builtin, &result_shape)?;
    Ok(())
}
