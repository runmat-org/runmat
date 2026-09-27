use super::*;

pub(super) async fn convert(
    value: Value,
    template: OutputTemplate,
    mut borrowed: Vec<runmat_accelerate_api::GpuTensorHandle>,
) -> BuiltinResult<Value> {
    // Services may transform the operand; protect both originals and the
    // actual prepared resident payload, and re-admit through the domain body.
    if let Value::GpuTensor(handle) = &value {
        borrowed.push(handle.clone());
    }
    let protected: Vec<_> = borrowed.iter().collect();
    let _borrowed_metadata: Vec<_> = protected
        .iter()
        .map(|handle| crate::builtins::common::gpu_helpers::HandleMetadataRestoreGuard::new(handle))
        .collect();
    let converted = match value {
        Value::GpuTensor(handle) => double_from_gpu(handle, &protected).await?,
        host => host::convert(host)?,
    };
    let owned =
        crate::builtins::common::resident_output::ResidentResultGuard::new(&converted, &protected)
            .map_err(|detail| {
                double_error_with_detail(&runmat_builtins::DOUBLE_ERROR_INTERNAL, detail)
            })?;
    let intermediate = match &converted {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let mut template_protected = protected.clone();
    if let Some(handle) = &intermediate {
        template_protected.push(handle);
    }
    let result = apply_output_template(converted, &template, &template_protected).await?;
    owned.publish(&result);
    Ok(result)
}
