use super::*;

pub(super) async fn convert(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    template: OutputTemplate,
    mut borrowed: Vec<runmat_accelerate_api::GpuTensorHandle>,
    host_cast_output: Option<crate::context::preparation::PreparedHostNumericalOutput>,
) -> BuiltinResult<RootedValue> {
    template.validate_input(&value)?;
    // Services may transform the operand; protect both originals and the
    // actual prepared resident payload, and re-admit through the domain body.
    if let Value::GpuTensor(handle) = &value {
        borrowed.push(handle.clone());
    }
    let protected: Vec<_> = borrowed.iter().collect();
    let _borrowed_metadata: Vec<_> = protected
        .iter()
        .map(|handle| crate::builtins::common::gpu_helpers::ResidencyRestoreGuard::new(handle))
        .collect();
    let converted = match value {
        Value::GpuTensor(handle) => single_from_gpu(handle, &protected).await?,
        host => host::convert(runtime, host, host_cast_output)?,
    };
    let intermediate = match &converted {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let mut template_protected = protected.clone();
    if let Some(handle) = &intermediate {
        template_protected.push(handle);
    }
    let result = apply_output_template(converted, &template, &template_protected).await?;
    RootedValue::new(result).map_err(|cause| cause.into_runtime_error())
}
