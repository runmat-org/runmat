//! Compatibility at the real-input/possibly-complex result boundary.
use crate::builtins::common::{gpu_helpers, provider_restore};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::{Tensor, Value};

pub(in crate::builtins::math::trigonometry) fn adapter(
    builtin: &'static str,
    extension: &'static BuiltinExtensionDescriptor,
    internal_error: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> crate::builtins::common::real_unary_preparation::RealUnaryAdapter<
    impl crate::builtins::common::real_unary_preparation::BeforeHost,
    impl crate::builtins::common::real_unary_preparation::HostPublication,
    impl crate::builtins::common::real_unary_preparation::DevicePublication,
> {
    crate::builtins::common::real_unary_preparation::RealUnaryAdapter {
        internal_error,
        before_host: |_: &GpuTensorHandle, _: runmat_types::RealUnaryClassification| Ok(()),
        publish_host: move |provider: &'static dyn runmat_accelerate_api::AccelProvider,
                            input: &GpuTensorHandle,
                            output: Value| {
            publish(provider, input, builtin, extension, output)
        },
        publish_real: move |provider: &'static dyn runmat_accelerate_api::AccelProvider,
                            input: &GpuTensorHandle,
                            output: GpuTensorHandle| {
            provider_restore::validate_real_unary_provider_output(provider, input, output, builtin)
        },
    }
}

pub(in crate::builtins::math::trigonometry) async fn compute(
    handle: GpuTensorHandle,
    builtin: &str,
    extension: &'static BuiltinExtensionDescriptor,
    compute: impl FnOnce(Tensor) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .with_identifier(format!("RunMat:{builtin}:Internal"))
            .build()
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = compute(tensor)?;
    publish(provider, &handle, builtin, extension, output)
}

/// The selected compound runner supplies its already-computed host value to
/// this same publication owner; it must not gather or probe the source again.
pub(in crate::builtins::math::trigonometry) fn publish(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    builtin: &str,
    extension: &'static BuiltinExtensionDescriptor,
    output: Value,
) -> BuiltinResult<Value> {
    if matches!(output, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        // Automatic residency cannot introduce an explicit-gpuArray extension
        // restriction. Legacy unannotated handles retain their resident policy.
        if crate::builtins::common::real_unary_preparation::RealUnaryPublication::AutomaticComplexHost
            .publishes_host(handle, runmat_types::NumericDomain::Complex)
        {
            return Ok(output);
        }
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    provider_restore::upload_value_like(provider, output, builtin, handle)
}
