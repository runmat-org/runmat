//! MATLAB-compatible `cos` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, COS_CHARACTER_INPUT_EXTENSION, COS_ERROR_ARG_COUNT,
    COS_ERROR_GPU_UNAVAILABLE, COS_ERROR_INTERNAL, COS_ERROR_INVALID_INPUT,
    COS_ERROR_INVALID_OPTION, COS_ERROR_LIKE_PROTOTYPE, COS_INTEGER_INPUT_EXTENSION,
    COS_LIKE_OUTPUT_EXTENSION, COS_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{COS_DESCRIPTOR, COS_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::NumericDType;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::symbolic::symbolic_function;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::SymbolicFunction;

const BUILTIN_NAME: &str = "cos";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cos",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_cos" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute cosine directly on device; runtimes gather to host when unary_cos is unavailable.",
};

fn cos_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cos_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::cos")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cos",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("cos({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `cos` calls; providers can override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "cos",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::cos"
)]
async fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    ensure_cos_extensions(&value, &rest).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "cos")?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Cos) {
        return apply_output_template(symbolic, &template).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => cos_gpu(handle).await?,
        Value::Complex(re, im) => Value::Complex(cos_complex_re(re, im), cos_complex_im(re, im)),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct)?,
        Value::CharArray(ca) => cos_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(cos_error_with_detail(
                &COS_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => cos_real(other)?,
    };
    apply_output_template(base, &template).await
}

async fn ensure_cos_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &COS_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COS_LIKE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn cos_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle);
    let requires_exact_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_exact_host_path {
        if let Some(provider) = provider {
            match provider.unary_cos(&handle).await {
                Ok(output)
                    if gpu_helpers::unary_gpu_output_matches(
                        &output,
                        &handle,
                        provider,
                        gpu_helpers::UnaryGpuOutputContract {
                            storage: runmat_accelerate_api::handle_storage(&handle),
                            precision: runmat_accelerate_api::handle_precision(&handle),
                            integer: None,
                            logical: false,
                            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
                        },
                    ) =>
                {
                    return Ok(gpu_helpers::resident_gpu_value(output));
                }
                Ok(output) => {
                    gpu_helpers::free_rejected_provider_output(&output, &[&handle], provider);
                    return Err(cos_error_with_detail(
                        &COS_ERROR_INTERNAL,
                        "provider returned an invalid unary cosine result",
                    ));
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(cos_error_with_detail(
                        &COS_ERROR_INTERNAL,
                        format!("provider unary cosine failed: {error}"),
                    ));
                }
            }
        }
    }
    let source = handle.clone();
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let host = match gathered {
        Value::Complex(re, im) => Ok(Value::Complex(
            cos_complex_re(re, im),
            cos_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => cos_complex_tensor(ct),
        Value::Tensor(tensor) => cos_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.cos())),
        other => cos_real(other),
    }?;
    gpu_helpers::restore_class_preserving_value(&source, host, BUILTIN_NAME)
}

fn cos_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("cos", value)
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INVALID_INPUT, e))?;
    cos_tensor(tensor).map(tensor::tensor_into_value)
}

fn cos_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.cos())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.cos())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))
}

fn cos_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    (
                        cos_complex_re(f64::from(re), f64::from(im)) as f32,
                        cos_complex_im(f64::from(re), f64::from(im)) as f32,
                    )
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| (cos_complex_re(re, im), cos_complex_im(re, im)))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn cos_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).cos())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn cos_complex_re(re: f64, im: f64) -> f64 {
    re.cos() * im.cosh()
}

#[inline]
fn cos_complex_im(re: f64, im: f64) -> f64 {
    -re.sin() * im.sinh()
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(Value),
}

fn parse_output_template(args: &[Value]) -> BuiltinResult<OutputTemplate> {
    match args.len() {
        0 => Ok(OutputTemplate::Default),
        1 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "unrecognised argument for cos",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(cos_error_with_detail(
                    &COS_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(cos_error(&COS_ERROR_ARG_COUNT)),
    }
}

async fn apply_output_template(value: Value, template: &OutputTemplate) -> BuiltinResult<Value> {
    match template {
        OutputTemplate::Default => Ok(value),
        OutputTemplate::Like(proto) => match proto {
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                {
                    convert_to_gpu_complex(value, handle).await
                } else {
                    convert_to_gpu(value, handle).await
                }
            }
            Value::Tensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => convert_to_host_like(value).await,
            Value::Complex(_, _) | Value::ComplexTensor(_) => convert_to_host_complex(value).await,
            _ => Err(cos_error_with_detail(
                &COS_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        cos_error_with_detail(
            &COS_ERROR_GPU_UNAVAILABLE,
            "GPU output requested via 'like' but no provider owns the prototype",
        )
    })?;
    match value {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
                && gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider)) =>
        {
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_gpu(gathered, prototype).await
        }
        Value::Tensor(tensor) => {
            let handle = gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype).await
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64()), prototype).await,
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 }), prototype).await,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(cos_error_with_detail(
            &COS_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        cos_error_with_detail(
            &COS_ERROR_GPU_UNAVAILABLE,
            "complex GPU output requested via 'like' but no provider owns the prototype",
        )
    })?;
    match value {
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(&handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                && gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider)) =>
        {
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::GpuTensor(handle) => {
            let same_owner = gpu_helpers::exact_provider_for_handle(&handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider));
            if same_owner
                && runmat_accelerate_api::handle_storage(&handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
            {
                match provider.complex_from_real(&handle).await {
                    Ok(output)
                        if gpu_helpers::unary_gpu_output_matches(
                            &output,
                            &handle,
                            provider,
                            gpu_helpers::UnaryGpuOutputContract {
                                storage:
                                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                                precision: runmat_accelerate_api::handle_precision(&handle),
                                integer: None,
                                logical: false,
                                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                            },
                        ) =>
                    {
                        return Ok(gpu_helpers::complex_gpu_value(output));
                    }
                    Ok(output) => {
                        gpu_helpers::free_rejected_provider_output(&output, &[&handle], provider);
                        return Err(cos_error_with_detail(
                            &COS_ERROR_INTERNAL,
                            "provider returned an invalid complex conversion result",
                        ));
                    }
                    Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                    Err(error) => {
                        return Err(cos_error_with_detail(
                            &COS_ERROR_INTERNAL,
                            format!("provider complex conversion failed: {error}"),
                        ));
                    }
                }
            }
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_gpu_complex(gathered, prototype).await
        }
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::Num(n) => convert_to_gpu_complex(Value::Complex(n, 0.0), prototype).await,
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::from_f64_values_with_dtype(
                data,
                tensor.shape.clone(),
                complex_floating_output_dtype(tensor.numeric_dtype()),
            )
            .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex), prototype).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor), prototype).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64()), prototype).await,
        Value::Bool(b) => {
            convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 }), prototype).await
        }
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex GPU output via 'like'"),
        )),
    }
}

async fn convert_to_host_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let proxy = Value::GpuTensor(handle);
            gpu_helpers::gather_value_async(&proxy).await
        }
        other => Ok(other),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_host_complex(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => Ok(value),
        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
        Value::Tensor(tensor) => {
            let values = tensor::tensor_values_f64_cow(&tensor);
            let data = values.iter().map(|&re| (re, 0.0)).collect::<Vec<_>>();
            let complex = ComplexTensor::from_f64_values_with_dtype(
                data,
                tensor.shape.clone(),
                complex_floating_output_dtype(tensor.numeric_dtype()),
            )
            .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            Ok(complex_tensor_into_value(complex))
        }
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            convert_to_host_complex(gathered).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| cos_error_with_detail(&COS_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(cos_error_with_detail(
            &COS_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex output via 'like'"),
        )),
    }
}

fn complex_floating_output_dtype(dtype: runmat_value::NumericDType) -> runmat_value::NumericDType {
    if dtype == runmat_value::NumericDType::F32 {
        runmat_value::NumericDType::F32
    } else {
        runmat_value::NumericDType::F64
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorOwned,
        HostTensorView, ProviderPrecision,
    };
    use runmat_value::{IntValue, StringArray, Tensor};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use crate::builtins::common::{gpu_helpers, test_support};

    fn cos_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::cos_builtin(value, rest))
    }

    struct MalformedCosProvider {
        device_id: u32,
        next_buffer: AtomicU64,
        malformed: AtomicU8,
        allocations: AtomicUsize,
        frees: AtomicUsize,
        buffers: Mutex<HashMap<u64, HostTensorOwned>>,
    }

    impl MalformedCosProvider {
        fn new() -> Self {
            Self {
                device_id: runmat_accelerate_api::next_device_id(),
                next_buffer: AtomicU64::new(8_700_000_000_000_000_000),
                malformed: AtomicU8::new(0),
                allocations: AtomicUsize::new(0),
                frees: AtomicUsize::new(0),
                buffers: Mutex::new(HashMap::new()),
            }
        }

        fn allocate(
            &self,
            data: Vec<f64>,
            shape: Vec<usize>,
            device_id: u32,
            precision: ProviderPrecision,
            storage: GpuTensorStorage,
        ) -> GpuTensorHandle {
            let buffer_id = self.next_buffer.fetch_add(1, Ordering::Relaxed);
            self.buffers.lock().unwrap().insert(
                buffer_id,
                HostTensorOwned {
                    data,
                    shape: shape.clone(),
                    storage,
                },
            );
            self.allocations.fetch_add(1, Ordering::Relaxed);
            GpuTensorHandle {
                shape,
                device_id,
                buffer_id,
                descriptor: runmat_accelerate_api::GpuTensorDescriptor::numeric(
                    match precision {
                        ProviderPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
                        ProviderPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
                    },
                    storage,
                ),
            }
        }
    }

    impl AccelProvider for MalformedCosProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(self.allocate(
                host.data.to_vec(),
                host.shape.to_vec(),
                self.device_id,
                ProviderPrecision::F64,
                GpuTensorStorage::Real,
            ))
        }

        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async move {
                self.buffers
                    .lock()
                    .unwrap()
                    .get(&handle.buffer_id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown test buffer"))
            })
        }

        fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
            if self
                .buffers
                .lock()
                .unwrap()
                .remove(&handle.buffer_id)
                .is_some()
            {
                self.frees.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        }

        fn device_info(&self) -> String {
            "malformed-cos-test-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn unary_cos<'a>(
            &'a self,
            input: &'a GpuTensorHandle,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move {
                let malformed = self.malformed.load(Ordering::Relaxed);
                let device_id = if malformed == 2 {
                    self.device_id.wrapping_add(10_000)
                } else {
                    self.device_id
                };
                let precision = if malformed == 0 {
                    ProviderPrecision::F32
                } else {
                    ProviderPrecision::F64
                };
                let storage = if malformed == 1 {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                };
                Ok(self.allocate(
                    vec![99.0; input.shape.iter().product()],
                    input.shape.clone(),
                    device_id,
                    precision,
                    storage,
                ))
            })
        }
    }

    #[test]
    fn cos_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = COS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = cos(X)"));
        assert!(labels.contains(&"Y = cos(X, \"like\", P)"));
        assert_eq!(COS_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn cos_integer_gate_all_classes_boundary_and_single_precision() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::cos_builtin(Value::Int(IntValue::I8(1)), Vec::new()))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            COS_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(_strict);

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ] {
            assert!(block_on(super::cos_builtin(Value::Int(value), Vec::new())).is_ok());
        }
        assert!(block_on(super::cos_builtin(
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
            Vec::new(),
        ))
        .is_err());
        assert!(block_on(super::cos_builtin(
            Value::Int(IntValue::U64(1_u64 << 54)),
            Vec::new(),
        ))
        .is_ok());

        let single = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(single_out) =
            block_on(super::cos_builtin(Value::Tensor(single), Vec::new())).unwrap()
        else {
            panic!("expected single tensor")
        };
        assert_eq!(single_out.numeric_dtype(), NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(complex_out) = block_on(super::cos_builtin(
            Value::ComplexTensor(complex),
            Vec::new(),
        ))
        .unwrap() else {
            panic!("expected single complex tensor")
        };
        assert_eq!(complex_out.numeric_dtype(), NumericDType::F32);
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_scalar_zero() {
        let result = cos_builtin(Value::Num(0.0), Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] + 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        let result = cos_builtin(Value::Tensor(tensor), Vec::new()).expect("cos");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [1.0, 1.0f64.cos(), 2.0f64.cos()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn cos_host_complex_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![-3, 0, 5]),
            vec![3, 1],
        )
        .expect("integer tensor");

        let result =
            block_on(convert_to_host_complex(Value::Tensor(tensor))).expect("complex conversion");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor result");
        };
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(
            out.materialize_f64(),
            vec![(-3.0, 0.0), (0.0, 0.0), (5.0, 0.0)]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_int_value_promotes() {
        let value = Value::Int(IntValue::I32(1));
        let result = cos_builtin(value, Vec::new()).expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0f64.cos()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_complex_scalar() {
        let result = cos_builtin(Value::Complex(1.0, 2.0), Vec::new()).expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.cos() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im + (1.0f64.sin() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_char_array_roundtrip() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = cos_builtin(Value::CharArray(chars), Vec::new()).expect("cos");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).cos();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[test]
    fn cos_rejects_and_frees_malformed_native_outputs() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(MalformedCosProvider::new()));
        unsafe {
            runmat_accelerate_api::register_provider(provider);
        }

        for malformed in 0..3_u8 {
            provider.malformed.store(malformed, Ordering::Relaxed);
            let input = provider
                .upload(&HostTensorView {
                    data: &[0.0, 1.0],
                    shape: &[2, 1],
                })
                .expect("input upload");
            let error = block_on(super::cos_gpu(input.clone()))
                .expect_err("malformed native output must be rejected");
            assert_eq!(error.identifier(), COS_ERROR_INTERNAL.identifier);
            let completed = usize::from(malformed) + 1;
            assert_eq!(provider.allocations.load(Ordering::Relaxed), completed * 2);
            assert_eq!(provider.frees.load(Ordering::Relaxed), completed * 2 - 1);
            provider.free(&input).expect("free input");
            assert_eq!(provider.frees.load(Ordering::Relaxed), completed * 2);
        }
        assert_eq!(
            provider.allocations.load(Ordering::Relaxed),
            provider.frees.load(Ordering::Relaxed)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_missing_prototype_errors() {
        let err =
            cos_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), COS_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[test]
    fn cos_validates_like_syntax_before_extension_gate() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(super::cos_builtin(
            Value::Int(IntValue::U8(1)),
            vec![Value::from("like")],
        ))
        .expect_err("malformed like syntax");
        assert_eq!(error.identifier(), COS_ERROR_INVALID_OPTION.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_complex_prototype_returns_complex() {
        let result = cos_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("cos");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.cos()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn cos_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = cos_builtin(Value::GpuTensor(handle), Vec::new()).expect("cos");
            let out = match result {
                Value::GpuTensor(handle) => {
                    assert_eq!(
                        runmat_accelerate_api::handle_storage(&handle),
                        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
                    );
                    match block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                        .expect("gather")
                    {
                        Value::ComplexTensor(out) => out,
                        other => panic!("expected gathered complex tensor, got {other:?}"),
                    }
                }
                Value::ComplexTensor(out) => out,
                other => panic!("expected complex output, got {other:?}"),
            };
            assert_eq!(out.shape, vec![1, 2]);
            for (idx, &(re, im)) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - cos_complex_re(re, im)).abs() < 1e-12);
                assert!((out.materialize_f64()[idx].1 - cos_complex_im(re, im)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_uploads_complex_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape, vec![2, 1]);
            for (idx, &re) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn cos_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let input_view = HostTensorView {
                data: &input.materialize_f64(),
                shape: &input.shape,
            };
            let input_handle = provider.upload(&input_view).expect("upload input");
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = cos_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("cos");
            let Value::GpuTensor(handle) = result else {
                panic!("expected complex gpu tensor, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered =
                block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
            let Value::ComplexTensor(out) = gathered else {
                panic!("expected gathered complex tensor, got {gathered:?}");
            };
            assert_eq!(out.shape, vec![2, 1]);
            for (idx, &re) in input.materialize_f64().iter().enumerate() {
                assert!((out.materialize_f64()[idx].0 - re.cos()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = cos_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("cos");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn cos_like_gpu_prototype_controls_owner_across_providers() {
        let _guard = test_support::accel_test_lock();
        let input_provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        let prototype_provider = Box::leak(Box::new(
            runmat_accelerate::simple_provider::InProcessProvider::new(),
        ));
        unsafe {
            runmat_accelerate_api::register_provider(input_provider);
            runmat_accelerate_api::register_provider(prototype_provider);
        }
        let input = input_provider
            .upload(&HostTensorView {
                data: &[0.0, 1.0],
                shape: &[2, 1],
            })
            .expect("upload input");
        let prototype = prototype_provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .expect("upload prototype");

        let result = cos_builtin(
            Value::GpuTensor(input),
            vec![Value::from("like"), Value::GpuTensor(prototype)],
        )
        .expect("mixed-provider like conversion");
        let Value::GpuTensor(output) = result else {
            panic!("expected GPU output")
        };
        assert_eq!(output.device_id, prototype_provider.device_id());
        let output_owner =
            runmat_accelerate_api::provider_for_handle(&output).expect("registered output owner");
        assert!(std::ptr::eq(
            output_owner,
            prototype_provider as &dyn runmat_accelerate_api::AccelProvider
        ));
        let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
        assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0_f64.cos()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_host_with_gpu_input_gathers() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = cos_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("cos");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.materialize_f64(), expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                Value::Num(_) => panic!("expected vector output"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_rejects_extra_arguments() {
        let err = cos_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_keyword_case_insensitive() {
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = cos_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.cos()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_char_array_keyword() {
        let keyword = CharArray::new_row("like");
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_like_string_array_keyword() {
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = cos_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("cos");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cos_unrecognised_option_errors() {
        let err =
            cos_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cos_wgpu_matches_cpu_elementwise() {
        let _guard = test_support::accel_test_lock();
        let Some(provider) = test_support::wgpu_provider_if_available() else {
            return;
        };
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = cos_real(Value::Tensor(t.clone())).unwrap();
        let view = HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = provider.upload(&view).unwrap();
        let gpu = block_on(cos_gpu(h)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match provider.precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
                }
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
