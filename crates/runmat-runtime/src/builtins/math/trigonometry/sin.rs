//! MATLAB-compatible `sin` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::SIN_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, SIN_CHARACTER_INPUT_EXTENSION, SIN_ERROR_ARG_COUNT,
    SIN_ERROR_GPU_UNAVAILABLE, SIN_ERROR_INTERNAL, SIN_ERROR_INVALID_INPUT,
    SIN_ERROR_INVALID_OPTION, SIN_ERROR_LIKE_PROTOTYPE, SIN_INTEGER_INPUT_EXTENSION,
    SIN_LIKE_OUTPUT_EXTENSION, SIN_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
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

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::sin")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sin",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_sin" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may execute sin in-place on the device; runtimes gather to host when unary_sin is unavailable.",
};

const BUILTIN_NAME: &str = "sin";

fn sin_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sin_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::sin")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sin",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("sin({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner emits WGSL `sin` calls; providers may override via fused elementwise kernels.",
};

#[runtime_builtin(
    name = "sin",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::sin"
)]
async fn sin_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let output = parse_output_template(&rest)?;
    ensure_sin_extensions(&value, &rest).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "sin")?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Sin) {
        return apply_output_template(symbolic, &output).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => sin_gpu(handle).await?,
        Value::Complex(re, im) => Value::Complex(sin_complex_re(re, im), sin_complex_im(re, im)),
        Value::ComplexTensor(ct) => sin_complex_tensor(ct)?,
        Value::CharArray(ca) => sin_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(sin_error_with_detail(
                &SIN_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => sin_real(other)?,
    };
    apply_output_template(base, &output).await
}

async fn ensure_sin_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &SIN_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIN_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIN_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SIN_LIKE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn sin_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle);
    let requires_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_host_path {
        if let Some(provider) = provider {
            match provider.unary_sin(&handle).await {
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
                    gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
                    return Err(sin_error_with_detail(
                        &SIN_ERROR_INTERNAL,
                        "provider returned an invalid unary sine result",
                    ));
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(sin_error_with_detail(
                        &SIN_ERROR_INTERNAL,
                        format!("provider unary sine failed: {error}"),
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
            sin_complex_re(re, im),
            sin_complex_im(re, im),
        )),
        Value::ComplexTensor(ct) => sin_complex_tensor(ct),
        Value::Tensor(tensor) => sin_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.sin())),
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INVALID_INPUT,
            format!("unsupported gathered gpuArray value {other:?}"),
        )),
    }?;
    gpu_helpers::restore_class_preserving_value(&source, host, BUILTIN_NAME)
}

fn sin_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("sin", value)
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INVALID_INPUT, e))?;
    sin_tensor(tensor).map(tensor::tensor_into_value)
}

fn sin_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == runmat_value::NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&value| value.sin())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|error| sin_error_with_detail(&SIN_ERROR_INTERNAL, error));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.sin())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))
}

fn sin_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|element| {
                    let (re, im) = element.into();
                    (
                        sin_complex_re(f64::from(re), f64::from(im)) as f32,
                        sin_complex_im(f64::from(re), f64::from(im)) as f32,
                    )
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|element| {
                    let (re, im) = element.into();
                    (sin_complex_re(re, im), sin_complex_im(re, im))
                })
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn sin_char_array(ca: CharArray) -> BuiltinResult<Value> {
    let data = ca
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).sin())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![ca.rows, ca.cols])
        .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn sin_complex_re(re: f64, im: f64) -> f64 {
    re.sin() * im.cosh()
}

#[inline]
fn sin_complex_im(re: f64, im: f64) -> f64 {
    re.cos() * im.sinh()
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
                Err(sin_error_with_detail(
                    &SIN_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(sin_error_with_detail(
                    &SIN_ERROR_INVALID_OPTION,
                    "unrecognised argument for sin",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(sin_error_with_detail(
                    &SIN_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(sin_error(&SIN_ERROR_ARG_COUNT)),
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
            _ => Err(sin_error_with_detail(
                &SIN_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        sin_error_with_detail(
            &SIN_ERROR_GPU_UNAVAILABLE,
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
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype).await
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64()), prototype).await,
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 }), prototype).await,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(sin_error_with_detail(
            &SIN_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        sin_error_with_detail(
            &SIN_ERROR_GPU_UNAVAILABLE,
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
                        gpu_helpers::free_unprotected_exact_owner(&output, &[&handle]);
                        return Err(sin_error_with_detail(
                            &SIN_ERROR_INTERNAL,
                            "provider returned an invalid complex conversion result",
                        ));
                    }
                    Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                    Err(error) => {
                        return Err(sin_error_with_detail(
                            &SIN_ERROR_INTERNAL,
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
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
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
            .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex), prototype).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor), prototype).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64()), prototype).await,
        Value::Bool(b) => {
            convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 }), prototype).await
        }
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INTERNAL,
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
            .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
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
                .map_err(|e| sin_error_with_detail(&SIN_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(sin_error_with_detail(
            &SIN_ERROR_INTERNAL,
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
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, Tensor};

    use crate::builtins::common::{gpu_helpers, test_support};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn sin_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = SIN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = sin(X)"));
        assert!(labels.contains(&"Y = sin(X, \"like\", P)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_scalar() {
        let value = Value::Num(std::f64::consts::PI / 2.0);
        let result = block_on(sin_builtin(value, Vec::new())).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
        let result = block_on(sin_builtin(Value::Tensor(tensor), Vec::new())).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert!((t.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!((t.materialize_f64()[1] - 0.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_preserves_single_tensor_storage() {
        let tensor = Tensor::from_f32(vec![0.0, std::f32::consts::FRAC_PI_2], vec![2, 1])
            .expect("single tensor");
        let result = block_on(sin_builtin(Value::Tensor(tensor), Vec::new())).expect("sin");
        let Value::Tensor(output) = result else {
            panic!("expected tensor result, got {result:?}");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape, vec![2, 1]);
        assert_eq!(output.as_f32_slice(), Some([0.0, 1.0].as_slice()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        let result = block_on(sin_builtin(Value::Tensor(tensor), Vec::new())).expect("sin");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.sin(), 2.0f64.sin()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sin_host_complex_conversion_reads_typed_integer_storage_exactly() {
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
    fn sin_int_value_promotes() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = Value::Int(IntValue::I32(1));
        let result = block_on(sin_builtin(value, Vec::new())).expect("sin");
        match result {
            Value::Num(v) => assert!((v - 1.0_f64.sin()).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_complex_scalar() {
        let result = block_on(sin_builtin(Value::Complex(1.0, 2.0), Vec::new())).expect("sin");
        match result {
            Value::Complex(re, im) => {
                assert!((re - (1.0f64.sin() * 2.0f64.cosh())).abs() < 1e-12);
                assert!((im - (1.0f64.cos() * 2.0f64.sinh())).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_preserves_single_complex_tensor_storage() {
        let input = ComplexTensor::from_f32(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2])
            .expect("single complex tensor");
        let result =
            block_on(sin_builtin(Value::ComplexTensor(input.clone()), Vec::new())).expect("sin");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor result, got {result:?}");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape, vec![1, 2]);
        for (actual, input) in output.materialize_f64().iter().zip(input.materialize_f64()) {
            assert!((actual.0 - sin_complex_re(input.0, input.1)).abs() < 1e-6);
            assert!((actual.1 - sin_complex_im(input.0, input.1)).abs() < 1e-6);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_char_array_roundtrip() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let result = block_on(sin_builtin(Value::CharArray(chars), Vec::new())).expect("sin");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                for (idx, ch) in ['a', 'b', 'c'].into_iter().enumerate() {
                    let expected = (ch as u32 as f64).sin();
                    assert!((t.materialize_f64()[idx] - expected).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(sin_builtin(Value::GpuTensor(handle), Vec::new())).expect("sin");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_missing_prototype_errors() {
        let err = block_on(sin_builtin(Value::Num(1.0), vec![Value::from("like")]))
            .expect_err("expected error");
        assert_eq!(err.identifier(), SIN_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_complex_prototype_returns_complex() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(sin_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        ))
        .expect("sin");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.sin()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_complex_prototype_preserves_single_computation_class() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Tensor::from_f32(vec![0.0, std::f32::consts::FRAC_PI_2], vec![1, 2])
            .expect("single tensor");
        let result = block_on(sin_builtin(
            Value::Tensor(input),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        ))
        .expect("sin");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor result, got {result:?}");
        };
        assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(output.shape, vec![1, 2]);
        let values = output.as_f32_slice().expect("single complex storage");
        assert_eq!(<(f32, f32)>::from(values[0]), (0.0, 0.0));
        assert_eq!(<(f32, f32)>::from(values[1]), (1.0, 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_gpu_prototype() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = block_on(sin_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            ))
            .expect("sin");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
                    assert_eq!(gathered.shape, vec![4, 1]);
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn sin_like_single_gpu_prototype_preserves_single_storage() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::from_f32(vec![0.0, std::f32::consts::FRAC_PI_2], vec![1, 2])
                .expect("single input");
            let prototype = Tensor::from_f32(vec![0.0], vec![1, 1]).expect("single prototype");
            let prototype = gpu_helpers::upload_tensor(provider, &prototype).expect("upload");
            let result = block_on(sin_builtin(
                Value::Tensor(input),
                vec![Value::from("like"), Value::GpuTensor(prototype)],
            ))
            .expect("sin");
            let Value::GpuTensor(output) = result else {
                panic!("expected provider-resident result, got {result:?}");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
                .expect("gather");
            let Value::Tensor(output) = gathered else {
                panic!("expected gathered tensor, got {gathered:?}");
            };
            assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
            assert_eq!(output.as_f32_slice(), Some([0.0, 1.0].as_slice()));
        });
    }

    #[test]
    fn sin_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = block_on(sin_builtin(Value::GpuTensor(handle), Vec::new())).expect("sin");
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
                assert!((out.materialize_f64()[idx].0 - sin_complex_re(re, im)).abs() < 1e-12);
                assert!((out.materialize_f64()[idx].1 - sin_complex_im(re, im)).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn sin_like_complex_gpu_prototype_uploads_complex_result() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = block_on(sin_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            ))
            .expect("sin");
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
                assert!((out.materialize_f64()[idx].0 - re.sin()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn sin_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            let result = block_on(sin_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            ))
            .expect("sin");
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
                assert!((out.materialize_f64()[idx].0 - re.sin()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_host_with_gpu_input_gathers() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(sin_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            ))
            .expect("sin");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
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
    fn sin_like_rejects_extra_arguments() {
        let err = block_on(sin_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        ))
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_keyword_case_insensitive() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let result = block_on(sin_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        ))
        .expect("sin");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.sin()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sin_like_char_array_keyword() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let keyword = CharArray::new_row("like");
        let result = block_on(sin_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        ))
        .expect("sin");
        match result {
            Value::Num(v) => assert!(v.abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sin_wgpu_matches_cpu_elementwise() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("WGPU provider");
        let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
        let cpu = sin_real(Value::Tensor(t.clone())).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &t.materialize_f64(),
            shape: &t.shape,
        };
        let h = provider.upload(&view).unwrap();
        let gpu = block_on(sin_gpu(h)).unwrap();
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

        let single = Tensor::from_f32(
            vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
            vec![3, 1],
        )
        .expect("single input");
        let prototype = Tensor::from_f32(vec![0.0], vec![1, 1]).expect("single prototype");
        let prototype = gpu_helpers::upload_tensor(provider, &prototype).expect("upload prototype");
        let result = block_on(sin_builtin(
            Value::Tensor(single),
            vec![Value::from("like"), Value::GpuTensor(prototype)],
        ))
        .expect("single WGPU sin");
        let Value::GpuTensor(result) = result else {
            panic!("expected resident single result, got {result:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&result),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(result)))
            .expect("gather single result");
        let Value::Tensor(gathered) = gathered else {
            panic!("expected gathered single tensor, got {gathered:?}");
        };
        assert_eq!(gathered.numeric_dtype(), runmat_value::NumericDType::F32);
        let values = gathered.as_f32_slice().expect("single storage");
        for (actual, expected) in values.iter().zip([0.0_f32, 1.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}
