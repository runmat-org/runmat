//! MATLAB-compatible `tan` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinErrorDescriptor, TAN_CHARACTER_INPUT_EXTENSION, TAN_ERROR_ARG_COUNT,
    TAN_ERROR_GPU_UNAVAILABLE, TAN_ERROR_INTERNAL, TAN_ERROR_INVALID_INPUT,
    TAN_ERROR_INVALID_OPTION, TAN_ERROR_LIKE_PROTOTYPE, TAN_INTEGER_INPUT_EXTENSION,
    TAN_LIKE_OUTPUT_EXTENSION, TAN_LOGICAL_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{TAN_DESCRIPTOR, TAN_INTEGER_CAPABILITIES};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

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

const BUILTIN_NAME: &str = "tan";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::tan")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "tan",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_tan" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute tan in place via unary_tan; runtimes gather to host when the hook is unavailable.",
};

fn tan_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn tan_error_with_detail(
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::trigonometry::tan")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "tan",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!("tan({input})"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes:
        "Fusion planner emits WGSL tan calls; providers can override with optimised fused kernels.",
};

#[runtime_builtin(
    name = "tan",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::trigonometry::tan"
)]
async fn tan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let template = parse_output_template(&rest)?;
    ensure_tan_extensions(&value, &rest).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, "tan")?;
    if let Some(symbolic) = symbolic_function(&value, SymbolicFunction::Tan) {
        return apply_output_template(symbolic, &template).await;
    }
    let base = match value {
        Value::GpuTensor(handle) => tan_gpu(handle).await?,
        Value::Complex(re, im) => {
            let (out_re, out_im) = tan_complex_components(re, im);
            Value::Complex(out_re, out_im)
        }
        Value::ComplexTensor(ct) => tan_complex_tensor(ct)?,
        Value::CharArray(ca) => tan_char_array(ca)?,
        Value::String(_) | Value::StringArray(_) => {
            return Err(tan_error_with_detail(
                &TAN_ERROR_INVALID_INPUT,
                "expected numeric input, got string",
            ))
        }
        other => tan_real(other)?,
    };
    apply_output_template(base, &template).await
}

async fn ensure_tan_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &TAN_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "X",
    )
    .await?;
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TAN_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::CharArray(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TAN_CHARACTER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !rest.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &TAN_LIKE_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn tan_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle);
    let requires_exact_host_path = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle);
    if !requires_exact_host_path {
        if let Some(provider) = provider {
            match provider.unary_tan(&handle).await {
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
                    return Err(tan_error_with_detail(
                        &TAN_ERROR_INTERNAL,
                        "provider returned an invalid unary tangent result",
                    ));
                }
                Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                Err(error) => {
                    return Err(tan_error_with_detail(
                        &TAN_ERROR_INTERNAL,
                        format!("provider unary tangent failed: {error}"),
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
        Value::Complex(re, im) => {
            let (out_re, out_im) = tan_complex_components(re, im);
            Ok(Value::Complex(out_re, out_im))
        }
        Value::ComplexTensor(ct) => tan_complex_tensor(ct),
        Value::Tensor(tensor) => tan_tensor(tensor).map(tensor::tensor_into_value),
        Value::Num(n) => Ok(Value::Num(n.tan())),
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INVALID_INPUT,
            format!("unsupported gathered gpuArray value {other:?}"),
        )),
    }?;
    gpu_helpers::restore_class_preserving_value(&source, host, BUILTIN_NAME)
}

fn tan_real(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for("tan", value)
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INVALID_INPUT, e))?;
    tan_tensor(tensor).map(tensor::tensor_into_value)
}

fn tan_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    if tensor.numeric_dtype() == NumericDType::F32 {
        let data = tensor
            .as_f32_slice()
            .expect("single tensor storage")
            .iter()
            .map(|&v| v.tan())
            .collect();
        return Tensor::from_f32(data, tensor.shape.clone())
            .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e));
    }
    let data = tensor::tensor_values_f64_cow(&tensor)
        .iter()
        .map(|&v| v.tan())
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone())
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))
}

fn tan_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let tensor = match ct.into_complex_storage() {
        ComplexStorage::F32(values) => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(re, im)| {
                    let (out_re, out_im) = tan_complex_components(f64::from(re), f64::from(im));
                    (out_re as f32, out_im as f32)
                })
                .collect(),
            shape,
        ),
        ComplexStorage::F64(values) => ComplexTensor::new(
            values
                .into_iter()
                .map(|(re, im)| tan_complex_components(re, im))
                .collect(),
            shape,
        ),
        ComplexStorage::Integer(_) => Err("typed complex integer input is unsupported".into()),
    }
    .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
    Ok(complex_tensor_into_value(tensor))
}

fn tan_char_array(array: CharArray) -> BuiltinResult<Value> {
    let data = array
        .data
        .iter()
        .map(|&ch| (ch as u32 as f64).tan())
        .collect::<Vec<_>>();
    let tensor = Tensor::new(data, vec![array.rows, array.cols])
        .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[inline]
fn tan_complex_components(re: f64, im: f64) -> (f64, f64) {
    let two_re = 2.0 * re;
    let two_im = 2.0 * im;
    let inv_cosh = 1.0 / two_im.cosh();
    let denom = 1.0 + two_re.cos() * inv_cosh;
    let real = (two_re.sin() * inv_cosh) / denom;
    let imag = two_im.tanh() / denom;
    (real, imag)
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
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "expected prototype after 'like'",
                ))
            } else {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "unrecognised argument for tan",
                ))
            }
        }
        2 => {
            if matches!(keyword_of(&args[0]).as_deref(), Some("like")) {
                Ok(OutputTemplate::Like(args[1].clone()))
            } else {
                Err(tan_error_with_detail(
                    &TAN_ERROR_INVALID_OPTION,
                    "unsupported option; only 'like' is accepted",
                ))
            }
        }
        _ => Err(tan_error(&TAN_ERROR_ARG_COUNT)),
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
            _ => Err(tan_error_with_detail(
                &TAN_ERROR_LIKE_PROTOTYPE,
                "unsupported prototype; provide a numeric or gpuArray prototype",
            )),
        },
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        tan_error_with_detail(
            &TAN_ERROR_GPU_UNAVAILABLE,
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
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::resident_gpu_value(handle))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1])
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype).await
        }
        Value::Int(i) => convert_to_gpu(Value::Num(i.to_f64()), prototype).await,
        Value::Bool(b) => convert_to_gpu(Value::Num(if b { 1.0 } else { 0.0 }), prototype).await,
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu(Value::Tensor(tensor), prototype).await
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(tan_error_with_detail(
            &TAN_ERROR_LIKE_PROTOTYPE,
            "GPU prototypes for 'like' only support real numeric outputs",
        )),
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
            format!("unsupported result type for GPU output via 'like' ({other:?})"),
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_gpu_complex(value: Value, prototype: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(prototype).ok_or_else(|| {
        tan_error_with_detail(
            &TAN_ERROR_GPU_UNAVAILABLE,
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
                        return Err(tan_error_with_detail(
                            &TAN_ERROR_INTERNAL,
                            "provider returned an invalid complex conversion result",
                        ));
                    }
                    Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
                    Err(error) => {
                        return Err(tan_error_with_detail(
                            &TAN_ERROR_INTERNAL,
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
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            Ok(gpu_helpers::complex_gpu_value(handle))
        }
        Value::ComplexTensor(tensor) => {
            let handle = gpu_helpers::upload_complex_tensor(provider, &tensor)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
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
            .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::ComplexTensor(complex), prototype).await
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_gpu_complex(Value::Tensor(tensor), prototype).await
        }
        Value::Int(i) => convert_to_gpu_complex(Value::Num(i.to_f64()), prototype).await,
        Value::Bool(b) => {
            convert_to_gpu_complex(Value::Num(if b { 1.0 } else { 0.0 }), prototype).await
        }
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
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
            .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
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
                .map_err(|e| tan_error_with_detail(&TAN_ERROR_INTERNAL, e))?;
            convert_to_host_complex(Value::Tensor(tensor)).await
        }
        Value::Int(i) => convert_to_host_complex(Value::Num(i.to_f64())).await,
        Value::Bool(b) => convert_to_host_complex(Value::Num(if b { 1.0 } else { 0.0 })).await,
        other => Err(tan_error_with_detail(
            &TAN_ERROR_INTERNAL,
            format!("cannot convert value {other:?} to complex output via 'like'"),
        )),
    }
}

fn complex_floating_output_dtype(dtype: NumericDType) -> NumericDType {
    if dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{CharArray, IntValue, StringArray, Tensor};

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    #[test]
    fn tan_descriptor_signatures_cover_like_overload() {
        let labels: Vec<&str> = TAN_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"Y = tan(X)"));
        assert!(labels.contains(&"Y = tan(X, \"like\", P)"));
        assert_eq!(TAN_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    }

    #[test]
    fn tan_integer_gate_all_classes_boundary_and_single_precision() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = block_on(super::tan_builtin(Value::Int(IntValue::I8(1)), Vec::new()))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            err.identifier(),
            TAN_INTEGER_INPUT_EXTENSION.error_identifier
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
            assert!(block_on(super::tan_builtin(Value::Int(value), Vec::new())).is_ok());
        }
        assert!(block_on(super::tan_builtin(
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
            Vec::new(),
        ))
        .is_err());
        assert!(block_on(super::tan_builtin(
            Value::Int(IntValue::U64(1_u64 << 54)),
            Vec::new(),
        ))
        .is_ok());

        let single = Tensor::from_f32(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let Value::Tensor(single_out) =
            block_on(super::tan_builtin(Value::Tensor(single), Vec::new())).unwrap()
        else {
            panic!("expected single tensor")
        };
        assert_eq!(single_out.numeric_dtype(), NumericDType::F32);
        let complex = ComplexTensor::from_f32(vec![(1.0, 0.5)], vec![1, 1]).unwrap();
        let Value::ComplexTensor(complex_out) = block_on(super::tan_builtin(
            Value::ComplexTensor(complex),
            Vec::new(),
        ))
        .unwrap() else {
            panic!("expected single complex tensor")
        };
        assert_eq!(complex_out.numeric_dtype(), NumericDType::F32);
    }

    fn tan_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        block_on(super::tan_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_scalar_pi_over_four() {
        let result = tan_builtin(Value::Num(std::f64::consts::FRAC_PI_4), Vec::new()).expect("tan");
        match result {
            Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_tensor_elements() {
        let tensor = Tensor::new(vec![0.0, std::f64::consts::FRAC_PI_4], vec![2, 1]).unwrap();
        let result = tan_builtin(Value::Tensor(tensor), Vec::new()).expect("tan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert!((out.materialize_f64()[0] - 0.0).abs() < 1e-12);
                assert!((out.materialize_f64()[1] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_reads_typed_integer_tensor_storage_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor =
            Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![0, 1, 2]), vec![3, 1])
                .expect("integer tensor");

        let result = tan_builtin(Value::Tensor(tensor), Vec::new()).expect("tan");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 1]);
                let expected = [0.0, 1.0f64.tan(), 2.0f64.tan()];
                for (actual, expected) in out.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                assert!(out.integer_storage().is_none());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn tan_host_complex_conversion_reads_typed_integer_storage_exactly() {
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
    fn tan_string_input_errors() {
        let err = tan_builtin(Value::from("invalid"), Vec::new()).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_int_promotes() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = tan_builtin(Value::Int(IntValue::I32(1)), Vec::new()).expect("tan");
        match result {
            Value::Num(v) => assert!((v - 1f64.tan()).abs() < 1e-12),
            other => panic!("expected numeric result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_complex_scalar_matches_formula() {
        let result = tan_builtin(Value::Complex(1.0, 0.5), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                let (expected_re, expected_im) = tan_complex_components(1.0, 0.5);
                assert!((re - expected_re).abs() < 1e-12);
                assert!((im - expected_im).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_complex_on_real_axis_matches_real_value() {
        let angle = std::f64::consts::FRAC_PI_2 * 0.9;
        let result = tan_builtin(Value::Complex(angle, 0.0), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                assert!((re - angle.tan()).abs() < 1e-12);
                assert_eq!(im, 0.0);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_char_array_roundtrip() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let result = tan_builtin(Value::CharArray(chars), Vec::new()).expect("tan");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                let expected: Vec<f64> = ['A', 'B']
                    .iter()
                    .map(|&ch| (ch as u32 as f64).tan())
                    .collect();
                for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((got - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.2, -0.3, 1.0], vec![4, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tan_builtin(Value::GpuTensor(handle), Vec::new()).expect("tan");
            let gathered = test_support::gather(result).expect("gather");
            let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.tan()).collect();
            assert_eq!(gathered.shape, vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), expected);
        });
    }

    #[test]
    fn tan_gpu_fallback_preserves_single_and_source_owner() {
        test_support::with_f32_test_provider(|provider| {
            let input = [0.0, 0.5, 1.0];
            let source = provider
                .upload(&HostTensorView {
                    data: &input,
                    shape: &[3, 1],
                })
                .expect("upload");
            let source_device = source.device_id;
            let result = block_on(super::tan_builtin(Value::GpuTensor(source), Vec::new()))
                .expect("tan fallback");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident result")
            };
            assert_eq!(handle.device_id, source_device);
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.shape, vec![3, 1]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_missing_prototype_errors() {
        let err =
            tan_builtin(Value::Num(1.0), vec![Value::from("like")]).expect_err("expected error");
        assert_eq!(err.identifier(), TAN_ERROR_INVALID_OPTION.identifier);
        let message = error_message(err);
        assert!(message.contains("prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_complex_prototype_returns_complex() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = tan_builtin(
            Value::Num(1.0),
            vec![Value::from("like"), Value::Complex(0.0, 1.0)],
        )
        .expect("tan");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 1.0_f64.tan()).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn tan_complex_large_imag_is_stable() {
        let result = tan_builtin(Value::Complex(0.0, 400.0), Vec::new()).expect("tan");
        match result {
            Value::Complex(re, im) => {
                assert!(re.abs() < 1e-12);
                assert!((im - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn tan_gpu_complex_input_preserves_complex_output() {
        test_support::with_test_provider(|provider| {
            let input = ComplexTensor::new(vec![(0.5, 0.75), (2.0, -0.25)], vec![1, 2]).unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &input).expect("upload");
            let result = tan_builtin(Value::GpuTensor(handle), Vec::new()).expect("tan");
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
                let (expected_re, expected_im) = tan_complex_components(re, im);
                assert!((out.materialize_f64()[idx].0 - expected_re).abs() < 1e-12);
                assert!((out.materialize_f64()[idx].1 - expected_im).abs() < 1e-12);
            }
        });
    }

    #[test]
    fn tan_like_complex_gpu_prototype_uploads_complex_result() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 0.5], vec![2, 1]).unwrap();
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = tan_builtin(
                Value::Tensor(input.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("tan");
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
                assert!((out.materialize_f64()[idx].0 - re.tan()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[test]
    fn tan_like_complex_gpu_prototype_converts_resident_real_gpu_result() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let input = Tensor::new(vec![0.0, 0.5], vec![2, 1]).unwrap();
            let input_view = HostTensorView {
                data: &input.materialize_f64(),
                shape: &input.shape,
            };
            let input_handle = provider.upload(&input_view).expect("upload input");
            let proto_tensor = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto = gpu_helpers::upload_complex_tensor(provider, &proto_tensor)
                .expect("upload complex prototype");
            let result = tan_builtin(
                Value::GpuTensor(input_handle),
                vec![Value::from("like"), Value::GpuTensor(proto)],
            )
            .expect("tan");
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
                assert!((out.materialize_f64()[idx].0 - re.tan()).abs() < 1e-12);
                assert!(out.materialize_f64()[idx].1.abs() < 1e-12);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_gpu_prototype() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.3, 0.6], vec![3, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            };
            let proto = provider.upload(&proto_view).expect("upload");
            let result = tan_builtin(
                Value::Tensor(tensor.clone()),
                vec![Value::from("like"), Value::GpuTensor(proto.clone())],
            )
            .expect("tan");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.tan()).collect();
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.materialize_f64(), expected);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_host_with_gpu_input_gathers() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.5], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = tan_builtin(
                Value::GpuTensor(handle),
                vec![Value::from("like"), Value::Num(0.0)],
            )
            .expect("tan");
            match result {
                Value::Tensor(t) => {
                    let expected: Vec<f64> =
                        tensor.materialize_f64().iter().map(|&v| v.tan()).collect();
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.materialize_f64(), expected);
                }
                Value::GpuTensor(_) => panic!("expected host result"),
                other => panic!("unexpected result {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_rejects_extra_arguments() {
        let err = tan_builtin(
            Value::Num(0.0),
            vec![Value::from("like"), Value::Num(0.0), Value::Num(1.0)],
        )
        .expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("too many input arguments"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_keyword_case_insensitive() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![0.0, 0.1], vec![2, 1]).unwrap();
        let result = tan_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::from("LIKE"), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Tensor(out) => {
                let expected: Vec<f64> =
                    tensor.materialize_f64().iter().map(|&v| v.tan()).collect();
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.materialize_f64(), expected);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_char_array_keyword() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let keyword = CharArray::new_row("like");
        let result = tan_builtin(
            Value::Num(0.0),
            vec![Value::CharArray(keyword), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_like_string_array_keyword() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let keyword = StringArray::new(vec!["LIKE".to_string()], vec![1]).unwrap();
        let result = tan_builtin(
            Value::Num(0.0),
            vec![Value::StringArray(keyword), Value::Num(0.0)],
        )
        .expect("tan");
        match result {
            Value::Num(v) => assert!((v - 0.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tan_unrecognised_option_errors() {
        let err =
            tan_builtin(Value::Num(0.0), vec![Value::from("invalid")]).expect_err("expected error");
        let message = error_message(err);
        assert!(message.contains("unrecognised argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn tan_wgpu_matches_cpu_elementwise() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 0.25, -0.5, 1.0], vec![4, 1]).unwrap();
        let cpu = tan_real(Value::Tensor(tensor.clone())).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .unwrap();
        let gpu = block_on(tan_gpu(handle)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(gt.shape, ct.shape);
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                    assert!((a - b).abs() < tol, "|{a} - {b}| >= {tol}");
                }
            }
            _ => panic!("unexpected comparison result"),
        }
    }
}
