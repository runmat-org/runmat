//! MATLAB-compatible `gather` builtin with provider-aware semantics.

use crate::builtins::acceleration::gpu::type_resolvers::gather_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gather")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gather",
    op_kind: GpuOpKind::Custom("gather"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("download")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Downloads gpuArray handles via the provider's `download` hook without mutating the source handle; host inputs pass through unchanged.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::acceleration::gpu::gather")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gather",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion output sink and materialises a host copy without clearing source gpuArray residency.",
};

const BUILTIN_NAME: &str = "gather";

const GATHER_CONTAINER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "gather-recursive-container",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "recursively gathering gpuArray values nested in containers is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GatherRecursiveContainerExtension"),
};

pub const GATHER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GATHER_CONTAINER_EXTENSION];

const GATHER_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Host integer inputs pass through unchanged; integer gpuArray inputs download exact native storage without changing class or shape.",
    }];

pub const GATHER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[X1, X2, ...] = gather(integer_A1, integer_A2, ...)",
        inputs: &GATHER_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each result preserves its corresponding input's exact integer class and shape. Explicit gather creates a host copy and leaves the source gpuArray valid and resident.",
    }];

const GATHER_OUTPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host-resident value gathered from input.",
}];

const GATHER_OUTPUT_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Host-resident outputs matching each gathered input.",
}];

const GATHER_INPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to gather from GPU to host.",
}];

const GATHER_INPUT_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input value to gather.",
    },
    BuiltinParamDescriptor {
        name: "Xn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input values to gather.",
    },
];

const GATHER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = gather(X)",
        inputs: &GATHER_INPUT_SINGLE,
        outputs: &GATHER_OUTPUT_SINGLE,
    },
    BuiltinSignatureDescriptor {
        label: "[X1, X2, ...] = gather(X1, X2, ...)",
        inputs: &GATHER_INPUT_VARIADIC,
        outputs: &GATHER_OUTPUT_VARIADIC,
    },
];

const GATHER_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:gather:NotEnoughInputs"),
    when: "No input arguments were provided.",
    message: "gather: not enough input arguments",
};

const GATHER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gather:TooManyOutputs"),
    when: "Requested outputs exceed one for single-input gather.",
    message: "gather: too many output arguments",
};

const GATHER_ERROR_OUTPUT_COUNT_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.OUTPUT_COUNT_MISMATCH",
    identifier: Some("RunMat:gather:OutputCountMismatch"),
    when: "Requested output count does not match number of input arguments.",
    message: "gather: number of outputs must match number of inputs",
};

const GATHER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GATHER.INTERNAL",
    identifier: Some("RunMat:gather:InternalError"),
    when: "Internal output container construction failed.",
    message: "gather: internal error",
};

const GATHER_ERRORS: [BuiltinErrorDescriptor; 4] = [
    GATHER_ERROR_NOT_ENOUGH_INPUTS,
    GATHER_ERROR_TOO_MANY_OUTPUTS,
    GATHER_ERROR_OUTPUT_COUNT_MISMATCH,
    GATHER_ERROR_INTERNAL,
];

pub const GATHER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GATHER_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GATHER_ERRORS,
};

fn gather_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    gather_error_with_message(error.message, error)
}

fn gather_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "gather",
    category = "acceleration/gpu",
    summary = "Gather gpuArray data back to host memory.",
    keywords = "gather,gpuArray,accelerate,download",
    accel = "sink",
    type_resolver(gather_type),
    descriptor(crate::builtins::acceleration::gpu::gather::GATHER_DESCRIPTOR),
    extensions(crate::builtins::acceleration::gpu::gather::GATHER_EXTENSIONS),
    integer_capabilities(crate::builtins::acceleration::gpu::gather::GATHER_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::acceleration::gpu::gather"
)]
async fn gather_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    let len = eval.len();
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if len == 1 {
            if out_count > 1 {
                return Err(gather_error(&GATHER_ERROR_TOO_MANY_OUTPUTS).into());
            }
            return Ok(Value::OutputList(vec![eval.into_first()]));
        }
        if out_count != len {
            return Err(gather_error(&GATHER_ERROR_OUTPUT_COUNT_MISMATCH).into());
        }
        return Ok(Value::OutputList(eval.into_outputs()));
    }
    if len == 1 {
        Ok(eval.into_first())
    } else {
        Ok(Value::OutputList(eval.into_outputs()))
    }
}

/// Combined gather result used by single- and multi-output call sites.
#[derive(Debug, Clone)]
pub struct GatherResult {
    outputs: Vec<Value>,
}

impl GatherResult {
    fn new(outputs: Vec<Value>) -> Self {
        Self { outputs }
    }

    /// Number of gathered outputs.
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Borrowed slice of outputs (in call-order).
    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    /// Consume the result, yielding all outputs.
    pub fn into_outputs(self) -> Vec<Value> {
        self.outputs
    }

    /// Consume the result, yielding the first output (requires at least one input).
    pub fn into_first(self) -> Value {
        self.outputs
            .into_iter()
            .next()
            .expect("gather requires at least one input")
    }
}

/// Evaluate `gather` for arbitrary argument lists and return all outputs.
pub async fn evaluate(args: &[Value]) -> crate::BuiltinResult<GatherResult> {
    if args.is_empty() {
        return Err(gather_error(&GATHER_ERROR_NOT_ENOUGH_INPUTS).into());
    }
    let mut outputs = Vec::with_capacity(args.len());
    for value in args {
        outputs.push(gather_argument(value).await?);
    }
    Ok(GatherResult::new(outputs))
}

async fn gather_argument(value: &Value) -> crate::BuiltinResult<Value> {
    if !matches!(value, Value::GpuTensor(_)) && crate::dispatcher::value_contains_gpu(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &GATHER_CONTAINER_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    crate::dispatcher::gather_if_needed_async(value).await
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{CellArray, StructValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_passes_through_host_values() {
        let value = Value::Num(42.0);
        let result = block_on(gather_builtin(vec![value.clone()])).expect("gather");
        assert_eq!(result, value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_downloads_gpu_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::mark_handle_explicit(&handle);
            let result =
                block_on(gather_builtin(vec![Value::GpuTensor(handle.clone())])).expect("gather");
            match result {
                Value::Tensor(host) => {
                    assert_eq!(host.shape, tensor.shape);
                    assert_eq!(host.materialize_f64(), tensor.materialize_f64());
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
            assert!(runmat_accelerate_api::handle_is_explicit(&handle));
            block_on(gather_builtin(vec![Value::GpuTensor(handle.clone())]))
                .expect("source handle remains gatherable");
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_preserves_logical_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let data = vec![0.0, 1.0, 1.0, 0.0];
            let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let result = block_on(gather_builtin(vec![Value::GpuTensor(handle)])).expect("gather");
            match result {
                Value::LogicalArray(logical) => {
                    assert_eq!(logical.shape, vec![2, 2]);
                    assert_eq!(logical.data, vec![0, 1, 1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_recurses_into_cells() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell = CellArray::new(vec![Value::GpuTensor(handle), Value::from("host")], 1, 2)
                .expect("cell");
            let result = block_on(gather_builtin(vec![Value::Cell(cell)])).expect("gather");
            let Value::Cell(gathered) = result else {
                panic!("expected cell result");
            };
            let first = gathered.get(0, 0).expect("first element");
            match first {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.materialize_f64(), tensor.materialize_f64());
                }
                other => panic!("expected tensor in cell, got {other:?}"),
            }
            let second = gathered.get(0, 1).expect("second element");
            assert_eq!(second, Value::from("host"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_recurses_into_structs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.5, -1.25], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let mut st = StructValue::new();
            st.insert("data", Value::GpuTensor(handle));
            st.insert("label", Value::from("gpu result"));

            let result = block_on(gather_builtin(vec![Value::Struct(st)])).expect("gather");
            let Value::Struct(gathered) = result else {
                panic!("expected struct result");
            };
            let Some(Value::Tensor(host)) = gathered.fields.get("data") else {
                panic!("missing tensor field");
            };
            assert_eq!(host.shape, vec![2, 1]);
            assert_eq!(host.materialize_f64(), tensor.materialize_f64());
            let Some(Value::String(label)) = gathered.fields.get("label") else {
                panic!("missing label");
            };
            assert_eq!(label, "gpu result");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_returns_output_list_for_multiple_inputs() {
        let result = block_on(gather_builtin(vec![Value::Num(1.0), Value::from("two")]))
            .expect("gather outputs");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list for multiple inputs");
        };
        assert_eq!(outputs, vec![Value::Num(1.0), Value::from("two")]);
    }

    #[test]
    fn gather_recursive_container_form_is_gated() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let values = tensor.materialize_f64();
            let handle = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &tensor.shape,
                })
                .expect("upload");
            let cell = CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(gather_builtin(vec![Value::Cell(cell)]))
                .expect_err("recursive gather is an extension");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GatherRecursiveContainerExtension")
            );
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn evaluate_returns_outputs_in_order() {
        let eval = block_on(evaluate(&[
            Value::Num(5.0),
            Value::Bool(true),
            Value::from("hello"),
        ]))
        .expect("eval");
        assert_eq!(eval.len(), 3);
        assert_eq!(eval.outputs()[0], Value::Num(5.0));
        assert_eq!(eval.outputs()[1], Value::Bool(true));
        assert_eq!(eval.outputs()[2], Value::from("hello"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_requires_at_least_one_argument() {
        let err = block_on(gather_builtin(Vec::new())).expect_err("expected error");
        assert_eq!(err.to_string(), GATHER_ERROR_NOT_ENOUGH_INPUTS.message);
        assert_eq!(err.identifier(), GATHER_ERROR_NOT_ENOUGH_INPUTS.identifier);
    }

    #[test]
    fn gather_type_resolves_corresponding_multiple_outputs() {
        assert_eq!(
            gather_type(&[Type::Num, Type::String], &ResolveContext::new(Vec::new())),
            Type::OutputList(vec![Type::Num, Type::String])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gather_wgpu_provider_roundtrip() {
        use runmat_accelerate_api::AccelProvider;

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
                let view = HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                };
                let handle = provider.upload(&view).expect("upload");
                let eval =
                    block_on(evaluate(&[Value::GpuTensor(handle.clone())])).expect("evaluate");
                let outputs = eval.into_outputs();
                assert_eq!(outputs.len(), 1);
                match outputs.into_iter().next().unwrap() {
                    Value::Tensor(host) => {
                        assert_eq!(host.shape, tensor.shape);
                        assert_eq!(host.materialize_f64(), tensor.materialize_f64());
                    }
                    other => panic!("expected tensor value, got {other:?}"),
                }
                let _ = provider.free(&handle);
            }
            Err(err) => {
                tracing::warn!("Skipping gather_wgpu_provider_roundtrip: {err}");
            }
        }
        // Restore the simple provider so subsequent tests see a predictable backend.
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }
}
