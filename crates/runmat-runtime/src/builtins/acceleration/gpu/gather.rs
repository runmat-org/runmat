//! MATLAB-compatible `gather` builtin with provider-aware semantics.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    GATHER_CONTAINER_EXTENSION, GATHER_ERROR_NOT_ENOUGH_INPUTS, GATHER_ERROR_OUTPUT_COUNT_MISMATCH,
    GATHER_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gather")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gather",
    op_kind: GpuOpKind::Custom("gather"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("download_numeric")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Downloads gpuArray handles through the provider's native numeric contract without mutating the source handle; host inputs pass through unchanged.",
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
    binding_variant = "default",
    builtin_path = "crate::builtins::acceleration::gpu::gather"
)]
async fn gather_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let eval = evaluate_arguments(args.view()).await?;
    let len = eval.len();
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if len == 1 {
            if out_count > 1 {
                return Err(gather_error(&GATHER_ERROR_TOO_MANY_OUTPUTS).into());
            }
            return Ok(Value::output_list(vec![eval.into_first()]));
        }
        if out_count != len {
            return Err(gather_error(&GATHER_ERROR_OUTPUT_COUNT_MISMATCH).into());
        }
        return Ok(Value::output_list(eval.into_outputs()));
    }
    if len == 1 {
        Ok(eval.into_first())
    } else {
        Ok(Value::output_list(eval.into_outputs()))
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
    evaluate_arguments(crate::call::FunctionArgumentView::from_slice(args)).await
}

async fn evaluate_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<GatherResult> {
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
    use runmat_value::{CellArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_passes_through_host_values() {
        let value = Value::Num(42.0);
        let result = block_on(gather_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![value.clone()]),
        ))
        .expect("gather");
        assert_eq!(result, value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_downloads_gpu_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let explicit = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit immutable view");
            drop(handle);
            let handle = explicit;
            let lease = handle.allocation_lease().downgrade();
            let result = block_on(gather_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle.clone(),
                )]),
            ))
            .expect("gather");
            match result {
                Value::Tensor(host) => {
                    assert_eq!(host.shape(), tensor.shape());
                    assert_eq!(host.materialize_f64(), tensor.materialize_f64());
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
            assert!(runmat_accelerate_api::handle_is_explicit(&handle));
            block_on(gather_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle.clone(),
                )]),
            ))
            .expect("source handle remains gatherable");
            handle.release().expect("release caller lease");
            assert!(lease.acquire().is_none());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_preserves_logical_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[0, 1, 1, 0],
                    shape: &[2, 2],
                })
                .expect("logical producer upload");
            let lease = handle.allocation_lease().downgrade();
            let result = block_on(gather_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                    handle,
                )]),
            ))
            .expect("gather");
            match result {
                Value::LogicalArray(logical) => {
                    assert_eq!(logical.shape, vec![2, 2]);
                    assert_eq!(logical.data, vec![0, 1, 1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
            assert!(lease.acquire().is_none());
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
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let cell = CellArray::new(vec![Value::GpuTensor(handle), Value::from("host")], 1, 2)
                .expect("cell");
            let result = block_on(gather_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Cell(cell)]),
            ))
            .expect("gather");
            let Value::Cell(gathered) = result else {
                panic!("expected cell result");
            };
            let first = gathered.get(0, 0).expect("first element");
            match first {
                Value::Tensor(t) => {
                    assert_eq!(t.shape(), &[2, 1]);
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
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let mut st = runmat_value::UnboundedStructValueBuilder::new();
            st.insert("data", Value::GpuTensor(handle));
            st.insert("label", Value::from("gpu result"));

            let result = block_on(gather_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Struct(
                    st.finish(),
                )]),
            ))
            .expect("gather");
            let Value::Struct(gathered) = result else {
                panic!("expected struct result");
            };
            let Some(Value::Tensor(host)) = gathered.fields.get("data") else {
                panic!("missing tensor field");
            };
            assert_eq!(host.shape(), &[2, 1]);
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
        let result = block_on(gather_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(1.0),
                Value::from("two"),
            ]),
        ))
        .expect("gather outputs");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list for multiple inputs");
        };
        assert_eq!(outputs.as_slice(), &[Value::Num(1.0), Value::from("two")]);
    }

    #[test]
    fn gather_recursive_container_form_is_gated() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let values = tensor.materialize_f64();
            let handle = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: tensor.shape(),
                })
                .expect("upload");
            let cell = CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
            provider.reset_telemetry();
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(gather_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Cell(cell)]),
            ))
            .expect_err("recursive gather is an extension");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:GatherRecursiveContainerExtension")
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert!(
                block_on(provider.download(&handle)).is_ok(),
                "caller alias remains live"
            );
            handle
                .release()
                .expect("release caller reference; managed cell remains GC-owned");
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
        let err = block_on(gather_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("expected error");
        assert_eq!(err.to_string(), GATHER_ERROR_NOT_ENOUGH_INPUTS.message);
        assert_eq!(err.identifier(), GATHER_ERROR_NOT_ENOUGH_INPUTS.identifier);
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
                    shape: tensor.shape(),
                };
                let handle = provider.upload(&view).expect("upload");
                let eval =
                    block_on(evaluate(&[Value::GpuTensor(handle.clone())])).expect("evaluate");
                let outputs = eval.into_outputs();
                assert_eq!(outputs.len(), 1);
                match outputs.into_iter().next().unwrap() {
                    Value::Tensor(host) => {
                        assert_eq!(host.shape(), tensor.shape());
                        assert_eq!(host.materialize_f64(), tensor.materialize_f64());
                    }
                    other => panic!("expected tensor value, got {other:?}"),
                }
                let lease = handle.allocation_lease().downgrade();
                handle.release().expect("release caller lease");
                assert!(lease.acquire().is_none());
            }
            Err(err) => {
                tracing::warn!("Skipping gather_wgpu_provider_roundtrip: {err}");
            }
        }
        // Restore the simple provider so subsequent tests see a predictable backend.
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }
}
