//! MATLAB-compatible `gather` builtin with provider-aware semantics.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::make_cell;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

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
    notes: "Downloads gpuArray handles via the provider's `download` hook and clears residency metadata; host inputs pass through unchanged.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::acceleration::gpu::gather")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gather",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a residency sink for fusion planning; always materialises host data and clears gpuArray residency tracking.",
};

#[runtime_builtin(
    name = "gather",
    category = "acceleration/gpu",
    summary = "Bring gpuArray data back to host memory.",
    keywords = "gather,gpuArray,accelerate,download",
    accel = "sink",
    builtin_path = "crate::builtins::acceleration::gpu::gather"
)]
fn gather_builtin(args: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&args)?;
    let len = eval.len();
    if len == 1 {
        Ok(eval.into_first())
    } else {
        let outputs = eval.into_outputs();
        make_cell(outputs, 1, len)
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
pub fn evaluate(args: &[Value]) -> Result<GatherResult, String> {
    if args.is_empty() {
        return Err("gather: not enough input arguments".to_string());
    }
    let mut outputs = Vec::with_capacity(args.len());
    for value in args {
        outputs.push(gather_argument(value)?);
    }
    Ok(GatherResult::new(outputs))
}

fn gather_argument(value: &Value) -> Result<Value, String> {
    match crate::dispatcher::gather_if_needed(value) {
        Ok(val) => Ok(val),
        Err(err) => {
            if err.trim_start().to_ascii_lowercase().starts_with("gather:") {
                Err(err)
            } else {
                Err(format!("gather: {err}"))
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, StructValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_passes_through_host_values() {
        let value = Value::Num(42.0);
        let result = gather_builtin(vec![value.clone()]).expect("gather");
        assert_eq!(result, value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_downloads_gpu_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = gather_builtin(vec![Value::GpuTensor(handle)]).expect("gather");
            match result {
                Value::Tensor(host) => {
                    assert_eq!(host.shape, tensor.shape);
                    assert_eq!(host.data, tensor.data);
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_preserves_logical_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let data = vec![0.0, 1.0, 1.0, 0.0];
            let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let result = gather_builtin(vec![Value::GpuTensor(handle)]).expect("gather");
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
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell = CellArray::new(vec![Value::GpuTensor(handle), Value::from("host")], 1, 2)
                .expect("cell");
            let result = gather_builtin(vec![Value::Cell(cell)]).expect("gather");
            let Value::Cell(gathered) = result else {
                panic!("expected cell result");
            };
            let first = gathered.get(0, 0).expect("first element");
            match first {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, tensor.data);
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
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.5, -1.25], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let mut st = StructValue::new();
            st.insert("data", Value::GpuTensor(handle));
            st.insert("label", Value::from("gpu result"));

            let result = gather_builtin(vec![Value::Struct(st)]).expect("gather");
            let Value::Struct(gathered) = result else {
                panic!("expected struct result");
            };
            let Some(Value::Tensor(host)) = gathered.fields.get("data") else {
                panic!("missing tensor field");
            };
            assert_eq!(host.shape, vec![2, 1]);
            assert_eq!(host.data, tensor.data);
            let Some(Value::String(label)) = gathered.fields.get("label") else {
                panic!("missing label");
            };
            assert_eq!(label, "gpu result");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_returns_cell_for_multiple_inputs() {
        let result =
            gather_builtin(vec![Value::Num(1.0), Value::from("two")]).expect("gather cell");
        let Value::Cell(cell) = result else {
            panic!("expected cell for multiple inputs");
        };
        assert_eq!(cell.rows, 1);
        assert_eq!(cell.cols, 2);
        assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
        assert_eq!(cell.get(0, 1).unwrap(), Value::from("two"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn evaluate_returns_outputs_in_order() {
        let eval =
            evaluate(&[Value::Num(5.0), Value::Bool(true), Value::from("hello")]).expect("eval");
        assert_eq!(eval.len(), 3);
        assert_eq!(eval.outputs()[0], Value::Num(5.0));
        assert_eq!(eval.outputs()[1], Value::Bool(true));
        assert_eq!(eval.outputs()[2], Value::from("hello"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_requires_at_least_one_argument() {
        let err = gather_builtin(Vec::new()).expect_err("expected error");
        assert_eq!(err, "gather: not enough input arguments");
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
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                let handle = provider.upload(&view).expect("upload");
                let eval = evaluate(&[Value::GpuTensor(handle.clone())]).expect("evaluate");
                let outputs = eval.into_outputs();
                assert_eq!(outputs.len(), 1);
                match outputs.into_iter().next().unwrap() {
                    Value::Tensor(host) => {
                        assert_eq!(host.shape, tensor.shape);
                        assert_eq!(host.data, tensor.data);
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
