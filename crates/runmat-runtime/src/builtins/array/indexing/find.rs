//! MATLAB-compatible `find` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{HostTensorView, ProviderFindResult};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::find")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "find",
    op_kind: GpuOpKind::Custom("find"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("find")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "WGPU provider executes find directly on device; other providers fall back to host and re-upload results to preserve residency.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::indexing::find")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "find",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Find drives control flow and currently bypasses fusion; metadata is present for completeness only.",
};

#[runtime_builtin(
    name = "find",
    category = "array/indexing",
    summary = "Locate indices and values of nonzero elements.",
    keywords = "find,nonzero,indices,row,column,gpu",
    accel = "custom",
    builtin_path = "crate::builtins::array::indexing::find"
)]
fn find_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(value, &rest)?;
    eval.linear_value()
}

/// Evaluate `find` and return an object that can materialise the various outputs.
pub fn evaluate(value: Value, args: &[Value]) -> Result<FindEval, String> {
    let options = parse_options(args)?;
    match value {
        Value::GpuTensor(handle) => {
            if let Some(result) = try_provider_find(&handle, &options) {
                return Ok(FindEval::from_gpu(result));
            }
            let (storage, _) = materialize_input(Value::GpuTensor(handle))?;
            let result = compute_find(&storage, &options);
            Ok(FindEval::from_host(result, true))
        }
        other => {
            let (storage, input_was_gpu) = materialize_input(other)?;
            let result = compute_find(&storage, &options);
            Ok(FindEval::from_host(result, input_was_gpu))
        }
    }
}

fn try_provider_find(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    options: &FindOptions,
) -> Option<ProviderFindResult> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = runmat_accelerate_api::provider()?;
    let direction = match options.direction {
        FindDirection::First => runmat_accelerate_api::FindDirection::First,
        FindDirection::Last => runmat_accelerate_api::FindDirection::Last,
    };
    let limit = options.effective_limit();
    provider.find(handle, limit, direction).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FindDirection {
    First,
    Last,
}

#[derive(Debug, Clone)]
struct FindOptions {
    limit: Option<usize>,
    direction: FindDirection,
}

impl Default for FindOptions {
    fn default() -> Self {
        Self {
            limit: None,
            direction: FindDirection::First,
        }
    }
}

impl FindOptions {
    fn effective_limit(&self) -> Option<usize> {
        match self.direction {
            FindDirection::Last => self.limit.or(Some(1)),
            FindDirection::First => self.limit,
        }
    }
}

#[derive(Clone)]
enum DataStorage {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl DataStorage {
    fn shape(&self) -> &[usize] {
        match self {
            DataStorage::Real(t) => &t.shape,
            DataStorage::Complex(t) => &t.shape,
        }
    }
}

#[derive(Clone)]
struct FindResult {
    shape: Vec<usize>,
    indices: Vec<usize>,
    values: FindValues,
}

#[derive(Clone)]
enum FindValues {
    Real(Vec<f64>),
    Complex(Vec<(f64, f64)>),
}

pub struct FindEval {
    inner: FindEvalInner,
}

enum FindEvalInner {
    Host {
        result: FindResult,
        prefer_gpu: bool,
    },
    Gpu {
        result: ProviderFindResult,
    },
}

impl FindEval {
    fn from_host(result: FindResult, prefer_gpu: bool) -> Self {
        Self {
            inner: FindEvalInner::Host { result, prefer_gpu },
        }
    }

    fn from_gpu(result: ProviderFindResult) -> Self {
        Self {
            inner: FindEvalInner::Gpu { result },
        }
    }

    pub fn linear_value(&self) -> Result<Value, String> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => {
                let tensor = result.linear_tensor()?;
                Ok(tensor_to_value(tensor, *prefer_gpu))
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.linear.clone())),
        }
    }

    pub fn row_value(&self) -> Result<Value, String> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => {
                let tensor = result.row_tensor()?;
                Ok(tensor_to_value(tensor, *prefer_gpu))
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.rows.clone())),
        }
    }

    pub fn column_value(&self) -> Result<Value, String> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => {
                let tensor = result.column_tensor()?;
                Ok(tensor_to_value(tensor, *prefer_gpu))
            }
            FindEvalInner::Gpu { result } => Ok(Value::GpuTensor(result.cols.clone())),
        }
    }

    pub fn values_value(&self) -> Result<Value, String> {
        match &self.inner {
            FindEvalInner::Host { result, prefer_gpu } => result.values_value(*prefer_gpu),
            FindEvalInner::Gpu { result } => result
                .values
                .as_ref()
                .map(|handle| Value::GpuTensor(handle.clone()))
                .ok_or_else(|| "find: provider did not return values buffer".to_string()),
        }
    }
}

fn parse_options(args: &[Value]) -> Result<FindOptions, String> {
    match args.len() {
        0 => Ok(FindOptions::default()),
        1 => {
            if is_direction_like(&args[0]) {
                let direction_opt = parse_direction(&args[0])?;
                let limit = if matches!(direction_opt, Some(FindDirection::Last)) {
                    Some(1)
                } else {
                    None
                };
                let direction = direction_opt.unwrap_or(FindDirection::First);
                Ok(FindOptions { limit, direction })
            } else {
                let limit = parse_limit(&args[0])?;
                Ok(FindOptions {
                    limit: Some(limit),
                    direction: FindDirection::First,
                })
            }
        }
        2 => {
            let limit = parse_limit(&args[0])?;
            let direction = parse_direction(&args[1])?
                .ok_or_else(|| "find: third argument must be 'first' or 'last'".to_string())?;
            Ok(FindOptions {
                limit: Some(limit),
                direction,
            })
        }
        _ => Err("find: too many input arguments".to_string()),
    }
}

fn parse_direction(value: &Value) -> Result<Option<FindDirection>, String> {
    if let Some(text) = tensor::value_to_string(value) {
        let lowered = text.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "first" => Ok(Some(FindDirection::First)),
            "last" => Ok(Some(FindDirection::Last)),
            _ => Err("find: direction must be 'first' or 'last'".to_string()),
        }
    } else {
        Ok(None)
    }
}

fn is_direction_like(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::StringArray(sa) => sa.data.len() == 1,
        Value::CharArray(ca) => ca.rows == 1,
        _ => false,
    }
}

fn parse_limit(value: &Value) -> Result<usize, String> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(handle)?;
            parse_limit_tensor(&tensor)
        }
        _ => {
            let tensor = tensor::value_to_tensor(value)?;
            parse_limit_tensor(&tensor)
        }
    }
}

fn parse_limit_tensor(tensor: &Tensor) -> Result<usize, String> {
    if tensor.data.len() != 1 {
        return Err("find: second argument must be a scalar".to_string());
    }
    parse_limit_scalar(tensor.data[0])
}

fn parse_limit_scalar(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err("find: K must be a finite, non-negative integer".to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err("find: K must be a finite, non-negative integer".to_string());
    }
    if rounded < 0.0 {
        return Err("find: K must be >= 0".to_string());
    }
    Ok(rounded as usize)
}

fn materialize_input(value: Value) -> Result<(DataStorage, bool), String> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor(&handle)?;
            Ok((DataStorage::Real(tensor), true))
        }
        Value::Tensor(tensor) => Ok((DataStorage::Real(tensor), false)),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Num(n) => {
            let tensor = Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("find: {e}"))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Int(i) => {
            let tensor =
                Tensor::new(vec![i.to_f64()], vec![1, 1]).map_err(|e| format!("find: {e}"))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Bool(b) => {
            let tensor = Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| format!("find: {e}"))?;
            Ok((DataStorage::Real(tensor), false))
        }
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(|e| format!("find: {e}"))?;
            Ok((DataStorage::Complex(tensor), false))
        }
        Value::ComplexTensor(tensor) => Ok((DataStorage::Complex(tensor), false)),
        Value::CharArray(chars) => {
            let mut data = Vec::with_capacity(chars.data.len());
            for c in 0..chars.cols {
                for r in 0..chars.rows {
                    let ch = chars.data[r * chars.cols + c] as u32;
                    data.push(ch as f64);
                }
            }
            let tensor = Tensor::new(data, vec![chars.rows, chars.cols])
                .map_err(|e| format!("find: {e}"))?;
            Ok((DataStorage::Real(tensor), false))
        }
        other => Err(format!(
            "find: unsupported input type {:?}; expected numeric, logical, or char data",
            other
        )),
    }
}

fn compute_find(storage: &DataStorage, options: &FindOptions) -> FindResult {
    let shape = storage.shape().to_vec();
    let limit = options.effective_limit();

    match storage {
        DataStorage::Real(tensor) => {
            let mut indices = Vec::new();
            let mut values = Vec::new();

            if matches!(limit, Some(0)) {
                return FindResult::new(shape, indices, FindValues::Real(values));
            }

            let len = tensor.data.len();
            match options.direction {
                FindDirection::First => {
                    for idx in 0..len {
                        let value = tensor.data[idx];
                        if value != 0.0 {
                            indices.push(idx + 1);
                            values.push(value);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
                FindDirection::Last => {
                    for idx in (0..len).rev() {
                        let value = tensor.data[idx];
                        if value != 0.0 {
                            indices.push(idx + 1);
                            values.push(value);
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
            }

            FindResult::new(shape, indices, FindValues::Real(values))
        }
        DataStorage::Complex(tensor) => {
            let mut indices = Vec::new();
            let mut values = Vec::new();

            if matches!(limit, Some(0)) {
                return FindResult::new(shape, indices, FindValues::Complex(values));
            }

            let len = tensor.data.len();
            match options.direction {
                FindDirection::First => {
                    for idx in 0..len {
                        let (re, im) = tensor.data[idx];
                        if re != 0.0 || im != 0.0 {
                            indices.push(idx + 1);
                            values.push((re, im));
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
                FindDirection::Last => {
                    for idx in (0..len).rev() {
                        let (re, im) = tensor.data[idx];
                        if re != 0.0 || im != 0.0 {
                            indices.push(idx + 1);
                            values.push((re, im));
                            if limit.is_some_and(|k| indices.len() >= k) {
                                break;
                            }
                        }
                    }
                }
            }

            FindResult::new(shape, indices, FindValues::Complex(values))
        }
    }
}

impl FindResult {
    fn new(shape: Vec<usize>, indices: Vec<usize>, values: FindValues) -> Self {
        Self {
            shape,
            indices,
            values,
        }
    }

    fn linear_tensor(&self) -> Result<Tensor, String> {
        let data: Vec<f64> = self.indices.iter().map(|&idx| idx as f64).collect();
        let rows = data.len();
        Tensor::new(data, vec![rows, 1]).map_err(|e| format!("find: {e}"))
    }

    fn row_tensor(&self) -> Result<Tensor, String> {
        let mut data = Vec::with_capacity(self.indices.len());
        let rows = self.shape.first().copied().unwrap_or(1).max(1);
        for &idx in &self.indices {
            let zero_based = idx - 1;
            let row = (zero_based % rows) + 1;
            data.push(row as f64);
        }
        Tensor::new(data, vec![self.indices.len(), 1]).map_err(|e| format!("find: {e}"))
    }

    fn column_tensor(&self) -> Result<Tensor, String> {
        let mut data = Vec::with_capacity(self.indices.len());
        let rows = self.shape.first().copied().unwrap_or(1).max(1);
        for &idx in &self.indices {
            let zero_based = idx - 1;
            let col = (zero_based / rows) + 1;
            data.push(col as f64);
        }
        Tensor::new(data, vec![self.indices.len(), 1]).map_err(|e| format!("find: {e}"))
    }

    fn values_value(&self, prefer_gpu: bool) -> Result<Value, String> {
        match &self.values {
            FindValues::Real(values) => {
                let tensor = Tensor::new(values.clone(), vec![values.len(), 1])
                    .map_err(|e| format!("find: {e}"))?;
                Ok(tensor_to_value(tensor, prefer_gpu))
            }
            FindValues::Complex(values) => {
                let tensor = ComplexTensor::new(values.clone(), vec![values.len(), 1])
                    .map_err(|e| format!("find: {e}"))?;
                Ok(complex_tensor_into_value(tensor))
            }
        }
    }
}

fn tensor_to_value(tensor: Tensor, prefer_gpu: bool) -> Value {
    if prefer_gpu {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Value::GpuTensor(handle);
            }
        }
    }
    tensor::tensor_into_value(tensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, IntValue};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_linear_indices_basic() {
        let tensor = Tensor::new(vec![0.0, 4.0, 0.0, 7.0, 0.0, 9.0], vec![2, 3]).unwrap();
        let value = find_builtin(Value::Tensor(tensor), Vec::new()).expect("find");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.data, vec![2.0, 4.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_limited_first() {
        let tensor = Tensor::new(vec![0.0, 3.0, 5.0, 0.0, 8.0], vec![1, 5]).unwrap();
        let result =
            find_builtin(Value::Tensor(tensor), vec![Value::Int(IntValue::I32(2))]).expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_last_single() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 6.0, 0.0, 2.0], vec![1, 6]).unwrap();
        let result = find_builtin(Value::Tensor(tensor), vec![Value::from("last")]).expect("find");
        match result {
            Value::Num(n) => assert_eq!(n, 6.0),
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![6.0]);
            }
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_complex_values() {
        let tensor =
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 2.0), (0.0, 0.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("find compute");
        let values = eval.values_value().expect("values");
        match values {
            Value::Complex(re, im) => {
                assert_eq!(re, 1.0);
                assert_eq!(im, 2.0);
            }
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 1]);
                assert_eq!(ct.data, vec![(1.0, 2.0)]);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 0.0, 7.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = find_builtin(Value::GpuTensor(handle), Vec::new()).expect("find");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 1]);
            assert_eq!(gathered.data, vec![2.0, 4.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_direction_error() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = find_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(1)), Value::from("invalid")],
        )
        .expect_err("expected error");
        assert!(err.contains("direction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_multi_output_rows_cols_values() {
        let tensor = Tensor::new(vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");

        let rows = test_support::gather(eval.row_value().expect("rows")).expect("gather rows");
        assert_eq!(rows.shape, vec![3, 1]);
        assert_eq!(rows.data, vec![2.0, 1.0, 2.0]);

        let cols = test_support::gather(eval.column_value().expect("cols")).expect("gather cols");
        assert_eq!(cols.shape, vec![3, 1]);
        assert_eq!(cols.data, vec![1.0, 2.0, 3.0]);

        let vals = test_support::gather(eval.values_value().expect("vals")).expect("gather vals");
        assert_eq!(vals.shape, vec![3, 1]);
        assert_eq!(vals.data, vec![2.0, 3.0, 6.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_last_order_descending() {
        let tensor = Tensor::new(vec![1.0, 0.0, 2.0, 3.0, 0.0], vec![1, 5]).unwrap();
        let result = find_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(2)), Value::from("last")],
        )
        .expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
                assert_eq!(t.data, vec![4.0, 3.0]);
            }
            Value::Num(_) => panic!("expected column vector"),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_limit_zero_returns_empty() {
        let tensor = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
        let result = find_builtin(Value::Tensor(tensor), vec![Value::Num(0.0)]).expect("find");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 1]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_char_array_supports_nonzero_codes() {
        let chars = CharArray::new(vec!['\0', 'A', '\0'], 1, 3).unwrap();
        let result = find_builtin(Value::CharArray(chars), Vec::new()).expect("find");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            Value::Tensor(t) => assert_eq!(t.data, vec![2.0]),
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn find_gpu_multi_outputs_return_gpu_handles() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 4.0, 5.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate");

            let rows = eval.row_value().expect("rows");
            assert!(matches!(rows, Value::GpuTensor(_)));
            let rows_host = test_support::gather(rows).expect("gather rows");
            assert_eq!(rows_host.data, vec![2.0, 1.0]);

            let cols = eval.column_value().expect("cols");
            assert!(matches!(cols, Value::GpuTensor(_)));
            let cols_host = test_support::gather(cols).expect("gather cols");
            assert_eq!(cols_host.data, vec![1.0, 2.0]);

            let vals = eval.values_value().expect("vals");
            assert!(matches!(vals, Value::GpuTensor(_)));
            let vals_host = test_support::gather(vals).expect("gather vals");
            assert_eq!(vals_host.data, vec![4.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn find_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 3.0, 4.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu evaluate");
        let cpu_linear =
            test_support::gather(cpu_eval.linear_value().expect("cpu linear")).expect("cpu gather");
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu evaluate");
        let gpu_linear =
            test_support::gather(gpu_eval.linear_value().expect("gpu linear")).expect("gpu gather");
        assert_eq!(gpu_linear.data, cpu_linear.data);
    }
}
