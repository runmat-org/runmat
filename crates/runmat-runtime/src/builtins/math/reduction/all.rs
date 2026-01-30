//! MATLAB-compatible `all` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{canonical_scalar_shape, is_scalar_shape, normalize_scalar_shape},
    tensor,
    type_shapes::reduce_logical_type,
};
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorOwned};
use runmat_builtins::{CharArray, ComplexTensor, LogicalArray, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

const NAME: &str = "all";

fn all_type(args: &[Type]) -> Type {
    reduce_logical_type(args)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::all")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "all",
    op_kind: GpuOpKind::Reduction,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        crate::builtins::common::spec::ProviderHook::Reduction {
            name: "reduce_all_dim",
        },
        crate::builtins::common::spec::ProviderHook::Reduction {
            name: "reduce_all",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: Some(256),
    workgroup_size: Some(256),
    accepts_nan_mode: true,
    notes: "Providers may execute device-side AND reductions; runtimes gather to host when hooks are unavailable.",
};

fn all_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::all")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "all",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "accumulator *= select(0.0, 1.0, ({input} != 0.0) || ({input} != {input}));"
            ))
        },
    }),
    emits_nan: false,
    notes: "Fusion reductions treat NaNs as true; providers can substitute native kernels when profitable.",
};

#[runtime_builtin(
    name = "all",
    category = "math/reduction",
    summary = "Test whether every element of an array is nonzero with MATLAB-compatible options.",
    keywords = "all,logical,reduction,omitnan,gpu",
    accel = "reduction",
    type_resolver(all_type),
    builtin_path = "crate::builtins::math::reduction::all"
)]
async fn all_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let (spec, nan_mode) = parse_arguments(&rest).await?;
    match value {
        Value::GpuTensor(handle) => all_gpu(handle, spec, nan_mode).await,
        other => all_host(other, spec, nan_mode).await,
    }
}

async fn all_host(
    value: Value,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    let truth = TruthTensor::from_value(value).await?;
    let reduced = apply_reduction(truth, spec, nan_mode)?;
    reduced.into_value()
}

async fn all_gpu(
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return gpu_fallback(handle, spec, nan_mode).await,
    };
    match try_all_gpu(provider, &handle, &spec, nan_mode).await? {
        Some(host) => logical_from_host(host),
        None => gpu_fallback(handle, spec, nan_mode).await,
    }
}

async fn gpu_fallback(
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    all_host(Value::Tensor(tensor), spec, nan_mode).await
}

async fn try_all_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Option<HostTensorOwned>> {
    let omit_nan = matches!(nan_mode, ReductionNaN::Omit);

    if let ReductionSpec::All = spec {
        if let Ok(tmp) = provider.reduce_all(handle, omit_nan).await {
            let host = download_handle_async(provider, &tmp)
                .await
                .map_err(|e| all_error(format!("all: {e}")))?;
            let _ = provider.free(&tmp);
            return Ok(Some(host));
        }
    }

    reduce_dims_gpu(provider, handle, spec, omit_nan).await
}

async fn reduce_dims_gpu(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    omit_nan: bool,
) -> BuiltinResult<Option<HostTensorOwned>> {
    let mut dims = dims_from_spec(spec, &handle.shape);
    if dims.is_empty() {
        return Ok(None);
    }
    dims.sort_unstable();
    dims.dedup();

    let mut current = handle.clone();
    let mut current_owned = false;
    let mut intermediates: Vec<GpuTensorHandle> = Vec::new();

    for dim in dims {
        if dim == 0 {
            if current_owned {
                let _ = provider.free(&current);
            }
            for owned in intermediates {
                let _ = provider.free(&owned);
            }
            return Ok(None);
        }
        let axis = dim - 1;
        if axis >= current.shape.len() {
            if current_owned {
                let _ = provider.free(&current);
            }
            for owned in intermediates {
                let _ = provider.free(&owned);
            }
            return Ok(None);
        }
        let next = provider.reduce_all_dim(&current, axis, omit_nan).await;
        match next {
            Ok(new_handle) => {
                if current_owned {
                    intermediates.push(current.clone());
                }
                current = new_handle;
                current_owned = true;
            }
            Err(_) => {
                if current_owned {
                    let _ = provider.free(&current);
                }
                for owned in intermediates {
                    let _ = provider.free(&owned);
                }
                return Ok(None);
            }
        }
    }

    if !current_owned {
        return Ok(None);
    }

    let host = download_handle_async(provider, &current)
        .await
        .map_err(|e| all_error(format!("all: {e}")))?;
    let _ = provider.free(&current);
    for owned in intermediates {
        let _ = provider.free(&owned);
    }
    Ok(Some(host))
}

fn logical_from_host(host: HostTensorOwned) -> BuiltinResult<Value> {
    if host.data.len() == 1 {
        return Ok(Value::Bool(host.data[0] != 0.0));
    }
    let shape = if tensor::element_count(&host.shape) == host.data.len() {
        normalize_scalar_shape(&host.shape)
    } else if is_scalar_shape(&host.shape) {
        if host.data.is_empty() {
            Vec::new()
        } else {
            vec![host.data.len()]
        }
    } else {
        host.shape.clone()
    };
    let logical_data: Vec<u8> = host
        .data
        .into_iter()
        .map(|v| if v != 0.0 { 1 } else { 0 })
        .collect();
    LogicalArray::new(logical_data, shape)
        .map(Value::LogicalArray)
        .map_err(|e| all_error(format!("all: {e}")))
}

fn dims_from_spec(spec: &ReductionSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        ReductionSpec::Default => vec![default_dimension_from_shape(shape)],
        ReductionSpec::Dim(dim) => vec![*dim],
        ReductionSpec::VecDim(dims) => {
            let mut sorted = dims.clone();
            sorted.sort_unstable();
            sorted.dedup();
            sorted
        }
        ReductionSpec::All => {
            if is_scalar_shape(shape) {
                vec![1]
            } else {
                (1..=shape.len()).collect()
            }
        }
    }
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[derive(Debug, Clone)]
enum ReductionSpec {
    Default,
    Dim(usize),
    VecDim(Vec<usize>),
    All,
}

#[derive(Clone)]
struct TruthTensor {
    shape: Vec<usize>,
    data: Vec<TruthValue>,
}

#[derive(Clone, Copy)]
struct TruthValue {
    truthy: bool,
    has_nan: bool,
}

impl TruthValue {
    fn from_bool(truthy: bool) -> Self {
        Self {
            truthy,
            has_nan: false,
        }
    }
}

impl TruthTensor {
    async fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(t) => Ok(Self::from_tensor(t)),
            Value::LogicalArray(logical) => Ok(Self::from_logical(logical)),
            Value::Num(n) => Ok(Self::from_tensor(
                Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| all_error(format!("{NAME}: {e}")))?,
            )),
            Value::Int(i) => Ok(Self::from_tensor(
                Tensor::new(vec![i.to_f64()], vec![1, 1])
                    .map_err(|e| all_error(format!("{NAME}: {e}")))?,
            )),
            Value::Bool(b) => Ok(Self {
                shape: vec![1, 1],
                data: vec![TruthValue {
                    truthy: b,
                    has_nan: false,
                }],
            }),
            Value::Complex(re, im) => Ok(Self {
                shape: vec![1, 1],
                data: vec![TruthValue {
                    truthy: if re.is_nan() || im.is_nan() {
                        true
                    } else {
                        re != 0.0 || im != 0.0
                    },
                    has_nan: re.is_nan() || im.is_nan(),
                }],
            }),
            Value::ComplexTensor(ct) => Ok(Self::from_complex_tensor(ct)),
            Value::CharArray(ca) => Ok(Self::from_char_array(ca)),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
                Ok(Self::from_tensor(tensor))
            }
            other => Err(all_error(format!(
                "{NAME}: unsupported input type {:?}; expected numeric, logical, complex, or char data",
                other
            ))),
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        let shape = if tensor::element_count(&tensor.shape) == tensor.data.len() {
            normalize_scalar_shape(&tensor.shape)
        } else if is_scalar_shape(&tensor.shape) {
            if tensor.data.is_empty() {
                Vec::new()
            } else {
                vec![tensor.data.len()]
            }
        } else {
            tensor.shape.clone()
        };
        let data = tensor
            .data
            .iter()
            .map(|&v| TruthValue {
                truthy: if v.is_nan() { true } else { v != 0.0 },
                has_nan: v.is_nan(),
            })
            .collect();
        TruthTensor { shape, data }
    }

    fn from_logical(logical: LogicalArray) -> Self {
        let data = logical
            .data
            .iter()
            .map(|&b| TruthValue {
                truthy: b != 0,
                has_nan: false,
            })
            .collect();
        TruthTensor {
            shape: logical.shape.clone(),
            data,
        }
    }

    fn from_complex_tensor(ct: ComplexTensor) -> Self {
        let data = ct
            .data
            .iter()
            .map(|&(re, im)| TruthValue {
                truthy: if re.is_nan() || im.is_nan() {
                    true
                } else {
                    re != 0.0 || im != 0.0
                },
                has_nan: re.is_nan() || im.is_nan(),
            })
            .collect();
        TruthTensor {
            shape: ct.shape.clone(),
            data,
        }
    }

    fn from_char_array(ca: CharArray) -> Self {
        let data = ca
            .data
            .iter()
            .map(|&ch| TruthValue {
                truthy: (ch as u32) != 0,
                has_nan: false,
            })
            .collect();
        TruthTensor {
            shape: vec![ca.rows, ca.cols],
            data,
        }
    }

    fn reduce_dim(&self, dim: usize, nan_mode: ReductionNaN) -> BuiltinResult<Self> {
        if dim == 0 {
            return Err(all_error("all: dimension must be >= 1"));
        }
        if is_scalar_shape(&self.shape) {
            let truth = self.data.first().copied().unwrap_or(TruthValue {
                truthy: true,
                has_nan: false,
            });
            let truthy = match nan_mode {
                ReductionNaN::Include => truth.truthy,
                ReductionNaN::Omit => {
                    if truth.has_nan {
                        true
                    } else {
                        truth.truthy
                    }
                }
            };
            return Ok(TruthTensor {
                shape: canonical_scalar_shape(),
                data: vec![TruthValue::from_bool(truthy)],
            });
        }
        if dim > self.shape.len() {
            return Ok(self.clone());
        }
        let axis = dim - 1;
        let reduce_len = self.shape[axis];
        let stride_before = product(&self.shape[..axis]);
        let stride_after = product(&self.shape[axis + 1..]);
        let mut out_shape = self.shape.clone();
        out_shape[axis] = 1;
        let mut out = Vec::with_capacity(stride_before.saturating_mul(stride_after));

        if stride_before == 0 || stride_after == 0 {
            return Ok(TruthTensor {
                shape: out_shape,
                data: out,
            });
        }

        for after in 0..stride_after {
            for before in 0..stride_before {
                let mut all_true = true;
                let mut saw_value = false;
                for k in 0..reduce_len {
                    let idx = before + k * stride_before + after * stride_before * reduce_len;
                    if let Some(value) = self.data.get(idx) {
                        match nan_mode {
                            ReductionNaN::Include => {
                                if value.has_nan {
                                    continue;
                                }
                                saw_value = true;
                                if !value.truthy {
                                    all_true = false;
                                    break;
                                }
                            }
                            ReductionNaN::Omit => {
                                if value.has_nan {
                                    continue;
                                }
                                saw_value = true;
                                if !value.truthy {
                                    all_true = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                if !saw_value {
                    all_true = true;
                }
                out.push(TruthValue::from_bool(all_true));
            }
        }

        Ok(TruthTensor {
            shape: out_shape,
            data: out,
        })
    }

    fn into_value(self) -> BuiltinResult<Value> {
        if self.data.len() == 1 {
            return Ok(Value::Bool(self.data[0].truthy));
        }
        let shape = if tensor::element_count(&self.shape) == self.data.len() {
            normalize_scalar_shape(&self.shape)
        } else if is_scalar_shape(&self.shape) {
            if self.data.is_empty() {
                Vec::new()
            } else {
                vec![self.data.len()]
            }
        } else {
            self.shape
        };
        let logical_data: Vec<u8> = self
            .data
            .into_iter()
            .map(|value| if value.truthy { 1 } else { 0 })
            .collect();
        LogicalArray::new(logical_data, shape)
            .map(Value::LogicalArray)
            .map_err(|e| all_error(format!("all: {e}")))
    }
}

fn apply_reduction(
    tensor: TruthTensor,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<TruthTensor> {
    match spec {
        ReductionSpec::Default => {
            tensor.reduce_dim(default_dimension_from_shape(&tensor.shape), nan_mode)
        }
        ReductionSpec::Dim(dim) => tensor.reduce_dim(dim, nan_mode),
        ReductionSpec::VecDim(mut dims) => {
            dims.sort_unstable();
            dims.dedup();
            let mut current = tensor;
            for dim in dims {
                current = current.reduce_dim(dim, nan_mode)?;
            }
            Ok(current)
        }
        ReductionSpec::All => {
            let mut current = tensor;
            if is_scalar_shape(&current.shape) {
                current = current.reduce_dim(1, nan_mode)?;
            } else {
                for dim in 1..=current.shape.len() {
                    current = current.reduce_dim(dim, nan_mode)?;
                }
            }
            Ok(current)
        }
    }
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<(ReductionSpec, ReductionNaN)> {
    let mut spec = ReductionSpec::Default;
    let mut nan_mode = ReductionNaN::Include;

    for arg in args {
        if is_all_token(arg) {
            if !matches!(spec, ReductionSpec::Default) {
                return Err(all_error(
                    "all: 'all' cannot be combined with dimension arguments",
                ));
            }
            spec = ReductionSpec::All;
            continue;
        }
        if let Some(mode) = parse_nan_mode(arg)? {
            if !matches!(nan_mode, ReductionNaN::Include) {
                return Err(all_error("all: multiple NaN handling options specified"));
            }
            nan_mode = mode;
            continue;
        }
        let dims = parse_dimensions(arg).await?;
        if dims.is_empty() {
            return Err(all_error(
                "all: dimension vector must contain at least one entry",
            ));
        }
        if dims.len() == 1 {
            if matches!(spec, ReductionSpec::Default) {
                spec = ReductionSpec::Dim(dims[0]);
            } else {
                return Err(all_error(
                    "all: multiple dimension specifications are not supported",
                ));
            }
        } else if matches!(spec, ReductionSpec::Default) {
            spec = ReductionSpec::VecDim(dims);
        } else {
            return Err(all_error(
                "all: multiple dimension specifications are not supported",
            ));
        }
    }

    Ok((spec, nan_mode))
}

fn parse_nan_mode(value: &Value) -> BuiltinResult<Option<ReductionNaN>> {
    let Some(text) = extract_text_token(value) else {
        return Ok(None);
    };
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "omitnan" => Ok(Some(ReductionNaN::Omit)),
        "includenan" => Ok(Some(ReductionNaN::Include)),
        _ => Err(all_error(format!("all: unknown option '{}'", text.trim()))),
    }
}

fn is_all_token(value: &Value) -> bool {
    extract_text_token(value)
        .map(|s| s.trim().eq_ignore_ascii_case("all"))
        .unwrap_or(false)
}

async fn parse_dimensions(value: &Value) -> BuiltinResult<Vec<usize>> {
    let dims = tensor::dims_from_value_async(value)
        .await
        .map_err(map_dims_error)?;
    let dims = match dims {
        Some(dims) => dims,
        None => match tensor::dimension_from_value_async(value, "all", false)
            .await
            .map_err(map_dims_error)?
        {
            Some(dim) => vec![dim],
            None => return Ok(Vec::new()),
        },
    };
    if dims.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for dim in dims {
        if dim < 1 {
            return Err(all_error("all: dimension values must be >= 1"));
        }
        if !out.contains(&dim) {
            out.push(dim);
        }
    }
    Ok(out)
}

fn map_dims_error(message: String) -> RuntimeError {
    if message.contains("finite") {
        return all_error("all: dimension values must be finite");
    }
    if message.contains("integer") {
        return all_error("all: dimension values must be integers");
    }
    if message.contains("non-negative") {
        return all_error("all: dimension values must be >= 1");
    }
    all_error(message)
}

fn extract_text_token(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, v| acc.saturating_mul(v))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, ComplexTensor, IntValue};

    fn all_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::all_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 1.0, 4.0, 5.0, 0.0, 6.0], vec![2, 3]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.data, vec![1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_zero_column_matrix_returns_empty_row() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![2, 0]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), Vec::new()).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 0]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_row_dimension() {
        let tensor = Tensor::new(vec![1.0, 1.0, 4.0, 5.0, 0.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = all_builtin(Value::Tensor(tensor), args).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_zero_row_matrix_dim_two() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let result = all_builtin(Value::Tensor(tensor), args).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![0, 1]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_vecdim_multiple_axes() {
        let tensor = Tensor::new((1..=24).map(|v| v as f64).collect(), vec![3, 4, 2]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), vec![Value::Tensor(vecdim)]).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1, 2]);
                assert_eq!(out.data, vec![1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_all_option_returns_scalar() {
        let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("all");
        match result {
            Value::Bool(flag) => assert!(!flag),
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_all_on_empty_returns_true() {
        let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
        let result = all_builtin(Value::Tensor(tensor), vec![Value::from("all")]).expect("all");
        match result {
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_handles_nan_modes() {
        let tensor = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 0.0], vec![2, 2]).unwrap();
        let includenan = all_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("all");
        match includenan {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }

        let omit =
            all_builtin(Value::Tensor(tensor), vec![Value::from("omitnan")]).expect("all omit");
        match omit {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_char_array_support() {
        let chars = CharArray::new("a\0c".chars().collect(), 1, 3).unwrap();
        let result =
            all_builtin(Value::CharArray(chars), vec![Value::Int(IntValue::I32(1))]).expect("all");
        match result {
            Value::LogicalArray(out) => assert_eq!(out.data, vec![1, 0, 1]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_includenan_keyword_allowed() {
        let tensor = Tensor::new(vec![f64::NAN, 1.0], vec![2, 1]).unwrap();
        let result =
            all_builtin(Value::Tensor(tensor), vec![Value::from("includenan")]).expect("all");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_complex_tensor_with_omitnan() {
        let complex = ComplexTensor::new(vec![(f64::NAN, 0.0), (1.0, 0.0)], vec![2, 1]).unwrap();
        let tensor_value = Value::ComplexTensor(complex);
        let omit = all_builtin(tensor_value.clone(), vec![Value::from("omitnan")]).expect("all");
        match omit {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical array, got {other:?}"),
        }
        let include = all_builtin(tensor_value, Vec::new()).expect("all include");
        match include {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1]);
                assert_eq!(out.data, vec![1]);
            }
            Value::Bool(flag) => assert!(flag),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_vecdim_with_omitnan() {
        let mut data = vec![0.0; 8];
        data[7] = f64::NAN;
        let tensor = Tensor::new(data, vec![2, 2, 2]).unwrap();
        let vecdim = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = vec![Value::Tensor(vecdim), Value::from("omitnan")];
        let result = all_builtin(Value::Tensor(tensor), args).expect("all vecdim omitnan");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![1, 1, 2]);
                assert_eq!(out.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_all_with_dim_errors() {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap();
        let args = vec![Value::from("all"), Value::Int(IntValue::I32(1))];
        let err = all_builtin(Value::Tensor(tensor), args).unwrap_err();
        assert!(
            err.message().contains("dimension"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = all_builtin(Value::GpuTensor(handle), Vec::new()).expect("all");
            match result {
                Value::LogicalArray(out) => {
                    assert_eq!(out.shape, vec![1, 2]);
                    assert_eq!(out.data, vec![1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_gpu_provider_omitnan_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![f64::NAN, f64::NAN, 1.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                all_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).expect("all");
            match result {
                Value::LogicalArray(out) => {
                    assert_eq!(out.shape, vec![1, 2]);
                    assert_eq!(out.data, vec![1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_wgpu_default_matches_cpu() {
        let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        }));
        let Ok(reg_result) = init else {
            tracing::warn!(
                "skipping all_wgpu_default_matches_cpu: wgpu provider panicked during init"
            );
            return;
        };
        if reg_result.is_err() {
            tracing::warn!("skipping all_wgpu_default_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0], vec![2, 3]).unwrap();
        let cpu = block_on(all_host(
            Value::Tensor(tensor.clone()),
            ReductionSpec::Default,
            ReductionNaN::Include,
        ))
        .unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => {
                tracing::warn!("skipping all_wgpu_default_matches_cpu: provider not registered");
                return;
            }
        };
        let handle = match provider.upload(&view) {
            Ok(h) => h,
            Err(err) => {
                tracing::warn!("skipping all_wgpu_default_matches_cpu: upload failed: {err}");
                return;
            }
        };
        let gpu = all_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();
        match (cpu, gpu) {
            (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
                assert_eq!(expected.shape, actual.shape);
                assert_eq!(expected.data, actual.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn all_wgpu_omitnan_matches_cpu() {
        let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        }));
        let Ok(reg_result) = init else {
            tracing::warn!(
                "skipping all_wgpu_omitnan_matches_cpu: wgpu provider panicked during init"
            );
            return;
        };
        if reg_result.is_err() {
            tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: wgpu provider unavailable");
            return;
        }
        let tensor = Tensor::new(vec![f64::NAN, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let cpu = block_on(all_host(
            Value::Tensor(tensor.clone()),
            ReductionSpec::Default,
            ReductionNaN::Omit,
        ))
        .unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => {
                tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: provider not registered");
                return;
            }
        };
        let handle = match provider.upload(&view) {
            Ok(h) => h,
            Err(err) => {
                tracing::warn!("skipping all_wgpu_omitnan_matches_cpu: upload failed: {err}");
                return;
            }
        };
        let gpu = all_builtin(Value::GpuTensor(handle), vec![Value::from("omitnan")]).unwrap();
        match (cpu, gpu) {
            (Value::LogicalArray(expected), Value::LogicalArray(actual)) => {
                assert_eq!(expected.shape, actual.shape);
                assert_eq!(expected.data, actual.data);
            }
            _ => panic!("unexpected shapes"),
        }
    }
}
