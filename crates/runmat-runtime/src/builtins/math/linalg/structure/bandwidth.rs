//! MATLAB-compatible `bandwidth` builtin with GPU-aware semantics for RunMat.

use log::debug;
use runmat_accelerate_api::{self, GpuTensorHandle};
use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::{BANDWIDTH_DESCRIPTOR, BANDWIDTH_INTEGER_CAPABILITIES};
use runmat_builtins::{
    BANDWIDTH_ERROR_INTERNAL, BANDWIDTH_ERROR_INVALID_ARGUMENT, BANDWIDTH_ERROR_INVALID_INPUT,
    BANDWIDTH_INTEGER_INPUT_EXTENSION, BANDWIDTH_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntegerStorage, LogicalArray, SparseTensor, Tensor, Value};

use super::kernel_error::{StructureKernelError, StructureKernelResult};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::control::frequency_response;
use crate::builtins::control::tf_model::TF_CLASS;
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::bandwidth_type;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "bandwidth";

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "bandwidth",
    op_kind: GpuOpKind::Custom("structure_analysis"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("bandwidth")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "WGPU providers compute bandwidth on-device when available; runtimes gather to the host as a fallback when providers lack the hook.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "bandwidth",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Structure query that returns host double scalar metadata; fusion treats it as a metadata operation.",
};

fn bandwidth_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn bandwidth_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    bandwidth_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandSelector {
    Both,
    Lower,
    Upper,
}

#[runtime_builtin(
    name = "bandwidth",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::linalg::structure::bandwidth"
)]
async fn bandwidth_builtin(
    runtime: Option<RuntimeContext>,
    matrix: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(&matrix, Value::Object(object) if object.is_class(TF_CLASS)) {
        if rest.len() > 1 || crate::output_count::current_output_count().is_some_and(|n| n > 1) {
            return Err(bandwidth_error_with_detail(
                &BANDWIDTH_ERROR_INVALID_ARGUMENT,
                "transfer-function form accepts at most one dB drop and one output",
            ));
        }
        let value = frequency_response::bandwidth(runtime.as_ref(), matrix, rest.first()).await?;
        return if crate::output_count::current_output_count() == Some(0) {
            Ok(Value::output_list(Vec::new()))
        } else {
            Ok(value)
        };
    }
    ensure_input_extensions(&matrix)?;
    let selector = parse_selector(rest.view())?;
    let data = MatrixData::from_value(matrix)?;
    let (lower, upper) = data.bandwidth().await?;
    match selector {
        BandSelector::Both => bandwidth_outputs(lower, upper),
        BandSelector::Lower => one_bandwidth_output(lower),
        BandSelector::Upper => one_bandwidth_output(upper),
    }
}

fn ensure_input_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        );
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BANDWIDTH_INTEGER_INPUT_EXTENSION,
            NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(sparse) if sparse.is_logical())
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        );
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BANDWIDTH_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn bandwidth_outputs(lower: usize, upper: usize) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count().unwrap_or(1) {
        0 => Ok(Value::output_list(Vec::new())),
        1 => Ok(Value::Num(lower as f64)),
        2 => Ok(Value::output_list(vec![
            Value::Num(lower as f64),
            Value::Num(upper as f64),
        ])),
        _ => Err(bandwidth_error_with_detail(
            &BANDWIDTH_ERROR_INVALID_ARGUMENT,
            "too many output arguments",
        )),
    }
}

fn one_bandwidth_output(value: usize) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count().unwrap_or(1) {
        0 => Ok(Value::output_list(Vec::new())),
        1 => Ok(Value::Num(value as f64)),
        _ => Err(bandwidth_error_with_detail(
            &BANDWIDTH_ERROR_INVALID_ARGUMENT,
            "too many output arguments",
        )),
    }
}

fn parse_selector(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<BandSelector> {
    match args.len() {
        0 => Ok(BandSelector::Both),
        1 => {
            let text = tensor::value_to_string(&args[0]).ok_or_else(|| {
                bandwidth_error_with_detail(
                    &BANDWIDTH_ERROR_INVALID_ARGUMENT,
                    "selector must be a character vector or string scalar",
                )
            })?;
            let trimmed = text.trim();
            let lowered = trimmed.to_ascii_lowercase();
            match lowered.as_str() {
                "lower" => Ok(BandSelector::Lower),
                "upper" => Ok(BandSelector::Upper),
                other => Err(bandwidth_error_with_detail(
                    &BANDWIDTH_ERROR_INVALID_ARGUMENT,
                    format!("unrecognized selector '{other}'; expected 'lower' or 'upper'"),
                )),
            }
        }
        _ => Err(bandwidth_error_with_detail(
            &BANDWIDTH_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        )),
    }
}

fn value_into_tensor_for(name: &str, value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::LogicalArray(logical) => logical_to_tensor(name, &logical),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| bandwidth_error_with_detail(&BANDWIDTH_ERROR_INTERNAL, e)),
        Value::Int(i) => Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
            .map_err(|e| bandwidth_error_with_detail(&BANDWIDTH_ERROR_INTERNAL, e)),
        Value::Bool(b) => Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|e| bandwidth_error_with_detail(&BANDWIDTH_ERROR_INTERNAL, e)),
        other => Err(bandwidth_error_with_detail(
            &BANDWIDTH_ERROR_INVALID_INPUT,
            format!(
                "unsupported input type {:?}; expected numeric or logical values",
                other
            ),
        )),
    }
}

fn logical_to_tensor(_name: &str, logical: &LogicalArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = logical
        .data
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    Tensor::new(data, logical.shape.clone())
        .map_err(|e| bandwidth_error_with_detail(&BANDWIDTH_ERROR_INTERNAL, e))
}

enum MatrixData {
    Real(Tensor),
    Complex(ComplexTensor),
    Sparse(SparseTensor),
    Gpu(GpuTensorHandle),
}

impl MatrixData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::ComplexTensor(ct) => Ok(Self::Complex(ct)),
            Value::Complex(re, im) => {
                let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| bandwidth_error_with_detail(&BANDWIDTH_ERROR_INTERNAL, e))?;
                Ok(Self::Complex(tensor))
            }
            Value::GpuTensor(handle) => Ok(Self::Gpu(handle)),
            Value::SparseTensor(sparse) => Ok(Self::Sparse(sparse)),
            other => {
                let tensor = value_into_tensor_for(NAME, other)?;
                Ok(Self::Real(tensor))
            }
        }
    }

    async fn bandwidth(&self) -> BuiltinResult<(usize, usize)> {
        match self {
            MatrixData::Real(tensor) => bandwidth_host_real_tensor(tensor).map_err(Into::into),
            MatrixData::Complex(tensor) => {
                bandwidth_host_complex_tensor(tensor).map_err(Into::into)
            }
            MatrixData::Sparse(sparse) => Ok(bandwidth_host_sparse(sparse)),
            MatrixData::Gpu(handle) => bandwidth_gpu(handle).await,
        }
    }
}

fn bandwidth_host_sparse(sparse: &SparseTensor) -> (usize, usize) {
    let (mut lower, mut upper) = (0, 0);
    super::sparse_pattern::for_each_nonzero(sparse, |row, col| {
        if row >= col {
            lower = lower.max(row - col);
        } else {
            upper = upper.max(col - row);
        }
    });
    (lower, upper)
}

async fn bandwidth_gpu(handle: &GpuTensorHandle) -> BuiltinResult<(usize, usize)> {
    let (rows, cols) = ensure_matrix_shape(handle.shape())?;
    gpu_helpers::require_resident_compute_owner(handle, || {
        bandwidth_error_with_detail(
            &BANDWIDTH_ERROR_INTERNAL,
            "no acceleration provider owns the input",
        )
    })?;
    if rows == 0 || cols == 0 {
        return Ok((0, 0));
    }
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        let tensor = gpu_helpers::gather_tensor_async(handle).await?;
        return bandwidth_host_real_tensor(&tensor).map_err(Into::into);
    }
    if let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) {
        let _residency = gpu_helpers::ResidencyRestoreGuard::new(handle);
        match provider.bandwidth(handle) {
            Ok(result) => {
                let lower = result.lower as usize;
                let upper = result.upper as usize;
                return Ok((lower, upper));
            }
            Err(err)
                if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(
                    &err,
                ) =>
            {
                debug!("bandwidth: provider bandwidth fallback: {err}");
            }
            Err(err) => {
                return Err(super::kernel_error::with_provider_source(
                    bandwidth_error_with_detail(&BANDWIDTH_ERROR_INTERNAL, err.to_string()),
                    runmat_accelerate_api::ProviderExecutionError::from(err),
                ))
            }
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
    bandwidth_host_real_tensor(&tensor).map_err(Into::into)
}

pub fn ensure_matrix_shape(shape: &[usize]) -> StructureKernelResult<(usize, usize)> {
    match shape.len() {
        0 => Ok((1, 1)),
        1 => Ok((1, shape[0])),
        _ => {
            if shape[2..].iter().any(|&dim| dim > 1) {
                Err(StructureKernelError::detail(
                    NAME,
                    &BANDWIDTH_ERROR_INVALID_INPUT,
                    "input must be a 2-D matrix",
                ))
            } else {
                Ok((shape[0], shape[1]))
            }
        }
    }
}

pub fn bandwidth_host_real_data(
    shape: &[usize],
    data: &[f64],
) -> StructureKernelResult<(usize, usize)> {
    let (rows, cols) = ensure_matrix_shape(shape)?;
    Ok(compute_real_bandwidth(rows, cols, data))
}

pub fn bandwidth_host_complex_data(
    shape: &[usize],
    data: &[(f64, f64)],
) -> StructureKernelResult<(usize, usize)> {
    let (rows, cols) = ensure_matrix_shape(shape)?;
    Ok(compute_complex_bandwidth(rows, cols, data))
}

pub fn bandwidth_host_real_tensor(tensor: &Tensor) -> StructureKernelResult<(usize, usize)> {
    if let Some(storage) = tensor.integer_storage() {
        let (rows, cols) = ensure_matrix_shape(tensor.shape())?;
        return Ok(compute_integer_bandwidth(rows, cols, storage));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    bandwidth_host_real_data(tensor.shape(), &values)
}

pub fn bandwidth_host_complex_tensor(
    tensor: &ComplexTensor,
) -> StructureKernelResult<(usize, usize)> {
    if tensor.integer_storage().is_some() {
        let (rows, cols) = ensure_matrix_shape(tensor.shape())?;
        return Ok(compute_complex_integer_bandwidth(rows, cols, tensor));
    }
    bandwidth_host_complex_data(tensor.shape(), &tensor.materialize_f64())
}

fn compute_real_bandwidth(rows: usize, cols: usize, data: &[f64]) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lower = 0usize;
    let mut upper = 0usize;
    let stride = rows;
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * stride;
            if idx >= data.len() {
                break;
            }
            let value = data[idx];
            if value != 0.0 || value.is_nan() {
                if row >= col {
                    lower = lower.max(row - col);
                } else {
                    upper = upper.max(col - row);
                }
            }
        }
    }
    (lower, upper)
}

fn compute_integer_bandwidth(
    rows: usize,
    cols: usize,
    storage: &runmat_value::IntegerStorage,
) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lower = 0usize;
    let mut upper = 0usize;
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * rows;
            let Some(value) = storage.value_at(idx) else {
                break;
            };
            if !value.is_zero() {
                if row >= col {
                    lower = lower.max(row - col);
                } else {
                    upper = upper.max(col - row);
                }
            }
        }
    }
    (lower, upper)
}

fn compute_complex_bandwidth(rows: usize, cols: usize, data: &[(f64, f64)]) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lower = 0usize;
    let mut upper = 0usize;
    let stride = rows;
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * stride;
            if idx >= data.len() {
                break;
            }
            let (re, im) = data[idx];
            if !(re == 0.0 && im == 0.0) {
                if row >= col {
                    lower = lower.max(row - col);
                } else {
                    upper = upper.max(col - row);
                }
            }
        }
    }
    (lower, upper)
}

fn compute_complex_integer_bandwidth(
    rows: usize,
    cols: usize,
    tensor: &ComplexTensor,
) -> (usize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lower = 0usize;
    let mut upper = 0usize;
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * rows;
            let Some((real, imag)) = tensor.numeric_value_at(idx) else {
                break;
            };
            let real = real.into_int_value().expect("integer complex real part");
            let imag = imag
                .into_int_value()
                .expect("integer complex imaginary part");
            if !real.is_zero() || !imag.is_zero() {
                if row >= col {
                    lower = lower.max(row - col);
                } else {
                    upper = upper.max(col - row);
                }
            }
        }
    }
    (lower, upper)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerComplexStorage, IntegerStorage, LogicalArray, SparseTensor};

    #[test]
    fn bandwidth_sparse_csc_uses_typed_nonzero_values_and_rectangular_offsets() {
        let double = SparseTensor::new(
            4,
            3,
            vec![0, 2, 3, 4],
            vec![0, 3, 1, 0],
            vec![0.0, f64::NAN, -0.0, 2.0],
        )
        .unwrap();
        assert_eq!(bandwidth_host_sparse(&double), (3, 2));
        let single =
            SparseTensor::new_f32(4, 3, vec![0, 1, 1, 2], vec![3, 0], vec![1.0, -2.0]).unwrap();
        assert_eq!(bandwidth_host_sparse(&single), (3, 2));
        let wide = SparseTensor::new_integer(
            4,
            3,
            vec![0, 1, 1, 2],
            vec![3, 0],
            IntegerStorage::U64(vec![u64::MAX, 0]),
        )
        .unwrap();
        assert_eq!(bandwidth_host_sparse(&wide), (3, 0));
        let complex = SparseTensor::new_complex_f32(
            4,
            3,
            vec![0, 1, 1, 2],
            vec![3, 0],
            vec![(0.0, 2.0), (0.0, 0.0)],
        )
        .unwrap();
        assert_eq!(bandwidth_host_sparse(&complex), (3, 0));
        let logical = SparseTensor::new_logical(4, 3, vec![0, 1, 1, 2], vec![3, 0]).unwrap();
        assert_eq!(bandwidth_host_sparse(&logical), (3, 2));
        assert_eq!(bandwidth_host_sparse(&SparseTensor::zeros(0, 3)), (0, 0));
    }

    #[test]
    fn bandwidth_sparse_public_outputs_and_extension_gate() {
        let sparse =
            SparseTensor::new(3, 3, vec![0, 1, 2, 3], vec![2, 1, 0], vec![1.0; 3]).unwrap();
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert_eq!(
            bandwidth_builtin(Value::SparseTensor(sparse), Vec::new()).unwrap(),
            Value::output_list(vec![Value::Num(2.0), Value::Num(2.0)])
        );
        let integer = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        assert!(bandwidth_builtin(Value::SparseTensor(integer), Vec::new()).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_diagonal_matrix() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let value = Value::Tensor(tensor);
        let result = bandwidth_builtin(value, Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn bandwidth_returns_public_two_output_contract() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 0.0], vec![2, 2]).unwrap();
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert_eq!(
            bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth"),
            Value::output_list(vec![Value::Num(1.0), Value::Num(1.0)])
        );
    }

    #[test]
    fn bandwidth_type_defaults_to_first_scalar_output() {
        let out = bandwidth_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn bandwidth_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = BANDWIDTH_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"[lower, upper] = bandwidth(A)"));
        assert!(labels.contains(&"b = bandwidth(A, selector)"));
    }

    #[test]
    fn bandwidth_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = BANDWIDTH_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.BANDWIDTH.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.BANDWIDTH.INVALID_INPUT"));
        assert!(codes.contains(&"RM.BANDWIDTH.INTERNAL"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_lower_selector() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        let args = vec![Value::from("lower")];
        let result = bandwidth_builtin(Value::Tensor(tensor), args).expect("bandwidth");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_upper_selector() {
        let tensor = Tensor::new(
            vec![1.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let args = vec![Value::from("upper")];
        let result = bandwidth_builtin(Value::Tensor(tensor), args).expect("bandwidth");
        match result {
            Value::Num(n) => assert_eq!(n, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_complex_matrix() {
        let data = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (0.0, 0.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result =
            bandwidth_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_rectangular_matrix() {
        let tensor = Tensor::new(
            vec![0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 7.0, 0.0, 0.0, 10.0],
            vec![4, 3],
        )
        .unwrap();
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_empty_matrix_returns_zero() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 0.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_nan_counts_as_nonzero() {
        let tensor =
            Tensor::new(vec![0.0, f64::NAN, 0.0, 0.0], vec![2, 2]).expect("tensor construction");
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(
            IntegerStorage::I16(vec![1, 0, 7, 0, 2, 0, 0, 5, 3]),
            vec![3, 3],
        )
        .expect("integer matrix");

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn bandwidth_reads_wide_integer_storage_without_the_float_mirror() {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![1, 0, u64::MAX, 1_u64 << 63, 1, 0, 0, 0, 1]),
            vec![3, 3],
        )
        .expect("integer matrix");

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 2.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[test]
    fn bandwidth_accepts_all_integer_classes_in_runmat_mode() {
        let cases = all_integer_bandwidth_storages();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in cases {
            let tensor = Tensor::new_integer(storage, vec![2, 2]).expect("integer matrix");
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            assert_eq!(
                bandwidth_builtin(Value::Tensor(tensor), Vec::new()).expect("bandwidth"),
                Value::output_list(vec![Value::Num(0.0), Value::Num(1.0)])
            );
        }
    }

    fn all_integer_bandwidth_storages() -> [IntegerStorage; 8] {
        [
            IntegerStorage::I8(vec![1, 0, 7, 0]),
            IntegerStorage::I16(vec![1, 0, 7, 0]),
            IntegerStorage::I32(vec![1, 0, 7, 0]),
            IntegerStorage::I64(vec![1, 0, 7, 0]),
            IntegerStorage::U8(vec![1, 0, 7, 0]),
            IntegerStorage::U16(vec![1, 0, 7, 0]),
            IntegerStorage::U32(vec![1, 0, 7, 0]),
            IntegerStorage::U64(vec![1, 0, u64::MAX, 0]),
        ]
    }

    #[test]
    fn bandwidth_integer_and_logical_inputs_are_mode_gated() {
        let integer =
            Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("integer matrix");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = bandwidth_builtin(Value::Tensor(integer), Vec::new())
            .expect_err("integer extension must reject");
        assert_eq!(
            error.identifier(),
            BANDWIDTH_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let error = bandwidth_builtin(Value::Bool(true), Vec::new())
            .expect_err("logical extension must reject");
        assert_eq!(
            error.identifier(),
            BANDWIDTH_LOGICAL_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn bandwidth_reads_complex_integer_structure_exactly() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1, 0, 0, 0]),
            IntegerStorage::U64(vec![0, 0, u64::MAX, 0]),
        )
        .expect("complex integer storage");
        let tensor = ComplexTensor::new_integer(storage, vec![2, 2]).expect("complex matrix");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert_eq!(
            bandwidth_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("bandwidth"),
            Value::output_list(vec![Value::Num(0.0), Value::Num(1.0)])
        );
    }

    #[test]
    fn bandwidth_integer_capability_covers_all_classes() {
        assert_eq!(BANDWIDTH_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            BANDWIDTH_INTEGER_CAPABILITIES[0].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_logical_input_supported() {
        let logical = LogicalArray::new(vec![1, 1, 1, 0], vec![2, 2]).expect("logical array");
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result =
            bandwidth_builtin(Value::LogicalArray(logical), Vec::new()).expect("bandwidth");
        match result {
            Value::Num(value) => assert_eq!(value, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_selector_validation() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err =
            bandwidth_builtin(Value::Tensor(tensor), vec![Value::from("middle")]).unwrap_err();
        assert_eq!(
            err.identifier(),
            BANDWIDTH_ERROR_INVALID_ARGUMENT.identifier
        );
        let message = err.to_string();
        assert!(
            message.contains("lower") && message.contains("upper"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_rejects_higher_dimensions() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 1, 2]).unwrap();
        let err = bandwidth_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("2-D"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bandwidth_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                bandwidth_builtin(Value::GpuTensor(handle), Vec::new()).expect("bandwidth");
            assert_eq!(result, Value::Num(1.0));
        });
    }

    #[test]
    fn bandwidth_gathers_all_resident_integer_classes_exactly() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            for storage in all_integer_bandwidth_storages() {
                let tensor =
                    Tensor::new_integer(storage, vec![2, 2]).expect("resident integer matrix");
                let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
                let _outputs =
                    crate::output_count::push_output_count(Some(2)).expect("output-count scope");
                assert_eq!(
                    bandwidth_builtin(Value::GpuTensor(handle), Vec::new()).expect("bandwidth"),
                    Value::output_list(vec![Value::Num(0.0), Value::Num(1.0)])
                );
            }
        });
    }

    #[test]
    fn bandwidth_resident_integer_gate_precedes_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U64(vec![1, 0, u64::MAX, 0]), vec![2, 2])
                    .expect("resident integer matrix");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = bandwidth_builtin(Value::GpuTensor(handle), Vec::new())
                .expect_err("resident integer extension must reject");
            assert_eq!(
                error.identifier(),
                BANDWIDTH_INTEGER_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn bandwidth_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let tensor = Tensor::new(
            vec![0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 6.0],
            vec![3, 3],
        )
        .unwrap();
        let cpu = super::bandwidth_host_real_tensor(&tensor).expect("cpu bandwidth");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_meta = provider.bandwidth(&handle).expect("provider bandwidth");
        assert_eq!(gpu_meta.lower as usize, cpu.0);
        assert_eq!(gpu_meta.upper as usize, cpu.1);

        let result =
            bandwidth_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("bandwidth");
        assert_eq!(result, Value::Num(cpu.0 as f64));
        drop(handle);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn bandwidth_wgpu_gathers_all_resident_integer_classes_exactly() {
        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_integer_bandwidth_storages() {
            let tensor = Tensor::new_integer(storage, vec![2, 2]).expect("resident integer matrix");
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            assert_eq!(
                bandwidth_builtin(Value::GpuTensor(handle), Vec::new()).expect("bandwidth"),
                Value::output_list(vec![Value::Num(0.0), Value::Num(1.0)])
            );
        }
    }

    fn bandwidth_builtin(matrix: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::bandwidth_builtin(
            None,
            matrix,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
}
