//! MATLAB-compatible `kron` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{CharArray, ComplexTensor, ResolveContext, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

type AlignedShapes = (Vec<usize>, Vec<usize>, Vec<usize>);

fn kron_type(args: &[Type], _context: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    match input {
        Type::Tensor { .. } => Type::tensor(),
        Type::Logical { .. } => Type::logical(),
        Type::Num | Type::Int | Type::Bool => Type::tensor(),
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::kron")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "kron",
    op_kind: GpuOpKind::Custom("kronecker"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("kron")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes entirely on-device when the provider implements `kron`; otherwise the runtime gathers inputs, computes on the host, and re-uploads the result when possible.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::kron")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "kron",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Kronecker products allocate a fresh tensor and terminate fusion graphs.",
};

fn kron_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("kron").build()
}

#[derive(Clone)]
enum KronNumericResult {
    Real(Tensor),
    Complex(ComplexTensor),
}

#[derive(Clone)]
enum KronInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

#[runtime_builtin(
    name = "kron",
    category = "array/shape",
    summary = "Compute the Kronecker (tensor) product of two arrays.",
    keywords = "kron,kronecker product,tensor product,block matrix,gpu",
    accel = "custom",
    type_resolver(kron_type),
    builtin_path = "crate::builtins::array::shape::kron"
)]
async fn kron_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(kron_error("kron: too many input arguments"));
    }

    match (a, b) {
        (Value::GpuTensor(left), Value::GpuTensor(right)) => Ok(kron_gpu_gpu(left, right).await?),
        (Value::GpuTensor(left), right) => Ok(kron_gpu_mixed_left(left, right).await?),
        (left, Value::GpuTensor(right)) => Ok(kron_gpu_mixed_right(left, right).await?),
        (left, right) => Ok(kron_host(left, right)?),
    }
}

async fn kron_gpu_gpu(
    left: GpuTensorHandle,
    right: GpuTensorHandle,
) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(handle) = provider.kron(&left, &right) {
            return Ok(Value::GpuTensor(handle));
        }
    }

    let left_tensor = gpu_helpers::gather_tensor_async(&left).await?;
    let right_tensor = gpu_helpers::gather_tensor_async(&right).await?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let _ = provider.free(&left);
        let _ = provider.free(&right);
    }
    let left_value = tensor::tensor_into_value(left_tensor);
    let right_value = tensor::tensor_into_value(right_tensor);
    let numeric = compute_numeric(left_value, right_value)?;
    finalize_numeric(numeric, true)
}

async fn kron_gpu_mixed_left(left: GpuTensorHandle, right: Value) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(tensor_right) = tensor::value_into_tensor_for("kron", right.clone()) {
            let view = HostTensorView {
                data: &tensor_right.data,
                shape: &tensor_right.shape,
            };
            if let Ok(uploaded) = provider.upload(&view) {
                match provider.kron(&left, &uploaded) {
                    Ok(handle) => {
                        let _ = provider.free(&uploaded);
                        return Ok(Value::GpuTensor(handle));
                    }
                    Err(_) => {
                        let _ = provider.free(&uploaded);
                    }
                }
            }
        }
    }

    let left_tensor = gpu_helpers::gather_tensor_async(&left).await?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let _ = provider.free(&left);
    }
    let left_value = tensor::tensor_into_value(left_tensor);
    let numeric = compute_numeric(left_value, right)?;
    finalize_numeric(numeric, true)
}

async fn kron_gpu_mixed_right(left: Value, right: GpuTensorHandle) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(tensor_left) = tensor::value_into_tensor_for("kron", left.clone()) {
            let view = HostTensorView {
                data: &tensor_left.data,
                shape: &tensor_left.shape,
            };
            if let Ok(uploaded) = provider.upload(&view) {
                match provider.kron(&uploaded, &right) {
                    Ok(handle) => {
                        let _ = provider.free(&uploaded);
                        return Ok(Value::GpuTensor(handle));
                    }
                    Err(_) => {
                        let _ = provider.free(&uploaded);
                    }
                }
            }
        }
    }

    let right_tensor = gpu_helpers::gather_tensor_async(&right).await?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let _ = provider.free(&right);
    }
    let right_value = tensor::tensor_into_value(right_tensor);
    let numeric = compute_numeric(left, right_value)?;
    finalize_numeric(numeric, true)
}

fn kron_host(left: Value, right: Value) -> crate::BuiltinResult<Value> {
    let numeric = compute_numeric(left, right)?;
    finalize_numeric(numeric, false)
}

fn compute_numeric(left: Value, right: Value) -> crate::BuiltinResult<KronNumericResult> {
    let left_input = value_into_kron_input(left)?;
    let right_input = value_into_kron_input(right)?;
    compute_numeric_inputs(left_input, right_input)
}

fn compute_numeric_inputs(
    left: KronInput,
    right: KronInput,
) -> crate::BuiltinResult<KronNumericResult> {
    match (left, right) {
        (KronInput::Real(a), KronInput::Real(b)) => {
            let tensor = kron_tensor(&a, &b)?;
            Ok(KronNumericResult::Real(tensor))
        }
        (KronInput::Complex(a), KronInput::Complex(b)) => {
            let tensor = kron_complex_tensor(&a, &b)?;
            Ok(KronNumericResult::Complex(tensor))
        }
        (KronInput::Real(a), KronInput::Complex(b)) => {
            let complex_a = tensor_to_complex(&a)?;
            let tensor = kron_complex_tensor(&complex_a, &b)?;
            Ok(KronNumericResult::Complex(tensor))
        }
        (KronInput::Complex(a), KronInput::Real(b)) => {
            let complex_b = tensor_to_complex(&b)?;
            let tensor = kron_complex_tensor(&a, &complex_b)?;
            Ok(KronNumericResult::Complex(tensor))
        }
    }
}

fn finalize_numeric(numeric: KronNumericResult, prefer_gpu: bool) -> crate::BuiltinResult<Value> {
    match numeric {
        KronNumericResult::Real(tensor) => {
            if prefer_gpu {
                if let Some(provider) = runmat_accelerate_api::provider() {
                    let view = HostTensorView {
                        data: &tensor.data,
                        shape: &tensor.shape,
                    };
                    if let Ok(handle) = provider.upload(&view) {
                        return Ok(Value::GpuTensor(handle));
                    }
                }
            }
            Ok(tensor::tensor_into_value(tensor))
        }
        KronNumericResult::Complex(tensor) => Ok(complex_tensor_into_value(tensor)),
    }
}

fn value_into_kron_input(value: Value) -> crate::BuiltinResult<KronInput> {
    match value {
        Value::Tensor(tensor) => Ok(KronInput::Real(tensor)),
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map(KronInput::Real)
            .map_err(|e| kron_error(format!("kron: {e}"))),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            tensor::value_into_tensor_for("kron", value)
                .map(KronInput::Real)
                .map_err(|e| kron_error(e))
        }
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map(KronInput::Complex)
            .map_err(|e| kron_error(format!("kron: {e}"))),
        Value::ComplexTensor(tensor) => Ok(KronInput::Complex(tensor)),
        Value::CharArray(chars) => char_array_to_tensor(&chars).map(KronInput::Real),
        other => Err(kron_error(format!(
            "kron: unsupported input type {:?}; expected numeric, logical, or complex values",
            other
        ))),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> crate::BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|e| kron_error(format!("kron: {e}")))
}

fn tensor_to_complex(tensor: &Tensor) -> crate::BuiltinResult<ComplexTensor> {
    let data: Vec<(f64, f64)> = tensor.data.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape.clone()).map_err(|e| kron_error(format!("kron: {e}")))
}

fn kron_tensor(a: &Tensor, b: &Tensor) -> crate::BuiltinResult<Tensor> {
    let (shape_a, shape_b, shape_out) = aligned_shapes(&a.shape, &b.shape)?;
    let total_out = checked_total(&shape_out, "kron")?;
    if total_out == 0 {
        return Tensor::new(Vec::new(), shape_out).map_err(|e| kron_error(format!("kron: {e}")));
    }

    let strides_out = column_major_strides(&shape_out);
    let mut coords_a = vec![0usize; shape_out.len()];
    let mut coords_b = vec![0usize; shape_out.len()];
    let mut data = vec![0.0f64; total_out];

    for (idx_a, &value_a) in a.data.iter().enumerate() {
        unravel_index(idx_a, &shape_a, &mut coords_a);
        for (idx_b, &value_b) in b.data.iter().enumerate() {
            unravel_index(idx_b, &shape_b, &mut coords_b);
            let out_index = combine_indices(&coords_a, &coords_b, &shape_b, &strides_out)?;
            data[out_index] = value_a * value_b;
        }
    }

    Tensor::new(data, shape_out).map_err(|e| kron_error(format!("kron: {e}")))
}

fn kron_complex_tensor(
    a: &ComplexTensor,
    b: &ComplexTensor,
) -> crate::BuiltinResult<ComplexTensor> {
    let (shape_a, shape_b, shape_out) = aligned_shapes(&a.shape, &b.shape)?;
    let total_out = checked_total(&shape_out, "kron")?;
    if total_out == 0 {
        return ComplexTensor::new(Vec::new(), shape_out)
            .map_err(|e| kron_error(format!("kron: {e}")));
    }

    let strides_out = column_major_strides(&shape_out);
    let mut coords_a = vec![0usize; shape_out.len()];
    let mut coords_b = vec![0usize; shape_out.len()];
    let mut data = vec![(0.0f64, 0.0f64); total_out];

    for (idx_a, &(ar, ai)) in a.data.iter().enumerate() {
        unravel_index(idx_a, &shape_a, &mut coords_a);
        for (idx_b, &(br, bi)) in b.data.iter().enumerate() {
            unravel_index(idx_b, &shape_b, &mut coords_b);
            let out_index = combine_indices(&coords_a, &coords_b, &shape_b, &strides_out)?;
            let real = ar * br - ai * bi;
            let imag = ar * bi + ai * br;
            data[out_index] = (real, imag);
        }
    }

    ComplexTensor::new(data, shape_out).map_err(|e| kron_error(format!("kron: {e}")))
}

fn aligned_shapes(shape_a: &[usize], shape_b: &[usize]) -> crate::BuiltinResult<AlignedShapes> {
    let rank = shape_a.len().max(shape_b.len()).max(1);
    let mut padded_a = vec![1usize; rank];
    let mut padded_b = vec![1usize; rank];

    for (idx, &dim) in shape_a.iter().enumerate() {
        padded_a[idx] = dim;
    }
    for (idx, &dim) in shape_b.iter().enumerate() {
        padded_b[idx] = dim;
    }

    let mut output = Vec::with_capacity(rank);
    for i in 0..rank {
        output.push(
            padded_a[i]
                .checked_mul(padded_b[i])
                .ok_or_else(|| kron_error("kron: requested output exceeds maximum size"))?,
        );
    }

    Ok((padded_a, padded_b, output))
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut current = 1usize;
    for &dim in shape {
        strides.push(current);
        current = current.saturating_mul(dim.max(1));
    }
    strides
}

fn unravel_index(mut index: usize, shape: &[usize], coords: &mut [usize]) {
    for (dim_idx, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            coords[dim_idx] = 0;
        } else {
            coords[dim_idx] = index % dim;
            index /= dim;
        }
    }
}

fn combine_indices(
    coords_a: &[usize],
    coords_b: &[usize],
    shape_b: &[usize],
    strides_out: &[usize],
) -> crate::BuiltinResult<usize> {
    let mut index = 0usize;
    for (dim, stride) in strides_out.iter().enumerate() {
        let scaled = coords_a
            .get(dim)
            .copied()
            .unwrap_or(0)
            .checked_mul(shape_b.get(dim).copied().unwrap_or(1))
            .ok_or_else(|| kron_error("kron: index overflow"))?;
        let coord = scaled
            .checked_add(coords_b.get(dim).copied().unwrap_or(0))
            .ok_or_else(|| kron_error("kron: index overflow"))?;
        index = index
            .checked_add(
                coord
                    .checked_mul(*stride)
                    .ok_or_else(|| kron_error("kron: index overflow"))?,
            )
            .ok_or_else(|| kron_error("kron: index overflow"))?;
    }
    Ok(index)
}

fn checked_total(shape: &[usize], context: &str) -> crate::BuiltinResult<usize> {
    let mut total = 1usize;
    for &dim in shape {
        if dim == 0 {
            return Ok(0);
        }
        total = total.checked_mul(dim).ok_or_else(|| {
            kron_error(format!("{context}: requested output exceeds maximum size"))
        })?;
    }
    Ok(total)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn kron_builtin(a: Value, b: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::kron_builtin(a, b, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{LogicalArray, Tensor, Type};

    #[test]
    fn kron_type_logical_returns_logical() {
        let out = kron_type(
            &[Type::Logical { shape: None }, Type::Logical { shape: None }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::logical());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_matrix_product() {
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();
        let result = kron_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4, 4]);
                assert_eq!(
                    t.data,
                    vec![
                        0.0, 6.0, 0.0, 18.0, 5.0, 7.0, 15.0, 21.0, 0.0, 12.0, 0.0, 24.0, 10.0,
                        14.0, 20.0, 28.0
                    ]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_scalar_scaling() {
        let a = Value::Num(3.0);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = kron_builtin(a, Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![3.0, 6.0, 9.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_complex_inputs() {
        let a = Value::Complex(1.0, 2.0);
        let b = Value::ComplexTensor(
            ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0)], vec![2, 1]).unwrap(),
        );
        let result = kron_builtin(a, b, Vec::new()).expect("kron");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 1]);
                assert_eq!(ct.data, vec![(0.0, 0.0), (1.0, 2.0 * 1.0 + 1.0 * 0.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_logical_promotes_to_double() {
        let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let tensor = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let result = kron_builtin(
            Value::LogicalArray(logical),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert_eq!(t.data, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_char_arrays_convert_to_double() {
        let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let result =
            kron_builtin(Value::CharArray(chars), Value::Tensor(tensor), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected: Vec<f64> = "AB"
                    .chars()
                    .flat_map(|ch| [ch as u32 as f64, 2.0 * ch as u32 as f64])
                    .collect();
                assert_eq!(t.data, expected);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_rejects_extra_arguments() {
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = kron_builtin(
            Value::Tensor(a),
            Value::Tensor(b),
            vec![Value::from("extra")],
        )
        .unwrap_err();
        assert!(
            err.to_string().to_ascii_lowercase().contains("too many"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();
            let view_a = HostTensorView {
                data: &a.data,
                shape: &a.shape,
            };
            let view_b = HostTensorView {
                data: &b.data,
                shape: &b.shape,
            };
            let handle_a = provider.upload(&view_a).expect("upload a");
            let handle_b = provider.upload(&view_b).expect("upload b");
            let result = kron_builtin(
                Value::GpuTensor(handle_a),
                Value::GpuTensor(handle_b),
                Vec::new(),
            )
            .expect("kron");
            match &result {
                Value::GpuTensor(_) => {}
                other => panic!("expected GPU tensor, got {other:?}"),
            }
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![4, 4]);
            assert_eq!(
                gathered.data,
                vec![
                    0.0, 6.0, 0.0, 18.0, 5.0, 7.0, 15.0, 21.0, 0.0, 12.0, 0.0, 24.0, 10.0, 14.0,
                    20.0, 28.0
                ]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_mixed_gpu_host_reuploads() {
        test_support::with_test_provider(|provider| {
            let gpu_tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &gpu_tensor.data,
                shape: &gpu_tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let host = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
            let result = kron_builtin(Value::GpuTensor(handle), Value::Tensor(host), Vec::new())
                .expect("kron");
            match result {
                Value::GpuTensor(_) => {}
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn kron_empty_inputs() {
        let a = Tensor::new(Vec::new(), vec![0, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = kron_builtin(Value::Tensor(a), Value::Tensor(b), Vec::new()).expect("kron");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 4]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn kron_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.0, 6.0, 5.0, 7.0], vec![2, 2]).unwrap();

        let cpu_value = kron_builtin(
            Value::Tensor(a.clone()),
            Value::Tensor(b.clone()),
            Vec::new(),
        )
        .expect("cpu");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_a = HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let handle_a = provider.upload(&view_a).expect("upload a");
        let handle_b = provider.upload(&view_b).expect("upload b");

        let gpu_value = kron_builtin(
            Value::GpuTensor(handle_a),
            Value::GpuTensor(handle_b),
            Vec::new(),
        )
        .expect("gpu");
        let gpu_tensor = test_support::gather(gpu_value).expect("gather");

        assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
        for (idx, (g, c)) in gpu_tensor
            .data
            .iter()
            .zip(cpu_tensor.data.iter())
            .enumerate()
        {
            assert!(
                (*g - *c).abs() < 1e-9,
                "mismatch at index {idx}: {g} vs {c}"
            );
        }
    }
}
