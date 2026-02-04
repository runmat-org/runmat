//! MATLAB-compatible `eye` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{ComplexTensor, LogicalArray, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::array::type_resolvers::{
    rank_from_dims_args, tensor_type_from_literal_dims, tensor_type_from_rank,
};
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::eye")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eye",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("eye"),
        ProviderHook::Custom("eye_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates identity tensors on the device when providers expose dedicated hooks; otherwise falls back to uploading a host-constructed identity tensor.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::eye")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eye",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Identity tensors are materialised directly; fusion planner treats eye() as a standalone allocation.",
};

fn eye_type(args: &[Type]) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    let rank = rank_from_dims_args(args);
    tensor_type_from_rank(rank)
}

fn eye_type_with_ctx(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if let Some(ty) = tensor_type_from_literal_dims(args, ctx) {
        return ty;
    }
    eye_type(args)
}

#[runtime_builtin(
    name = "eye",
    category = "array/creation",
    summary = "Identity matrix or N-D identity tensor.",
    keywords = "eye,identity,matrix,gpu,like,logical",
    accel = "array_construct",
    type_resolver(eye_type_with_ctx),
    builtin_path = "crate::builtins::array::creation::eye"
)]
async fn eye_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedEye::parse(rest).await?;
    build_output(parsed).await.map_err(Into::into)
}

struct ParsedEye {
    shape: Vec<usize>,
    template: EyeTemplate,
}

#[derive(Clone)]
enum EyeTemplate {
    Double,
    Logical,
    Like(Value),
}

impl ParsedEye {
    async fn parse(args: Vec<Value>) -> Result<Self, String> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<EyeTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(
                                "eye: multiple 'like' specifications are not supported".to_string()
                            );
                        }
                        if class_override.is_some() {
                            return Err("eye: cannot combine 'like' with other class specifiers"
                                .to_string());
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err("eye: expected prototype after 'like'".to_string());
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err("eye: cannot combine 'like' with 'logical'".to_string());
                        }
                        class_override = Some(EyeTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err("eye: cannot combine 'like' with 'double'".to_string());
                        }
                        class_override = Some(EyeTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        return Err(
                            "eye: single precision output is not implemented yet".to_string()
                        );
                    }
                    other => {
                        return Err(format!("eye: unrecognised option '{other}'"));
                    }
                }
            }

            if let Some(parsed_dims) = extract_dims(&arg).await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        let template = if let Some(proto) = like_proto {
            EyeTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            EyeTemplate::Like(proto)
        } else {
            EyeTemplate::Double
        };

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedEye) -> Result<Value, String> {
    let shape = normalize_shape(&parsed.shape);
    match parsed.template {
        EyeTemplate::Double => eye_double(&shape),
        EyeTemplate::Logical => eye_logical(&shape),
        EyeTemplate::Like(proto) => eye_like(&proto, &shape).await,
    }
}

fn eye_double(shape: &[usize]) -> Result<Value, String> {
    let tensor = identity_tensor(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn eye_logical(shape: &[usize]) -> Result<Value, String> {
    let shape = shape.to_vec();
    let mut logical = LogicalArray::zeros(shape.clone());
    visit_identity_positions(&shape, |idx| logical.data[idx] = 1);
    Ok(Value::LogicalArray(logical))
}

fn eye_complex(shape: &[usize]) -> Result<Value, String> {
    let shape = shape.to_vec();
    let mut tensor = ComplexTensor::zeros(shape.clone());
    visit_identity_positions(&shape, |idx| tensor.data[idx] = (1.0, 0.0));
    Ok(Value::ComplexTensor(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn eye_like(proto: &Value, shape: &[usize]) -> Result<Value, String> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => eye_logical(shape),
        Value::ComplexTensor(_) | Value::Complex(_, _) => eye_complex(shape),
        Value::GpuTensor(handle) => eye_like_gpu(handle, shape).await,
        Value::Tensor(_) | Value::Num(_) | Value::Int(_) => eye_double(shape),
        Value::CharArray(_) | Value::Cell(_) => eye_double(shape),
        other => {
            let gathered = crate::dispatcher::gather_if_needed_async(other)
                .await
                .map_err(|e| format!("eye: {e}"))?;
            eye_like(&gathered, shape).await
        }
    }
}

#[async_recursion::async_recursion(?Send)]
async fn eye_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> Result<Value, String> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let shape_vec = shape.to_vec();
    if let Some(provider) = runmat_accelerate_api::provider() {
        let attempt = if handle.shape == shape_vec {
            provider.eye_like(handle)
        } else {
            provider.eye(&shape_vec)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        if let Ok(host) = identity_tensor(&shape_vec) {
            let view = HostTensorView {
                data: &host.data,
                shape: &host.shape,
            };
            if let Ok(gpu) = provider.upload(&view) {
                return Ok(Value::GpuTensor(gpu));
            }
        }
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| format!("eye: {e}"))?;
    eye_like(&gathered, shape).await
}

fn identity_tensor(shape: &[usize]) -> Result<Tensor, String> {
    let shape_vec = shape.to_vec();
    let total = tensor::element_count(&shape_vec);
    let mut data = vec![0.0; total];
    visit_identity_positions(&shape_vec, |idx| data[idx] = 1.0);
    Tensor::new(data, shape_vec).map_err(|e| format!("eye: {e}"))
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn extract_dims(value: &Value) -> Result<Option<Vec<usize>>, String> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    tensor::dims_from_value_async(value)
        .await
        .map_err(|e| format!("eye: {e}"))
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(t) => Ok(t.shape.clone()),
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("eye: unsupported prototype {other:?}")),
    }
}

fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => {
            let n = shape[0];
            vec![n, n]
        }
        _ => shape.to_vec(),
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn visit_identity_positions(shape: &[usize], mut f: impl FnMut(usize)) {
    if shape.len() < 2 {
        return;
    }
    let rows = shape[0];
    let cols = shape[1];
    let diag_len = rows.min(cols);
    if diag_len == 0 {
        return;
    }
    let strides = compute_strides(shape);
    let extra_dims = &shape[2..];
    let extra_count = if extra_dims.is_empty() {
        1
    } else {
        tensor::element_count(extra_dims)
    };
    let mut coords = vec![0usize; shape.len()];
    for extra_index in 0..extra_count {
        for coord in coords.iter_mut().skip(2) {
            *coord = 0;
        }
        if !extra_dims.is_empty() {
            let mut remainder = extra_index;
            for (offset, size) in extra_dims.iter().copied().enumerate() {
                let dim = offset + 2;
                if size == 0 {
                    coords[dim] = 0;
                } else {
                    coords[dim] = remainder % size;
                    remainder /= size;
                }
            }
        }
        for diag in 0..diag_len {
            coords[0] = diag;
            coords[1] = diag;
            let mut linear = 0usize;
            for (dim, &coord) in coords.iter().enumerate() {
                linear += coord * strides[dim];
            }
            f(linear);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::Type;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_default_scalar() {
        let result = block_on(eye_builtin(Vec::new())).expect("eye");
        assert_eq!(result, Value::Num(1.0));
    }

    #[test]
    fn eye_type_defaults_to_num() {
        assert_eq!(eye_type(&[]), Type::Num);
    }

    #[test]
    fn eye_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            eye_type(&[Type::Num]),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_square_from_scalar_dimension() {
        let args = vec![Value::Num(3.0)];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                for r in 0..3 {
                    for c in 0..3 {
                        let idx = r + c * 3;
                        if r == c {
                            assert_eq!(t.data[idx], 1.0);
                        } else {
                            assert_eq!(t.data[idx], 0.0);
                        }
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_rectangular_from_two_dims() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert_eq!(t.data.len(), 8);
                for r in 0..2 {
                    for c in 0..4 {
                        let idx = r + c * 2;
                        if r == c {
                            assert_eq!(t.data[idx], 1.0);
                        } else {
                            assert_eq!(t.data[idx], 0.0);
                        }
                    }
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_accepts_size_vector_argument() {
        let size_vec = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();
        let result = block_on(eye_builtin(vec![Value::Tensor(size_vec)])).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert_eq!(t.data.len(), 8);
                assert_eq!(t.data[0], 1.0);
                assert_eq!(t.data[1], 0.0);
                assert_eq!(t.data[2], 0.0);
                assert_eq!(t.data[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_zero_dimension_matrix() {
        let result = block_on(eye_builtin(vec![Value::Num(0.0)])).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_uses_tensor_argument_shape_and_type() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor.clone())];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, tensor.shape);
                assert_eq!(t.data, vec![1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_logical_output() {
        let args = vec![Value::Num(4.0), Value::from("logical")];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![4, 4]);
                for r in 0..4 {
                    for c in 0..4 {
                        let idx = r + c * 4;
                        if r == c {
                            assert_eq!(logical.data[idx], 1);
                        } else {
                            assert_eq!(logical.data[idx], 0);
                        }
                    }
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_bool_produces_logical() {
        let args = vec![Value::Num(3.0), Value::from("like"), Value::Bool(true)];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![3, 3]);
                for row in 0..3 {
                    for col in 0..3 {
                        let idx = row + col * 3;
                        if row == col {
                            assert_eq!(logical.data[idx], 1);
                        } else {
                            assert_eq!(logical.data[idx], 0);
                        }
                    }
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_prototype_with_logical_override() {
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let args = vec![Value::Tensor(proto), Value::from("logical")];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 3]);
                assert_eq!(logical.data.len(), 6);
                assert_eq!(&logical.data[..6], &[1, 0, 0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_complex_like() {
        let args = vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data[0], (1.0, 0.0));
                assert_eq!(t.data[1], (0.0, 0.0));
                assert_eq!(t.data[2], (0.0, 0.0));
                assert_eq!(t.data[3], (1.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_extra_dimensions_replicate_identity() {
        let args = vec![Value::Num(2.0), Value::Num(3.0), Value::Num(2.0)];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3, 2]);
                // slice 0
                assert_eq!(t.data[0], 1.0);
                assert_eq!(t.data[1], 0.0);
                assert_eq!(t.data[2], 0.0);
                assert_eq!(t.data[3], 1.0);
                // slice 1 offset = rows * cols = 6
                assert_eq!(t.data[6], 1.0);
                assert_eq!(t.data[7], 0.0);
                assert_eq!(t.data[8], 0.0);
                assert_eq!(t.data[9], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_requires_prototype() {
        assert!(block_on(eye_builtin(vec![Value::from("like")])).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(3.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(eye_builtin(args)).expect("eye");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 3]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert_eq!(gathered.shape, vec![2, 3]);
                    assert_eq!(gathered.data[0], 1.0);
                    assert_eq!(gathered.data[1], 0.0);
                    assert_eq!(gathered.data[2], 0.0);
                    assert_eq!(gathered.data[3], 1.0);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_gpu_prototype_infers_shape() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let prototype = provider.upload(&view).expect("upload proto");
            let result = block_on(eye_builtin(vec![Value::GpuTensor(prototype)])).expect("eye");
            let gathered = test_support::gather(result).expect("gather gpu identity");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 0.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_string_first_argument() {
        let proto = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = block_on(eye_builtin(args)).expect("eye");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.data, vec![1.0, 0.0, 0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_like_and_logical_conflict() {
        let args = vec![
            Value::Num(2.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Num(1.0),
        ];
        assert!(block_on(eye_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_rejects_negative_dimension() {
        let args = vec![Value::Num(-1.0)];
        assert!(block_on(eye_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye_rejects_non_integer_dimension() {
        let args = vec![Value::Num(2.5)];
        assert!(block_on(eye_builtin(args)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eye_wgpu_matches_cpu() {
        let _ =
            wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let cpu_value = block_on(eye_builtin(vec![Value::Num(3.0)])).expect("cpu eye");
        let host_proto = Tensor::zeros(vec![3, 3]);
        let view = HostTensorView {
            data: &host_proto.data,
            shape: &host_proto.shape,
        };
        let gpu_proto = provider.upload(&view).expect("upload proto");

        let gpu_value = block_on(eye_builtin(vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::GpuTensor(gpu_proto),
        ]))
        .expect("gpu eye");

        let gathered = test_support::gather(gpu_value).expect("gather gpu eye");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor for cpu eye, got {other:?}"),
        };

        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }
}
