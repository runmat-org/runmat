//! MATLAB-compatible `sub2ind` builtin with GPU-aware semantics for RunMat.

#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::GpuTensorHandle;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, IntValue, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use super::common::{
    build_strides, dims_from_tokens, fits_positive_platform_index, materialize_value, parse_dims,
};
use crate::builtins::array::type_resolvers::is_scalar_type;
use crate::builtins::common::arg_tokens::tokens_from_context;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::shape_rules::element_count_if_known;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::sub2ind")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sub2ind",
    op_kind: GpuOpKind::Custom("indexing"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("sub2ind")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers can implement the custom `sub2ind` hook to execute on device; runtimes fall back to host computation otherwise.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::indexing::sub2ind")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sub2ind",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Index conversion executes eagerly on the host; fusion does not apply.",
};

fn sub2ind_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.len() < 2 {
        return Type::Unknown;
    }
    if let Some(dims) = dims_from_tokens(&tokens_from_context(ctx)) {
        if args.len() - 1 != dims.len() {
            return Type::Unknown;
        }
    }
    let subscripts = &args[1..];
    if subscripts.iter().all(|ty| is_scalar_type(ty)) {
        return Type::Num;
    }
    for ty in subscripts {
        if let Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } = ty {
            if element_count_if_known(shape).unwrap_or(0) > 1 {
                return Type::Tensor {
                    shape: Some(shape.clone()),
                };
            }
        }
    }
    Type::tensor()
}

const BUILTIN_NAME: &str = "sub2ind";

const SUB2IND_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The size vector accepts every native integer class and is parsed directly as positive platform dimensions.",
    },
    BuiltinIntegerInputCapability {
        name: "I1...In",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every subscript role accepts all eight integer classes, including scalar expansion and full-width exact bounds checks.",
    },
];
pub const SUB2IND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "ind = sub2ind(integer_sz, integer_I1, integer_In...)",
        inputs: &SUB2IND_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Dimensions and subscripts are read from authoritative integer storage, combined with checked column-major index arithmetic, and returned as documented double indices. Resident integer inputs use exact gather fallback and return resident double output when the owner supports upload.",
    }];

const SUB2IND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ind",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column-major linear indices corresponding to provided subscripts.",
}];

const SUB2IND_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size vector describing source array dimensions.",
    },
    BuiltinParamDescriptor {
        name: "I1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First-dimension subscript values.",
    },
    BuiltinParamDescriptor {
        name: "In",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Remaining per-dimension subscript arrays/scalars.",
    },
];

const SUB2IND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ind = sub2ind(sz, I1, In...)",
    inputs: &SUB2IND_INPUTS,
    outputs: &SUB2IND_OUTPUT,
}];

const SUB2IND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.INVALID_INPUT",
    identifier: Some("RunMat:sub2ind:InvalidInput"),
    when: "Size vector, subscript count, or subscript types are invalid.",
    message: "sub2ind: invalid input arguments",
};

const SUB2IND_ERROR_INDEX_BOUNDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.INDEX_BOUNDS",
    identifier: Some("RunMat:sub2ind:IndexBounds"),
    when: "At least one subscript lies outside bounds for its dimension.",
    message: "sub2ind: subscript index exceeds dimension bounds",
};

const SUB2IND_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.PROVIDER",
    identifier: Some("RunMat:sub2ind:ProviderError"),
    when: "GPU provider sub2ind hook fails.",
    message: "sub2ind: provider execution failed",
};

const SUB2IND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUB2IND.INTERNAL",
    identifier: Some("RunMat:sub2ind:InternalError"),
    when: "Internal tensor conversion/output construction fails.",
    message: "sub2ind: internal error",
};

const SUB2IND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    SUB2IND_ERROR_INVALID_INPUT,
    SUB2IND_ERROR_INDEX_BOUNDS,
    SUB2IND_ERROR_PROVIDER,
    SUB2IND_ERROR_INTERNAL,
];

pub const SUB2IND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUB2IND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUB2IND_ERRORS,
};

fn sub2ind_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sub2ind_input_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_INVALID_INPUT)
}

fn sub2ind_bounds_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_INDEX_BOUNDS)
}

fn sub2ind_provider_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_PROVIDER)
}

fn sub2ind_internal_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_error_with_message(message, &SUB2IND_ERROR_INTERNAL)
}

#[runtime_builtin(
    name = "sub2ind",
    category = "array/indexing",
    summary = "Convert N-D subscripts to MATLAB-style column-major linear indices.",
    keywords = "sub2ind,linear index,column major,gpu indexing",
    accel = "custom",
    type_resolver(sub2ind_type),
    descriptor(crate::builtins::array::indexing::sub2ind::SUB2IND_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::indexing::sub2ind::SUB2IND_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::indexing::sub2ind"
)]
async fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (dims_value, dims_was_gpu) = materialize_value(dims_val, "sub2ind").await?;
    let dims = parse_dims(&dims_value, "sub2ind").await?;
    if dims.is_empty() {
        return Err(sub2ind_error("Size vector must have at least one element."));
    }

    if rest.len() != dims.len() {
        return Err(sub2ind_error(
            "The number of subscripts supplied must equal the number of dimensions in the size vector.",
        ));
    }

    if let Some(value) = try_gpu_sub2ind(&dims, &rest)? {
        return Ok(value);
    }

    let mut saw_gpu = dims_was_gpu;
    let mut subscripts: Vec<Tensor> = Vec::with_capacity(rest.len());
    for value in rest {
        let (materialised, was_gpu) = materialize_value(value, "sub2ind").await?;
        saw_gpu |= was_gpu;
        let tensor = tensor::value_into_tensor_for("sub2ind", materialised)
            .map_err(|message| sub2ind_error(message))?;
        subscripts.push(tensor);
    }

    let (result_data, result_shape) = compute_indices(&dims, &subscripts)?;
    let want_gpu_output = saw_gpu && runmat_accelerate_api::provider().is_some();

    if want_gpu_output {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let shape = result_shape.clone().unwrap_or_else(|| vec![1, 1]);
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &result_data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    build_host_value(result_data, result_shape)
}

fn try_gpu_sub2ind(dims: &[usize], subs: &[Value]) -> crate::BuiltinResult<Option<Value>> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (dims, subs);
        Ok(None)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if subs
                .iter()
                .any(|v| matches!(v, Value::GpuTensor(h) if h.device_id != 0))
            {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let provider = match runmat_accelerate_api::provider() {
            Some(p) => p,
            None => return Ok(None),
        };
        if !subs
            .iter()
            .all(|value| matches!(value, Value::GpuTensor(_)))
        {
            return Ok(None);
        }
        if subs.iter().any(
            |value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some()),
        ) {
            return Ok(None);
        }
        if dims.is_empty() {
            return Ok(None);
        }

        let mut handles: Vec<&GpuTensorHandle> = Vec::with_capacity(subs.len());
        for value in subs {
            if let Value::GpuTensor(handle) = value {
                handles.push(handle);
            }
        }

        if handles.len() != dims.len() {
            return Err(sub2ind_error(
            "The number of subscripts supplied must equal the number of dimensions in the size vector.",
        ));
        }

        let mut scalar_mask: Vec<bool> = Vec::with_capacity(handles.len());
        let mut target_shape: Option<Vec<usize>> = None;
        let mut result_len: usize = 1;
        let mut saw_non_scalar = false;

        for handle in &handles {
            let len = tensor::element_count(&handle.shape);
            let is_scalar = len == 1;
            scalar_mask.push(is_scalar);
            if !is_scalar {
                saw_non_scalar = true;
                if let Some(existing) = &target_shape {
                    if existing != &handle.shape {
                        return Err(sub2ind_error("Subscript inputs must have the same size."));
                    }
                } else {
                    target_shape = Some(handle.shape.clone());
                    result_len = len;
                }
            }
        }

        if !saw_non_scalar {
            target_shape = Some(vec![1, 1]);
            result_len = 1;
        } else if let Some(shape) = &target_shape {
            result_len = tensor::element_count(shape);
        }

        let strides = build_strides(dims, "sub2ind")?;
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || result_len > u32::MAX as usize
        {
            return Ok(None);
        }

        let output_shape = target_shape.clone().unwrap_or_else(|| vec![1, 1]);
        match provider.sub2ind(
            dims,
            &strides,
            &handles,
            &scalar_mask,
            result_len,
            &output_shape,
        ) {
            Ok(handle) => Ok(Some(Value::GpuTensor(handle))),
            Err(err) => Err(sub2ind_provider_error(err.to_string())),
        }
    }
}

fn compute_indices(
    dims: &[usize],
    subscripts: &[Tensor],
) -> crate::BuiltinResult<(Vec<f64>, Option<Vec<usize>>)> {
    let mut target_shape: Option<Vec<usize>> = None;
    let mut result_len: usize = 1;
    let mut has_non_scalar = false;

    for tensor in subscripts {
        let len = tensor_element_len(tensor);
        if len != 1 {
            has_non_scalar = true;
            if let Some(shape) = &target_shape {
                if &tensor.shape != shape {
                    return Err(sub2ind_error("Subscript inputs must have the same size."));
                }
            } else {
                target_shape = Some(tensor.shape.clone());
                result_len = len;
            }
        }
    }

    if !has_non_scalar {
        // All scalars -> scalar output
        target_shape = Some(vec![1, 1]);
        result_len = 1;
    }

    if result_len == 0 {
        return Ok((Vec::new(), target_shape));
    }

    let strides = build_strides(dims, "sub2ind")?;
    let mut output = Vec::with_capacity(result_len);

    for idx in 0..result_len {
        let mut offset: usize = 0;
        for (dim_index, (&dim, tensor)) in dims.iter().zip(subscripts.iter()).enumerate() {
            let raw = subscript_value(tensor, idx);
            let coerced = coerce_subscript_value(raw, dim_index + 1, dim)?;
            let term = coerced
                .checked_sub(1)
                .and_then(|v| v.checked_mul(strides[dim_index]))
                .ok_or_else(|| sub2ind_bounds_error("Index exceeds array dimensions."))?;
            offset = offset
                .checked_add(term)
                .ok_or_else(|| sub2ind_bounds_error("Index exceeds array dimensions."))?;
        }
        output.push((offset + 1) as f64);
    }

    Ok((output, target_shape))
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

enum SubscriptValue {
    Float(f64),
    Integer(IntValue),
}

fn subscript_value(tensor: &Tensor, idx: usize) -> SubscriptValue {
    if let Some(storage) = tensor.integer_storage() {
        let index = if storage.len() == 1 { 0 } else { idx };
        return SubscriptValue::Integer(
            storage
                .value_at(index)
                .expect("subscript index is within integer storage bounds"),
        );
    }
    if tensor::is_scalar_tensor(tensor) {
        SubscriptValue::Float(tensor::tensor_value_f64(tensor, 0))
    } else {
        SubscriptValue::Float(tensor::tensor_value_f64(tensor, idx))
    }
}

fn coerce_subscript_value(
    value: SubscriptValue,
    dim_number: usize,
    dim_size: usize,
) -> crate::BuiltinResult<usize> {
    match value {
        SubscriptValue::Float(value) => coerce_subscript(value, dim_number, dim_size),
        SubscriptValue::Integer(value) => coerce_integer_subscript(&value, dim_number, dim_size),
    }
}

fn coerce_integer_subscript(
    value: &IntValue,
    dim_number: usize,
    dim_size: usize,
) -> crate::BuiltinResult<usize> {
    let Some(index) = value.try_to_usize() else {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    };
    if index < 1 {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    if index > dim_size {
        return Err(dimension_bounds_error(dim_number));
    }
    Ok(index)
}

fn coerce_subscript(value: f64, dim_number: usize, dim_size: usize) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    if rounded < 1.0 {
        return Err(sub2ind_error(
            "Subscript indices must either be real positive integers or logicals.",
        ));
    }
    if !fits_positive_platform_index(rounded) {
        return Err(sub2ind_error(
            "Subscript indices exceed the maximum supported index range.",
        ));
    }
    if rounded > dim_size as f64 {
        return Err(dimension_bounds_error(dim_number));
    }
    Ok(rounded as usize)
}

fn dimension_bounds_error(dim_number: usize) -> RuntimeError {
    let message = match dim_number {
        1 => format!("Index exceeds the number of rows in dimension {dim_number}."),
        2 => format!("Index exceeds the number of columns in dimension {dim_number}."),
        3 => format!("Index exceeds the number of pages in dimension {dim_number}."),
        _ => "Index exceeds array dimensions.".to_string(),
    };
    sub2ind_bounds_error(message)
}

fn build_host_value(data: Vec<f64>, shape: Option<Vec<usize>>) -> crate::BuiltinResult<Value> {
    let shape = shape.unwrap_or_else(|| vec![1, 1]);
    if data.len() == 1 && tensor::element_count(&shape) == 1 {
        Ok(Value::Num(data[0]))
    } else {
        let tensor = Tensor::new(data, shape).map_err(|e| {
            sub2ind_internal_error(format!("Unable to construct sub2ind output: {e}"))
        })?;
        Ok(Value::Tensor(tensor))
    }
}

fn sub2ind_error(message: impl Into<String>) -> RuntimeError {
    sub2ind_input_error(message)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor, Type, Value};

    fn sub2ind_builtin(dims_val: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::sub2ind_builtin(dims_val, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn converts_scalar_indices() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result =
            sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(2.0), Value::Num(3.0)]).unwrap();
        assert_eq!(result, Value::Num(8.0));
    }

    #[test]
    fn sub2ind_type_scalar_outputs_num() {
        assert_eq!(
            sub2ind_type(
                &[Type::Tensor { shape: None }, Type::Num, Type::Int],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Num
        );
    }

    #[test]
    fn sub2ind_type_vector_outputs_tensor() {
        let subs = Type::Tensor {
            shape: Some(vec![Some(3), Some(1)]),
        };
        assert_eq!(
            sub2ind_type(
                &[Type::Tensor { shape: None }, subs.clone(), Type::Num],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn broadcasts_scalars_over_vectors() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Num(4.0)],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![10.0, 11.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn sub2ind_typed_integer_subscripts_read_exact_storage() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
        let cols = Tensor::new_integer(IntegerStorage::I16(vec![4]), vec![1, 1]).unwrap();

        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Tensor(cols)],
        )
        .unwrap();

        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 1]);
                assert_eq!(tensor.materialize_f64(), vec![10.0, 11.0, 12.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_three_dimensions() {
        let dims = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let row = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let col = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let page = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(row), Value::Tensor(col), Value::Tensor(page)],
        )
        .unwrap();
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 2]);
                assert_eq!(t.materialize_f64(), vec![3.0, 11.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_out_of_range_subscripts() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(4.0), Value::Num(1.0)])
            .unwrap_err();
        assert!(
            err.to_string().contains("Index exceeds"),
            "expected index bounds error, got {err}"
        );
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INDEX_BOUNDS.identifier
        );
    }

    #[test]
    fn sub2ind_accepts_every_integer_class_and_returns_double() {
        for prototype in [
            IntegerStorage::I8(Vec::new()),
            IntegerStorage::I16(Vec::new()),
            IntegerStorage::I32(Vec::new()),
            IntegerStorage::I64(Vec::new()),
            IntegerStorage::U8(Vec::new()),
            IntegerStorage::U16(Vec::new()),
            IntegerStorage::U32(Vec::new()),
            IntegerStorage::U64(Vec::new()),
        ] {
            let typed = |values: &[i8], shape| {
                let values = values
                    .iter()
                    .map(|value| prototype.cast_exact_assignment(&IntValue::I8(*value)))
                    .collect();
                Tensor::new_integer(
                    prototype
                        .from_same_class_values(values)
                        .expect("same-class values"),
                    shape,
                )
                .expect("typed tensor")
            };
            let result = sub2ind_builtin(
                Value::Tensor(typed(&[2, 3], vec![1, 2])),
                vec![
                    Value::Tensor(typed(&[1, 2], vec![1, 2])),
                    Value::Tensor(typed(&[3], vec![1, 1])),
                ],
            )
            .expect("sub2ind");
            let Value::Tensor(result) = result else {
                panic!("expected double tensor output");
            };
            assert_eq!(result.shape, vec![1, 2]);
            assert_eq!(result.materialize_f64(), vec![5.0, 6.0]);
            assert!(result.integer_storage().is_none());
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_shape_mismatch() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let cols = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::Tensor(rows), Value::Tensor(cols)],
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("same size"),
            "expected size mismatch error, got {err}"
        );
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_integer_subscripts() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = sub2ind_builtin(Value::Tensor(dims), vec![Value::Num(1.5), Value::Num(1.0)])
            .unwrap_err();
        assert!(
            err.to_string().contains("real positive integers"),
            "expected integer coercion error, got {err}"
        );
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );
    }

    #[test]
    fn rejects_oversized_float_subscripts_before_casting() {
        let dims = Value::Int(IntValue::U64(usize::MAX.saturating_sub(1) as u64));

        let err = sub2ind_builtin(dims.clone(), vec![Value::Num(1.0e300)])
            .expect_err("huge float subscript must reject");
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );

        let err = sub2ind_builtin(dims, vec![Value::Num(usize::MAX as f64)])
            .expect_err("platform boundary float subscript must reject");
        assert_eq!(
            err.identifier(),
            super::SUB2IND_ERROR_INVALID_INPUT.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accepts_integer_value_variants() {
        let dims = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
        let result = sub2ind_builtin(dims, vec![Value::Int(IntValue::I32(2))]).expect("sub2ind");
        assert_eq!(result, Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sub2ind_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
            let rows = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let cols = Tensor::new(vec![4.0, 4.0, 4.0], vec![3, 1]).unwrap();

            let dims_handle = provider
                .upload(&HostTensorView {
                    data: &dims.materialize_f64(),
                    shape: &dims.shape,
                })
                .expect("upload dims");
            let rows_handle = provider
                .upload(&HostTensorView {
                    data: &rows.materialize_f64(),
                    shape: &rows.shape,
                })
                .expect("upload rows");
            let cols_handle = provider
                .upload(&HostTensorView {
                    data: &cols.materialize_f64(),
                    shape: &cols.shape,
                })
                .expect("upload cols");

            let result = sub2ind_builtin(
                Value::GpuTensor(dims_handle),
                vec![Value::GpuTensor(rows_handle), Value::GpuTensor(cols_handle)],
            )
            .expect("sub2ind");

            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).unwrap();
                    assert_eq!(gathered.shape, vec![3, 1]);
                    assert_eq!(gathered.materialize_f64(), vec![10.0, 11.0, 12.0]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn sub2ind_integer_gpu_inputs_fall_back_exactly_to_resident_double() {
        test_support::with_test_provider(|provider| {
            let dims =
                Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2]).expect("dims");
            let rows = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[1, 2]),
                    shape: &[1, 2],
                })
                .expect("rows");
            let cols = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[3]),
                    shape: &[1, 1],
                })
                .expect("cols");
            let result = sub2ind_builtin(
                Value::Tensor(dims),
                vec![Value::GpuTensor(rows), Value::GpuTensor(cols)],
            )
            .expect("sub2ind");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident double output");
            };
            assert_eq!(runmat_accelerate_api::handle_integer_type(handle), None);
            let result = test_support::gather(result).expect("gather result");
            assert_eq!(result.shape, vec![1, 2]);
            assert_eq!(result.materialize_f64(), vec![5.0, 6.0]);
            assert!(result.integer_storage().is_none());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn sub2ind_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            panic!("wgpu provider not available");
        };

        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let rows = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let cols = Tensor::new(vec![4.0, 4.0, 4.0], vec![3, 1]).unwrap();

        let cpu = sub2ind_builtin(
            Value::Tensor(dims.clone()),
            vec![Value::Tensor(rows.clone()), Value::Tensor(cols.clone())],
        )
        .expect("cpu sub2ind");

        let rows_handle = provider
            .upload(&HostTensorView {
                data: &rows.materialize_f64(),
                shape: &rows.shape,
            })
            .expect("upload rows");
        let cols_handle = provider
            .upload(&HostTensorView {
                data: &cols.materialize_f64(),
                shape: &cols.shape,
            })
            .expect("upload cols");

        let result = sub2ind_builtin(
            Value::Tensor(dims),
            vec![Value::GpuTensor(rows_handle), Value::GpuTensor(cols_handle)],
        )
        .expect("wgpu sub2ind");

        let gathered = test_support::gather(result).expect("gather");
        let expected = match cpu {
            Value::Tensor(t) => t,
            Value::Num(v) => Tensor::new(vec![v], vec![1, 1]).unwrap(),
            other => panic!("unexpected cpu result {other:?}"),
        };
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }
}
