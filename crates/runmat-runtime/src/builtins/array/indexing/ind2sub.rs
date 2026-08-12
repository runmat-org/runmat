//! MATLAB-compatible `ind2sub` builtin with GPU-aware semantics for RunMat.

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
    total_elements,
};
use crate::builtins::array::type_resolvers::size_vector_len;
use crate::builtins::common::arg_tokens::tokens_from_context;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, make_cell, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::indexing::ind2sub")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ind2sub",
    op_kind: GpuOpKind::Custom("indexing"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ind2sub")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "A binary64 WGPU provider executes admitted `ind2sub` calls entirely on-device. Kernel or adapter limits and other providers use exact host fallback; results return to the source owner only when that owner can truthfully represent double output.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::indexing::ind2sub")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ind2sub",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Index conversion is eager and does not participate in fusion today.",
};

fn ind2sub_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let Some(dims) = args.first() else {
        return Type::Unknown;
    };
    let length = dims_from_tokens(&tokens_from_context(ctx))
        .map(|values| values.len())
        .or_else(|| size_vector_len(dims));
    Type::Cell {
        element_type: Some(Box::new(Type::tensor())),
        length,
    }
}

const BUILTIN_NAME: &str = "ind2sub";

const IND2SUB_SIZE_CLASSES: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "size vector",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The size vector accepts single, double, and every integer class, must contain at least two positive integral elements, and is read exactly from authoritative storage.",
    },
    BuiltinIntegerInputCapability {
        name: "linear indices",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Linear indices accept single, double, and every integer class; outputs are double and preserve the index array shape.",
    },
];

pub const IND2SUB_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[I1,...,In] = ind2sub(integer_size, integer_indices)",
        inputs: &IND2SUB_SIZE_CLASSES,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Requested output count controls trailing-dimension collapse; indices beyond prod(sz) expand the final effective dimension. Resident inputs always produce double output and return to the exact source owner only when that owner can physically preserve binary64.",
    }];

const IND2SUB_OUTPUT_CELL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "subs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array containing one subscript output per dimension.",
}];

const IND2SUB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size vector describing source array dimensions.",
    },
    BuiltinParamDescriptor {
        name: "ind",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear indices to convert into per-dimension subscripts.",
    },
];

const IND2SUB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "subs = ind2sub(sz, ind)",
    inputs: &IND2SUB_INPUTS,
    outputs: &IND2SUB_OUTPUT_CELL,
}];

const IND2SUB_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2SUB.INVALID_INPUT",
    identifier: Some("RunMat:ind2sub:InvalidInput"),
    when: "Size vector or linear index inputs are malformed or unsupported.",
    message: "ind2sub: invalid input arguments",
};

const IND2SUB_ERROR_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2SUB.PROVIDER",
    identifier: Some("RunMat:ind2sub:ProviderError"),
    when: "Provider-side ind2sub execution fails or returns malformed outputs.",
    message: "ind2sub: provider execution failed",
};

const IND2SUB_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2SUB.INTERNAL",
    identifier: Some("RunMat:ind2sub:InternalError"),
    when: "Internal tensor/materialization logic fails while building outputs.",
    message: "ind2sub: internal error",
};

const IND2SUB_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IND2SUB_ERROR_INVALID_INPUT,
    IND2SUB_ERROR_PROVIDER,
    IND2SUB_ERROR_INTERNAL,
];

pub const IND2SUB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IND2SUB_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2SUB_ERRORS,
};

fn ind2sub_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ind2sub_input_error(message: impl Into<String>) -> RuntimeError {
    ind2sub_error_with_message(message, &IND2SUB_ERROR_INVALID_INPUT)
}

fn ind2sub_internal_error(message: impl Into<String>) -> RuntimeError {
    ind2sub_error_with_message(message, &IND2SUB_ERROR_INTERNAL)
}

fn ind2sub_provider_error(message: impl Into<String>) -> RuntimeError {
    ind2sub_error_with_message(message, &IND2SUB_ERROR_PROVIDER)
}

#[runtime_builtin(
    name = "ind2sub",
    category = "array/indexing",
    summary = "Convert linear indices to subscripts.",
    keywords = "ind2sub,linear index,subscripts,column major,gpu indexing",
    accel = "custom",
    type_resolver(ind2sub_type),
    descriptor(crate::builtins::array::indexing::ind2sub::IND2SUB_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::indexing::ind2sub::IND2SUB_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::indexing::ind2sub"
)]
async fn ind2sub_builtin(dims_val: Value, indices_val: Value) -> crate::BuiltinResult<Value> {
    let output_source = match (&indices_val, &dims_val) {
        (Value::GpuTensor(handle), _) => Some(handle.clone()),
        (_, Value::GpuTensor(handle)) => Some(handle.clone()),
        _ => None,
    };
    let (dims_value, dims_was_gpu) = materialize_value(dims_val, "ind2sub").await?;
    let dims = parse_dims(&dims_value, "ind2sub").await?;
    if dims.len() < 2 {
        return Err(ind2sub_error(
            "Size vector must have at least two elements.",
        ));
    }

    let requested_outputs = crate::output_count::current_output_count().unwrap_or(dims.len());
    if requested_outputs == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }
    let effective_dims = effective_output_dims(&dims, requested_outputs)?;
    let total = total_elements(&effective_dims, "ind2sub")?;
    let strides = build_strides(&effective_dims, "ind2sub")?;

    if let Some(result) = try_gpu_ind2sub(&effective_dims, &strides, total, &indices_val)? {
        return Ok(result);
    }

    let (indices_value, indices_was_gpu) = materialize_value(indices_val, "ind2sub").await?;
    let indices_tensor = tensor::value_into_tensor_for("ind2sub", indices_value)
        .map_err(|message| ind2sub_error(message))?;

    let subscripts = compute_subscripts(&effective_dims, &strides, &indices_tensor)?;

    let host_tensors = subscripts;
    let outputs = if dims_was_gpu || indices_was_gpu {
        restore_host_outputs_transactionally(output_source.as_ref(), &host_tensors)?
    } else {
        host_tensors
            .into_iter()
            .map(tensor::tensor_into_value)
            .collect()
    };

    finish_outputs(outputs)
}

fn restore_host_outputs_transactionally(
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    host_tensors: &[Tensor],
) -> crate::BuiltinResult<Vec<Value>> {
    let host_outputs = || {
        host_tensors
            .iter()
            .cloned()
            .map(tensor::tensor_into_value)
            .collect::<Vec<_>>()
    };
    let Some(source) = source else {
        return Ok(host_outputs());
    };
    let mut restored = Vec::with_capacity(host_tensors.len());
    for tensor in host_tensors {
        match gpu_helpers::restore_class_preserving_value(
            source,
            Value::Tensor(tensor.clone()),
            BUILTIN_NAME,
        ) {
            Ok(Value::GpuTensor(handle)) => restored.push(Value::GpuTensor(handle)),
            Ok(_) => {
                free_restored_outputs(&restored, source);
                return Ok(host_outputs());
            }
            Err(error) => {
                free_restored_outputs(&restored, source);
                return Err(error);
            }
        }
    }
    Ok(restored)
}

fn free_restored_outputs(outputs: &[Value], source: &runmat_accelerate_api::GpuTensorHandle) {
    for output in outputs {
        if let Value::GpuTensor(handle) = output {
            gpu_helpers::free_unprotected_exact_owner(handle, &[source]);
        }
    }
}

fn effective_output_dims(
    dims: &[usize],
    requested_outputs: usize,
) -> crate::BuiltinResult<Vec<usize>> {
    if requested_outputs == 0 {
        return Ok(Vec::new());
    }
    if requested_outputs >= dims.len() {
        let mut effective = dims.to_vec();
        effective.resize(requested_outputs, 1);
        return Ok(effective);
    }
    let mut effective = dims[..requested_outputs.saturating_sub(1)].to_vec();
    let trailing = dims[requested_outputs.saturating_sub(1)..]
        .iter()
        .try_fold(1usize, |product, dim| product.checked_mul(*dim))
        .ok_or_else(|| {
            ind2sub_input_error("ind2sub: collapsed dimensions exceed platform limits")
        })?;
    effective.push(trailing);
    Ok(effective)
}

fn finish_outputs(outputs: Vec<Value>) -> crate::BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some() {
        return Ok(Value::OutputList(outputs));
    }
    let len = outputs.len();
    make_cell(outputs, 1, len).map_err(|message| ind2sub_error(message))
}

fn try_gpu_ind2sub(
    dims: &[usize],
    strides: &[usize],
    total: usize,
    indices: &Value,
) -> crate::BuiltinResult<Option<Value>> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (dims, strides, total, indices);
        Ok(None)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if let Value::GpuTensor(h) = indices {
                if h.device_id != 0 {
                    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                    );
                }
            }
        }
        let handle = match indices {
            Value::GpuTensor(handle) => handle,
            _ => return Ok(None),
        };
        let provider = match gpu_helpers::exact_provider_for_handle(handle) {
            Some(provider) => provider,
            None => {
                return Err(ind2sub_provider_error(
                    "ind2sub: no acceleration provider owns the index handle.",
                ))
            }
        };
        if !provider.supports_ind2sub() {
            return Ok(None);
        }
        if runmat_accelerate_api::handle_integer_type(handle).is_some() {
            return Ok(None);
        }
        let expected_precision = provider.precision();
        if expected_precision != runmat_accelerate_api::ProviderPrecision::F64 {
            return Ok(None);
        }
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle) != Some(expected_precision)
            || !gpu_helpers::gpu_class_metadata_matches(
                handle,
                Some(expected_precision),
                None,
                false,
            )
        {
            return Err(ind2sub_provider_error(
                "ind2sub: index handle metadata contradicts its provider payload.",
            ));
        }
        if dims.len() != strides.len() {
            return Err(ind2sub_error("Size vector must have at least one element."));
        }
        if dims.iter().any(|&d| d > u32::MAX as usize)
            || strides.iter().any(|&s| s > u32::MAX as usize)
            || total > u32::MAX as usize
        {
            return Ok(None);
        }
        let len = if handle.shape.is_empty() {
            1usize
        } else {
            handle.shape.iter().copied().product()
        };
        if total == 0 && len > 0 {
            return Err(ind2sub_error(
                "Index exceeds number of array elements. Index must not exceed 0.",
            ));
        }
        if len > u32::MAX as usize {
            return Ok(None);
        }
        let output_shape = if handle.shape.is_empty() {
            vec![len, 1]
        } else {
            handle.shape.clone()
        };
        let source_metadata = gpu_helpers::snapshot_handle_metadata(handle);
        let result = provider.ind2sub(dims, strides, handle, total, len, &output_shape);
        gpu_helpers::restore_handle_metadata(handle, &source_metadata);
        match result {
            Ok(handles) => {
                if handles.len() != dims.len() {
                    for output in &handles {
                        gpu_helpers::free_unprotected_exact_owner(output, &[handle]);
                    }
                    return Err(ind2sub_provider_error(
                        "ind2sub: provider returned an unexpected number of outputs.",
                    ));
                }
                let valid = handles.iter().enumerate().all(|(index, output)| {
                    output.shape == output_shape
                        && output.device_id == provider.device_id()
                        && gpu_helpers::exact_provider_for_handle(output)
                            .is_some_and(|owner| std::ptr::eq(owner, provider))
                        && runmat_accelerate_api::handle_storage(output)
                            == runmat_accelerate_api::GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_precision(output)
                            == Some(expected_precision)
                        && runmat_accelerate_api::handle_integer_type(output).is_none()
                        && !runmat_accelerate_api::handle_is_logical(output)
                        && gpu_helpers::gpu_class_metadata_matches(
                            output,
                            Some(expected_precision),
                            None,
                            false,
                        )
                        && !gpu_helpers::same_gpu_handle(output, handle)
                        && handles[..index]
                            .iter()
                            .all(|prior| !gpu_helpers::same_gpu_handle(output, prior))
                });
                if !valid {
                    for output in &handles {
                        gpu_helpers::free_unprotected_exact_owner(output, &[handle]);
                    }
                    return Err(ind2sub_provider_error(
                        "ind2sub: provider returned an invalid output handle.",
                    ));
                }
                let provenance = runmat_accelerate_api::handle_provenance(handle)
                    .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
                for output in &handles {
                    runmat_accelerate_api::set_handle_provenance(output, provenance);
                    runmat_accelerate_api::mark_residency(output);
                }
                let values: Vec<Value> = handles.into_iter().map(Value::GpuTensor).collect();
                finish_outputs(values).map(Some)
            }
            Err(err) => {
                let message = err.to_string();
                if message.contains("GPU kernel limits")
                    || message.contains("storage buffers")
                    || message.contains("bind group entries")
                {
                    Ok(None)
                } else {
                    Err(ind2sub_provider_error(message))
                }
            }
        }
    }
}

fn compute_subscripts(
    dims: &[usize],
    strides: &[usize],
    indices: &Tensor,
) -> crate::BuiltinResult<Vec<Tensor>> {
    if strides.len() != dims.len() {
        return Err(ind2sub_error("Size vector must have at least one element."));
    }

    let len = tensor::tensor_element_len(indices);
    let mut outputs: Vec<Vec<f64>> = dims.iter().map(|_| Vec::with_capacity(len)).collect();

    for value_index in 0..len {
        let idx = coerce_linear_index_value(linear_index_value(indices, value_index))?;
        let zero_based = idx - 1;
        for (dim_index, (&dim, &stride)) in dims.iter().zip(strides.iter()).enumerate() {
            let coord = if dim_index + 1 == dims.len() {
                (zero_based / stride) + 1
            } else {
                ((zero_based / stride) % dim) + 1
            };
            outputs[dim_index].push(coord as f64);
        }
    }

    let output_shape = if indices.shape.is_empty() {
        vec![len, 1]
    } else {
        indices.shape.clone()
    };

    let mut tensors = Vec::with_capacity(dims.len());
    for data in outputs {
        let tensor = Tensor::new(data, output_shape.clone())
            .map_err(|e| ind2sub_internal_error(format!("ind2sub: {e}")))?;
        tensors.push(tensor);
    }
    Ok(tensors)
}

enum LinearIndexValue {
    Float(f64),
    Integer(IntValue),
}

fn linear_index_value(indices: &Tensor, value_index: usize) -> LinearIndexValue {
    if let Some(storage) = indices.integer_storage() {
        return LinearIndexValue::Integer(
            storage
                .value_at(value_index)
                .expect("linear index is within integer storage bounds"),
        );
    }
    LinearIndexValue::Float(tensor::tensor_value_f64(indices, value_index))
}

fn coerce_linear_index_value(value: LinearIndexValue) -> crate::BuiltinResult<usize> {
    match value {
        LinearIndexValue::Float(value) => coerce_linear_index(value),
        LinearIndexValue::Integer(value) => coerce_integer_linear_index(&value),
    }
}

fn coerce_integer_linear_index(value: &IntValue) -> crate::BuiltinResult<usize> {
    let Some(coerced) = value.try_to_usize() else {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    };
    if coerced < 1 {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    Ok(coerced)
}

fn coerce_linear_index(value: f64) -> crate::BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    if rounded < 1.0 {
        return Err(ind2sub_error("Linear indices must be positive integers."));
    }
    if !fits_positive_platform_index(rounded) {
        return Err(ind2sub_error(
            "Index exceeds maximum supported size for this platform.",
        ));
    }
    let coerced = rounded as usize;
    Ok(coerced)
}

fn ind2sub_error(message: impl Into<String>) -> RuntimeError {
    ind2sub_input_error(message)
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, IntegerStorage, ResolveContext, Tensor, Type, Value};

    fn ind2sub_builtin(dims_val: Value, indices_val: Value) -> crate::BuiltinResult<Value> {
        block_on(super::ind2sub_builtin(dims_val, indices_val))
    }

    fn cell_to_vec(cell: &runmat_builtins::CellArray) -> Vec<Value> {
        cell.data.clone()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn recovers_tensor_indices() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = ind2sub_builtin(Value::Tensor(dims), Value::Num(8.0)).unwrap();
        match result {
            Value::Cell(cell) => {
                let values = cell_to_vec(&cell);
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], Value::Num(2.0));
                assert_eq!(values[1], Value::Num(3.0));
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[test]
    fn ind2sub_type_infers_cell_length() {
        let dims = Type::Tensor {
            shape: Some(vec![Some(1), Some(3)]),
        };
        assert_eq!(
            super::ind2sub_type(&[dims, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Cell {
                element_type: Some(Box::new(Type::tensor())),
                length: Some(3)
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handles_vector_indices() {
        let dims = Tensor::new(vec![3.0, 5.0], vec![1, 2]).unwrap();
        let idx = Tensor::new(vec![7.0, 8.0, 9.0], vec![1, 3]).unwrap();
        let result =
            ind2sub_builtin(Value::Tensor(dims), Value::Tensor(idx)).expect("ind2sub result");
        match result {
            Value::Cell(cell) => {
                let values = cell_to_vec(&cell);
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![1, 3]);
                        assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
                    }
                    other => panic!("expected tensor rows, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![1, 3]);
                        assert_eq!(t.materialize_f64(), vec![3.0, 3.0, 3.0]);
                    }
                    other => panic!("expected tensor cols, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_integer_linear_index_identifier() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err = ind2sub_builtin(Value::Tensor(dims), Value::Num(1.25))
            .expect_err("expected non-integer index error");
        assert_eq!(
            err.identifier(),
            super::IND2SUB_ERROR_INVALID_INPUT.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn indices_beyond_nominal_size_expand_the_final_dimension() {
        let dims = Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap();
        let result = ind2sub_builtin(Value::Tensor(dims), Value::Num(9.0)).expect("expanded index");
        let Value::Cell(result) = result else {
            panic!("expected subscript outputs");
        };
        assert_eq!(result.data, vec![Value::Num(1.0), Value::Num(5.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn recovers_three_dimensional_indices() {
        let dims = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let idx = Tensor::new(vec![3.0, 11.0], vec![1, 2]).unwrap();
        let result =
            ind2sub_builtin(Value::Tensor(dims), Value::Tensor(idx)).expect("ind2sub result");
        if let Value::Cell(cell) = result {
            let values = cell_to_vec(&cell);
            assert_eq!(values.len(), 3);
            assert_eq!(
                values[0],
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap())
            );
            assert_eq!(
                values[1],
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap())
            );
            assert_eq!(
                values[2],
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap())
            );
        } else {
            panic!("expected cell output");
        }
    }

    #[test]
    fn requested_outputs_collapse_trailing_dimensions_and_extend_with_singletons() {
        let dims = Value::Tensor(Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap());
        let indices = Value::Tensor(Tensor::new(vec![3.0, 11.0], vec![1, 2]).unwrap());
        {
            let _outputs = crate::output_count::push_output_count(Some(2));
            let result = ind2sub_builtin(dims.clone(), indices.clone()).expect("two outputs");
            let Value::OutputList(outputs) = result else {
                panic!("expected output list");
            };
            assert_eq!(outputs.len(), 2);
            assert_eq!(
                outputs[0],
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap())
            );
            assert_eq!(
                outputs[1],
                Value::Tensor(Tensor::new(vec![2.0, 6.0], vec![1, 2]).unwrap())
            );
        }
        {
            let _outputs = crate::output_count::push_output_count(Some(4));
            let result = ind2sub_builtin(dims, indices).expect("four outputs");
            let Value::OutputList(outputs) = result else {
                panic!("expected output list");
            };
            assert_eq!(outputs.len(), 4);
            assert_eq!(
                outputs[3],
                Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap())
            );
        }
    }

    #[test]
    fn scalar_size_vector_is_rejected_by_current_contract() {
        let error =
            ind2sub_builtin(Value::Num(4.0), Value::Num(1.0)).expect_err("scalar size must reject");
        assert_eq!(error.identifier(), Some("RunMat:ind2sub:InvalidInput"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ind2sub_linear_indices_read_typed_integer_storage_exactly() {
        let dims = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let idx =
            Tensor::new_integer(IntegerStorage::U16(vec![3, 11]), vec![1, 2]).expect("indices");

        let result =
            ind2sub_builtin(Value::Tensor(dims), Value::Tensor(idx)).expect("ind2sub result");
        let Value::Cell(cell) = result else {
            panic!("expected cell output");
        };
        let values = cell_to_vec(&cell);
        assert_eq!(values.len(), 3);
        assert_eq!(
            values[0],
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap())
        );
        assert_eq!(
            values[1],
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap())
        );
        assert_eq!(
            values[2],
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap())
        );
    }

    #[test]
    fn ind2sub_accepts_every_integer_class_and_returns_double() {
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
            let result = ind2sub_builtin(
                Value::Tensor(typed(&[2, 3], vec![1, 2])),
                Value::Tensor(typed(&[5, 6], vec![1, 2])),
            )
            .expect("ind2sub");
            let Value::Cell(result) = result else {
                panic!("expected cell output");
            };
            let outputs = cell_to_vec(&result);
            assert_eq!(outputs.len(), 2);
            for (output, expected) in outputs.iter().zip([vec![1.0, 2.0], vec![3.0, 3.0]]) {
                let Value::Tensor(output) = output else {
                    panic!("expected double tensor output");
                };
                assert_eq!(output.shape, vec![1, 2]);
                assert_eq!(output.materialize_f64(), expected);
                assert!(output.integer_storage().is_none());
            }
        }
    }

    #[test]
    fn ind2sub_uses_wide_typed_index_exactly_before_double_output() {
        let max = 9_007_199_254_740_992_u64;
        let wide_index = max + 1;
        assert_eq!(max as f64, wide_index as f64);

        let dims =
            Tensor::new_integer(IntegerStorage::U64(vec![2, max]), vec![1, 2]).expect("typed dims");
        let index = Tensor::new_integer(IntegerStorage::U64(vec![wide_index]), vec![1, 1])
            .expect("typed index");

        let result = ind2sub_builtin(Value::Tensor(dims), Value::Tensor(index))
            .expect("wide index remains exact through subscript computation");
        let Value::Cell(result) = result else {
            panic!("expected cell output");
        };
        let outputs = cell_to_vec(&result);
        assert_eq!(outputs[0], Value::Num(1.0));
        assert_eq!(outputs[1], Value::Num(4_503_599_627_370_497.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn expands_last_dimension_for_out_of_range_index() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let result = ind2sub_builtin(Value::Tensor(dims), Value::Num(13.0)).expect("expanded");
        let Value::Cell(result) = result else {
            panic!("expected subscript outputs");
        };
        assert_eq!(result.data, vec![Value::Num(1.0), Value::Num(5.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_zero_index() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err =
            ind2sub_builtin(Value::Tensor(dims), Value::Num(0.0)).expect_err("expected failure");
        assert!(
            err.contains("Linear indices must be positive integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_fractional_index() {
        let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let err =
            ind2sub_builtin(Value::Tensor(dims), Value::Num(2.5)).expect_err("expected failure");
        assert!(
            err.contains("Linear indices must be positive integers"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_oversized_float_linear_indices_before_casting() {
        let dims = Tensor::new_integer(
            IntegerStorage::U64(vec![usize::MAX.saturating_sub(1) as u64]),
            vec![1, 1],
        )
        .unwrap();

        let err = ind2sub_builtin(Value::Tensor(dims.clone()), Value::Num(1.0e300))
            .expect_err("huge float index must reject");
        assert_eq!(
            err.identifier(),
            super::IND2SUB_ERROR_INVALID_INPUT.identifier
        );

        let err = ind2sub_builtin(Value::Tensor(dims), Value::Num(usize::MAX as f64))
            .expect_err("platform boundary float index must reject");
        assert_eq!(
            err.identifier(),
            super::IND2SUB_ERROR_INVALID_INPUT.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn errors_on_invalid_size_elements() {
        let dims = Tensor::new(vec![3.5, 4.0], vec![1, 2]).unwrap();
        let err = ind2sub_builtin(Value::Tensor(dims), Value::Num(5.0)).expect_err("expected fail");
        assert!(
            err.to_string()
                .contains("Size arguments must be positive integers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ind2sub_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
            let idx_tensor = Tensor::new(vec![10.0, 11.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &idx_tensor.materialize_f64(),
                shape: &idx_tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload indices");
            let result = ind2sub_builtin(Value::Tensor(dims), Value::GpuTensor(handle)).unwrap();
            match result {
                Value::Cell(cell) => {
                    let values = cell_to_vec(&cell);
                    assert_eq!(values.len(), 2);
                    match &values[0] {
                        Value::GpuTensor(_) => {}
                        other => panic!("expected gpu tensor output, got {other:?}"),
                    }
                    match &values[1] {
                        Value::GpuTensor(_) => {}
                        other => panic!("expected gpu tensor output, got {other:?}"),
                    }
                    let rows = test_support::gather(values[0].clone()).expect("gather rows");
                    assert_eq!(rows.shape, vec![2, 1]);
                    assert_eq!(rows.materialize_f64(), vec![1.0, 2.0]);
                    let cols = test_support::gather(values[1].clone()).expect("gather cols");
                    assert_eq!(cols.shape, vec![2, 1]);
                    assert_eq!(cols.materialize_f64(), vec![4.0, 4.0]);
                }
                other => panic!("expected cell output, got {other:?}"),
            }
        });
    }

    #[test]
    fn ind2sub_integer_gpu_input_falls_back_exactly_to_resident_double() {
        test_support::with_test_provider(|provider| {
            let dims =
                Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2]).expect("dims");
            let indices = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[5, 6]),
                    shape: &[1, 2],
                })
                .expect("indices");
            let result =
                ind2sub_builtin(Value::Tensor(dims), Value::GpuTensor(indices)).expect("ind2sub");
            let Value::Cell(result) = result else {
                panic!("expected cell output");
            };
            let outputs = cell_to_vec(&result);
            assert_eq!(outputs.len(), 2);
            for (output, expected) in outputs.into_iter().zip([vec![1.0, 2.0], vec![3.0, 3.0]]) {
                let Value::GpuTensor(handle) = &output else {
                    panic!("expected resident double output");
                };
                assert_eq!(runmat_accelerate_api::handle_integer_type(handle), None);
                let output = test_support::gather(output).expect("gather output");
                assert_eq!(output.shape, vec![1, 2]);
                assert_eq!(output.materialize_f64(), expected);
                assert!(output.integer_storage().is_none());
            }
        });
    }

    #[test]
    fn ind2sub_scalar_resident_fallback_restores_double_to_the_exact_owner() {
        test_support::with_test_provider(|provider| {
            let indices = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[5]),
                    shape: &[1, 1],
                })
                .expect("scalar index");
            runmat_accelerate_api::set_handle_provenance(
                &indices,
                runmat_accelerate_api::GpuHandleProvenance::Explicit,
            );
            let result = ind2sub_builtin(
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
                Value::GpuTensor(indices.clone()),
            )
            .expect("scalar fallback");
            let Value::Cell(cell) = result else {
                panic!("expected cell output");
            };
            for output in cell_to_vec(&cell) {
                let Value::GpuTensor(handle) = output else {
                    panic!("scalar fallback must restore resident output");
                };
                assert!(super::gpu_helpers::exact_provider_for_handle(&handle)
                    .is_some_and(|owner| std::ptr::eq(owner, provider)));
                assert_eq!(
                    runmat_accelerate_api::handle_provenance(&handle),
                    Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                );
                let _ = provider.free(&handle);
            }
            let _ = provider.free(&indices);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ind2sub_wgpu_matches_cpu() {
        let provider_init = std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        });
        if let Ok(Ok(_)) = provider_init {
            // provider successfully registered
        } else {
            return;
        }

        let dims_tensor = Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap();
        let idx_tensor = Tensor::new(vec![7.0, 8.0, 9.0], vec![1, 3]).unwrap();

        let cpu = ind2sub_builtin(
            Value::Tensor(dims_tensor.clone()),
            Value::Tensor(idx_tensor.clone()),
        )
        .expect("cpu ind2sub");

        let provider = runmat_accelerate_api::provider().unwrap();
        let view = HostTensorView {
            data: &idx_tensor.materialize_f64(),
            shape: &idx_tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload indices");

        let gpu = ind2sub_builtin(Value::Tensor(dims_tensor), Value::GpuTensor(handle))
            .expect("gpu ind2sub");

        let cpu_values = match cpu {
            Value::Cell(cell) => cell_to_vec(&cell),
            other => panic!("expected cell output, got {other:?}"),
        };
        let gpu_values = match gpu {
            Value::Cell(cell) => cell_to_vec(&cell),
            other => panic!("expected cell output, got {other:?}"),
        };

        assert_eq!(cpu_values.len(), gpu_values.len());

        for (cpu_val, gpu_val) in cpu_values.iter().zip(gpu_values.iter()) {
            let host_cpu = test_support::gather(cpu_val.clone()).expect("gather cpu");
            let host_gpu = test_support::gather(gpu_val.clone()).expect("gather gpu");
            assert_eq!(host_cpu.shape, host_gpu.shape);
            assert_eq!(host_cpu.materialize_f64(), host_gpu.materialize_f64());
        }
    }
}
