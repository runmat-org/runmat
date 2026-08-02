//! MATLAB-compatible `gradient` builtin with scalar and coordinate-vector spacing support.

use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "gradient";

fn gradient_type(args: &[Type], ctx: &ResolveContext) -> Type {
    numeric_unary_type(args, ctx)
}

const GRADIENT_OUTPUT_G: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Primary gradient component.",
}];

const GRADIENT_OUTPUT_GS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Gi",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Gradient components ordered by MATLAB axis semantics.",
}];

const GRADIENT_INPUTS_F: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
}];

const GRADIENT_INPUTS_F_H: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Scalar spacing shared across all output dimensions, or a coordinate vector for vector inputs.",
    },
];

const GRADIENT_INPUTS_F_HS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "h_i",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Per-dimension scalar or coordinate-vector spacings (one per gradient dimension).",
    },
];

const GRADIENT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "G = gradient(F)",
        inputs: &GRADIENT_INPUTS_F,
        outputs: &GRADIENT_OUTPUT_G,
    },
    BuiltinSignatureDescriptor {
        label: "G = gradient(F, h)",
        inputs: &GRADIENT_INPUTS_F_H,
        outputs: &GRADIENT_OUTPUT_G,
    },
    BuiltinSignatureDescriptor {
        label: "[G1, G2, ...] = gradient(F)",
        inputs: &GRADIENT_INPUTS_F,
        outputs: &GRADIENT_OUTPUT_GS,
    },
    BuiltinSignatureDescriptor {
        label: "[G1, G2, ...] = gradient(F, h1, h2, ...)",
        inputs: &GRADIENT_INPUTS_F_HS,
        outputs: &GRADIENT_OUTPUT_GS,
    },
];

const GRADIENT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRADIENT.INVALID_ARGUMENT",
    identifier: Some("RunMat:gradient:InvalidArgument"),
    when: "Output-count or spacing argument grammar is invalid.",
    message: "gradient: invalid argument",
};

const GRADIENT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRADIENT.INVALID_INPUT",
    identifier: Some("RunMat:gradient:InvalidInput"),
    when: "Input value cannot be converted to a supported gradient domain.",
    message: "gradient: invalid input",
};

const GRADIENT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRADIENT.INTERNAL",
    identifier: Some("RunMat:gradient:Internal"),
    when: "Gradient execution fails due to gather, conversion, allocation, or indexing operations.",
    message: "gradient: internal failure",
};

const GRADIENT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GRADIENT_ERROR_INVALID_ARGUMENT,
    GRADIENT_ERROR_INVALID_INPUT,
    GRADIENT_ERROR_INTERNAL,
];

pub const GRADIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRADIENT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GRADIENT_ERRORS,
};

fn gradient_descriptor_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn gradient_descriptor_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    gradient_descriptor_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn gradient_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    gradient_descriptor_error_with_detail(&GRADIENT_ERROR_INVALID_ARGUMENT, detail)
}

fn gradient_invalid_input(detail: impl AsRef<str>) -> RuntimeError {
    gradient_descriptor_error_with_detail(&GRADIENT_ERROR_INVALID_INPUT, detail)
}

fn gradient_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    gradient_descriptor_error_with_detail(&GRADIENT_ERROR_INTERNAL, detail)
}

#[derive(Clone, Debug, PartialEq)]
enum GradientSpacing {
    Scalar(f64),
    Coordinates(Vec<f64>),
}

impl GradientSpacing {
    fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::gradient")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gradient",
    op_kind: GpuOpKind::Custom("numerical-gradient"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Custom("gradient_dim"),
        ProviderHook::Custom("gradient_dim_with_coordinates"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers may keep scalar-spacing gradients on device via `gradient_dim` and coordinate-vector spacing via `gradient_dim_with_coordinates`.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::gradient")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gradient",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Gradient preserves input shape and uses edge-aware finite differences, so providers expose it through a custom sink hook.",
};

#[runtime_builtin(
    name = "gradient",
    category = "math/reduction",
    summary = "Compute numerical gradients.",
    keywords = "gradient,numerical gradient,finite difference,vector field,gpu",
    accel = "gradient",
    type_resolver(gradient_type),
    descriptor(crate::builtins::math::reduction::gradient::GRADIENT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::reduction::gradient"
)]
async fn gradient_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    if requested_outputs == 0 {
        return Ok(Value::OutputList(Vec::new()));
    }

    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(gradient_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }

    let available_outputs = gradient_output_dims(value_shape(&value), value_len(&value));
    if requested_outputs > available_outputs.len() {
        return Err(gradient_invalid_argument(format!(
            "gradient: requested {requested_outputs} outputs, but input supports at most {}",
            available_outputs.len()
        )));
    }

    let dim_lengths =
        gradient_dim_lengths(value_shape(&value), value_len(&value), &available_outputs);
    let spacings = parse_spacings(&rest, &available_outputs, &dim_lengths).await?;
    let outputs =
        evaluate_gradient_outputs(value, &available_outputs[..requested_outputs], &spacings)
            .await?;

    if crate::output_count::current_output_count().is_some() {
        return Ok(Value::OutputList(outputs));
    }

    Ok(outputs
        .into_iter()
        .next()
        .expect("single-output gradient result"))
}

async fn evaluate_gradient_outputs(
    value: Value,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
) -> BuiltinResult<Vec<Value>> {
    if let Value::GpuTensor(handle) = value {
        return gradient_gpu_outputs(handle, requested_dims, all_spacings).await;
    }

    evaluate_host_gradient_outputs(value, requested_dims, all_spacings)
}

fn evaluate_host_gradient_outputs(
    value: Value,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
) -> BuiltinResult<Vec<Value>> {
    match value {
        Value::Tensor(tensor) => {
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(
                    gradient_real_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(gradient_invalid_input)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(
                    gradient_real_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let tensor =
                tensor::value_into_tensor_for(NAME, value).map_err(gradient_invalid_input)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(tensor::tensor_into_value(
                    gradient_real_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::Complex(re, im) => {
            let tensor =
                ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(gradient_invalid_input)?;
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(complex_tensor_into_value(
                    gradient_complex_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        Value::ComplexTensor(tensor) => {
            let mut outputs = Vec::with_capacity(requested_dims.len());
            for &dim in requested_dims {
                let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
                outputs.push(complex_tensor_into_value(
                    gradient_complex_tensor_host_with_spacing(tensor.clone(), dim, spacing)?,
                ));
            }
            Ok(outputs)
        }
        other => Err(gradient_invalid_input(format!(
            "gradient: unsupported input type {:?}; expected numeric or logical data",
            other
        ))),
    }
}

async fn gradient_gpu_outputs(
    handle: GpuTensorHandle,
    requested_dims: &[usize],
    all_spacings: &[GradientSpacing],
) -> BuiltinResult<Vec<Value>> {
    let complex_storage =
        runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::ComplexInterleaved;

    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(&handle).or_else(runmat_accelerate_api::provider)
    {
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let mut outputs = Vec::with_capacity(requested_dims.len());
        for &dim in requested_dims {
            let spacing = spacing_for_dim(dim, requested_dims, all_spacings);
            let device_result = match spacing {
                GradientSpacing::Scalar(spacing) => {
                    provider.gradient_dim(&handle, dim.saturating_sub(1), *spacing)
                }
                GradientSpacing::Coordinates(coordinates) => {
                    let shape = vec![coordinates.len(), 1];
                    let coord_handle =
                        match provider.upload(&runmat_accelerate_api::HostTensorView {
                            data: coordinates,
                            shape: &shape,
                        }) {
                            Ok(handle) => handle,
                            Err(_) => {
                                let gathered =
                                    gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                                        .await?;
                                return evaluate_host_gradient_outputs(
                                    gathered,
                                    requested_dims,
                                    all_spacings,
                                );
                            }
                        };
                    let result = provider.gradient_dim_with_coordinates(
                        &handle,
                        dim.saturating_sub(1),
                        &coord_handle,
                    );
                    let _ = provider.free(&coord_handle);
                    result
                }
            };
            match device_result {
                Ok(device_result) => {
                    if complex_storage
                        || runmat_accelerate_api::handle_storage(&device_result)
                            == GpuTensorStorage::ComplexInterleaved
                    {
                        outputs.push(gpu_helpers::complex_gpu_value(device_result));
                    } else {
                        outputs.push(gpu_helpers::resident_gpu_value(device_result));
                    }
                }
                Err(_) => {
                    let gathered =
                        gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
                    return evaluate_host_gradient_outputs(gathered, requested_dims, all_spacings);
                }
            }
        }
        return Ok(outputs);
    }

    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
    evaluate_host_gradient_outputs(gathered, requested_dims, all_spacings)
}

fn spacing_for_dim<'a>(
    dim: usize,
    available_dims: &[usize],
    spacings: &'a [GradientSpacing],
) -> &'a GradientSpacing {
    let index = available_dims
        .iter()
        .position(|candidate| *candidate == dim)
        .expect("spacing lookup requires matching dimension");
    &spacings[index]
}

async fn parse_spacings(
    args: &[Value],
    available_dims: &[usize],
    dim_lengths: &[usize],
) -> BuiltinResult<Vec<GradientSpacing>> {
    match args.len() {
        0 => Ok(vec![GradientSpacing::Scalar(1.0); available_dims.len()]),
        1 => {
            let spacing = parse_spacing_argument(&args[0], dim_lengths[0]).await?;
            if spacing.is_scalar() {
                Ok(vec![spacing; available_dims.len()])
            } else if available_dims.len() == 1 {
                Ok(vec![spacing])
            } else {
                Err(gradient_invalid_argument(
                    "gradient: coordinate-vector spacing for arrays requires one spacing argument per gradient dimension",
                ))
            }
        }
        count if count == available_dims.len() => {
            let mut spacings = Vec::with_capacity(args.len());
            for (value, &dim_len) in args.iter().zip(dim_lengths.iter()) {
                spacings.push(parse_spacing_argument(value, dim_len).await?);
            }
            Ok(spacings)
        }
        _ => Err(gradient_invalid_argument(format!(
            "gradient: expected 0, 1, or {} scalar/coordinate-vector spacing arguments",
            available_dims.len()
        ))),
    }
}

async fn parse_spacing_argument(value: &Value, dim_len: usize) -> BuiltinResult<GradientSpacing> {
    if let Value::GpuTensor(_) = value {
        let gathered = gpu_helpers::gather_value_async(value).await?;
        return parse_host_spacing_argument(&gathered, dim_len);
    }
    parse_host_spacing_argument(value, dim_len)
}

fn parse_host_spacing_argument(value: &Value, dim_len: usize) -> BuiltinResult<GradientSpacing> {
    let tensor =
        tensor::value_into_tensor_for(NAME, value.clone()).map_err(gradient_invalid_argument)?;
    if tensor_len(&tensor) == 0 {
        return Err(gradient_invalid_argument(
            "gradient: empty spacing arguments are not supported",
        ));
    }

    let spacing_values = tensor::tensor_into_values_f64(tensor);
    if spacing_values.len() == 1 {
        let spacing = spacing_values[0];
        validate_scalar_spacing(spacing)?;
        return Ok(GradientSpacing::Scalar(spacing));
    }

    validate_coordinate_spacing(&spacing_values, dim_len)?;
    Ok(GradientSpacing::Coordinates(spacing_values))
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn validate_scalar_spacing(spacing: f64) -> BuiltinResult<()> {
    if !spacing.is_finite() {
        return Err(gradient_invalid_argument(
            "gradient: spacing must be finite",
        ));
    }
    if spacing == 0.0 {
        return Err(gradient_invalid_argument(
            "gradient: spacing must be nonzero",
        ));
    }
    Ok(())
}

fn validate_coordinate_spacing(coords: &[f64], dim_len: usize) -> BuiltinResult<()> {
    if coords.len() != dim_len {
        return Err(gradient_invalid_argument(format!(
            "gradient: coordinate-vector spacing length {} does not match dimension length {dim_len}",
            coords.len()
        )));
    }

    if coords.iter().any(|coord| !coord.is_finite()) {
        return Err(gradient_invalid_argument(
            "gradient: coordinate-vector spacing must be finite",
        ));
    }

    if coords.len() <= 1 {
        return Ok(());
    }

    if coords[1] == coords[0] {
        return Err(gradient_invalid_argument(
            "gradient: coordinate-vector spacing points must be distinct",
        ));
    }

    for k in 1..coords.len() {
        if coords[k] == coords[k - 1] {
            return Err(gradient_invalid_argument(
                "gradient: coordinate-vector spacing points must be distinct",
            ));
        }
    }

    for k in 1..coords.len() - 1 {
        if coords[k + 1] == coords[k - 1] {
            return Err(gradient_invalid_argument(
                "gradient: coordinate-vector spacing cannot produce zero finite-difference denominator",
            ));
        }
    }
    Ok(())
}

fn value_shape(value: &Value) -> &[usize] {
    match value {
        Value::Tensor(tensor) => &tensor.shape,
        Value::LogicalArray(logical) => &logical.shape,
        Value::ComplexTensor(tensor) => &tensor.shape,
        Value::GpuTensor(handle) => &handle.shape,
        _ => &[],
    }
}

fn value_len(value: &Value) -> usize {
    match value {
        Value::Tensor(tensor) => tensor_len(tensor),
        Value::LogicalArray(logical) => logical.data.len(),
        Value::ComplexTensor(tensor) => tensor.materialize_f64().len(),
        Value::GpuTensor(handle) => product(&handle.shape),
        _ => 1,
    }
}

pub fn matlab_gradient_shape(shape: &[usize], len: usize) -> Vec<usize> {
    if shape.is_empty() {
        if len == 0 {
            Vec::new()
        } else {
            vec![1, 1]
        }
    } else if shape.len() == 1 {
        if shape[0] == 1 {
            vec![1, 1]
        } else {
            vec![1, shape[0]]
        }
    } else {
        shape.to_vec()
    }
}

fn gradient_output_dims(shape: &[usize], len: usize) -> Vec<usize> {
    let normalized_shape = matlab_gradient_shape(shape, len);
    let mut ext_shape = if normalized_shape.is_empty() {
        if len == 0 {
            vec![0, 0]
        } else {
            vec![1, 1]
        }
    } else {
        normalized_shape
    };
    if ext_shape.len() == 1 {
        ext_shape.push(1);
    }

    if ext_shape.len() <= 2 {
        let rows = ext_shape.first().copied().unwrap_or(1);
        let cols = ext_shape.get(1).copied().unwrap_or(1);
        if rows == 1 && cols == 1 {
            vec![1]
        } else if rows == 1 {
            vec![2]
        } else if cols == 1 {
            vec![1]
        } else {
            vec![2, 1]
        }
    } else {
        let mut dims = vec![2, 1];
        for dim in 3..=ext_shape.len() {
            dims.push(dim);
        }
        dims
    }
}

fn gradient_dim_lengths(shape: &[usize], len: usize, dims: &[usize]) -> Vec<usize> {
    let mut ext_shape = matlab_gradient_shape(shape, len);
    if ext_shape.is_empty() {
        ext_shape = if len == 0 { vec![0, 0] } else { vec![1, 1] };
    }

    let max_dim = dims.iter().copied().max().unwrap_or(1);
    while ext_shape.len() < max_dim {
        ext_shape.push(1);
    }

    dims.iter()
        .map(|dim| ext_shape[dim.saturating_sub(1)])
        .collect()
}

pub fn gradient_real_tensor_host(
    tensor: Tensor,
    dim: usize,
    spacing: f64,
) -> BuiltinResult<Tensor> {
    let spacing = GradientSpacing::Scalar(spacing);
    gradient_real_tensor_host_with_spacing(tensor, dim, &spacing)
}

#[allow(dead_code)]
pub fn gradient_real_tensor_host_with_coordinates(
    tensor: Tensor,
    dim: usize,
    coordinates: Vec<f64>,
) -> BuiltinResult<Tensor> {
    let spacing = GradientSpacing::Coordinates(coordinates);
    gradient_real_tensor_host_with_spacing(tensor, dim, &spacing)
}

fn gradient_real_tensor_host_with_spacing(
    tensor: Tensor,
    dim: usize,
    spacing: &GradientSpacing,
) -> BuiltinResult<Tensor> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| gradient_internal_error(format!("gradient: {error}")))?;
    let output_dtype = match storage.numeric_dtype() {
        NumericDType::F32 => NumericDType::F32,
        NumericDType::F64
        | NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => NumericDType::F64,
    };
    let data = storage.materialize_f64();
    let dim_index = dim.saturating_sub(1);
    let mut shape = matlab_gradient_shape(&shape, data.len());

    if data.is_empty() {
        // Return early before the `push(1)` padding loop: that loop would give a
        // shape like [1] or [1,1] whose product is 1 ≠ 0, violating Tensor's
        // invariant. Use the normalised shape directly, falling back to [0,0] if
        // matlab_gradient_shape returned an empty vec (untyped empty tensor).
        let empty_shape = if shape.is_empty() { vec![0, 0] } else { shape };
        return Tensor::new_with_dtype(Vec::new(), empty_shape, output_dtype)
            .map_err(|e| gradient_internal_error(format!("gradient: {e}")));
    }

    while shape.len() <= dim_index {
        shape.push(1);
    }

    let mut ext_shape = shape.clone();
    while ext_shape.len() <= dim_index {
        ext_shape.push(1);
    }
    let len_dim = ext_shape[dim_index];
    let stride_before = if dim_index == 0 {
        1usize
    } else {
        product(&ext_shape[..dim_index]).max(1)
    };
    let stride_after = if dim_index + 1 >= ext_shape.len() {
        1usize
    } else {
        product(&ext_shape[dim_index + 1..]).max(1)
    };

    let mut out = vec![0.0; data.len()];
    if len_dim > 1 {
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| gradient_internal_error("gradient: block size overflow"))?;
        for after in 0..stride_after {
            let base = after
                .checked_mul(block)
                .ok_or_else(|| gradient_internal_error("gradient: indexing overflow"))?;
            for before in 0..stride_before {
                for k in 0..len_dim {
                    let idx = base + before + k * stride_before;
                    out[idx] = if k == 0 {
                        (data[idx + stride_before] - data[idx])
                            / spacing_denominator(spacing, k, len_dim)
                    } else if k + 1 == len_dim {
                        (data[idx] - data[idx - stride_before])
                            / spacing_denominator(spacing, k, len_dim)
                    } else {
                        (data[idx + stride_before] - data[idx - stride_before])
                            / spacing_denominator(spacing, k, len_dim)
                    };
                }
            }
        }
    }

    Tensor::new_with_dtype(out, shape, output_dtype)
        .map_err(|e| gradient_internal_error(format!("gradient: {e}")))
}

pub fn gradient_complex_tensor_host(
    tensor: ComplexTensor,
    dim: usize,
    spacing: f64,
) -> BuiltinResult<ComplexTensor> {
    let spacing = GradientSpacing::Scalar(spacing);
    gradient_complex_tensor_host_with_spacing(tensor, dim, &spacing)
}

#[allow(dead_code)]
pub fn gradient_complex_tensor_host_with_coordinates(
    tensor: ComplexTensor,
    dim: usize,
    coordinates: Vec<f64>,
) -> BuiltinResult<ComplexTensor> {
    let spacing = GradientSpacing::Coordinates(coordinates);
    gradient_complex_tensor_host_with_spacing(tensor, dim, &spacing)
}

fn gradient_complex_tensor_host_with_spacing(
    tensor: ComplexTensor,
    dim: usize,
    spacing: &GradientSpacing,
) -> BuiltinResult<ComplexTensor> {
    let output_dtype = if tensor.numeric_dtype() == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let shape = tensor.shape.clone();
    let data = tensor.materialize_f64();
    let dim_index = dim.saturating_sub(1);
    let mut shape = matlab_gradient_shape(&shape, data.len());

    if data.is_empty() {
        // Same fix as gradient_real_tensor_host: avoid padding the shape with 1s
        // before the early return, which would produce product ≠ 0 for empty data.
        let empty_shape = if shape.is_empty() { vec![0, 0] } else { shape };
        return ComplexTensor::from_f64_values_with_dtype(Vec::new(), empty_shape, output_dtype)
            .map_err(|e| gradient_internal_error(format!("gradient: {e}")));
    }

    while shape.len() <= dim_index {
        shape.push(1);
    }

    let mut ext_shape = shape.clone();
    while ext_shape.len() <= dim_index {
        ext_shape.push(1);
    }
    let len_dim = ext_shape[dim_index];
    let stride_before = if dim_index == 0 {
        1usize
    } else {
        product(&ext_shape[..dim_index]).max(1)
    };
    let stride_after = if dim_index + 1 >= ext_shape.len() {
        1usize
    } else {
        product(&ext_shape[dim_index + 1..]).max(1)
    };

    let mut out = vec![(0.0, 0.0); data.len()];
    if len_dim > 1 {
        let block = stride_before
            .checked_mul(len_dim)
            .ok_or_else(|| gradient_internal_error("gradient: block size overflow"))?;
        for after in 0..stride_after {
            let base = after
                .checked_mul(block)
                .ok_or_else(|| gradient_internal_error("gradient: indexing overflow"))?;
            for before in 0..stride_before {
                for k in 0..len_dim {
                    let idx = base + before + k * stride_before;
                    out[idx] = if k == 0 {
                        scale_complex(
                            sub_complex(data[idx + stride_before], data[idx]),
                            1.0 / spacing_denominator(spacing, k, len_dim),
                        )
                    } else if k + 1 == len_dim {
                        scale_complex(
                            sub_complex(data[idx], data[idx - stride_before]),
                            1.0 / spacing_denominator(spacing, k, len_dim),
                        )
                    } else {
                        scale_complex(
                            sub_complex(data[idx + stride_before], data[idx - stride_before]),
                            1.0 / spacing_denominator(spacing, k, len_dim),
                        )
                    };
                }
            }
        }
    }

    ComplexTensor::from_f64_values_with_dtype(out, shape, output_dtype)
        .map_err(|e| gradient_internal_error(format!("gradient: {e}")))
}

fn spacing_denominator(spacing: &GradientSpacing, k: usize, len_dim: usize) -> f64 {
    match spacing {
        GradientSpacing::Scalar(spacing) => {
            if k == 0 || k + 1 == len_dim {
                *spacing
            } else {
                2.0 * spacing
            }
        }
        GradientSpacing::Coordinates(coords) => {
            if k == 0 {
                coords[1] - coords[0]
            } else if k + 1 == len_dim {
                coords[len_dim - 1] - coords[len_dim - 2]
            } else {
                coords[k + 1] - coords[k - 1]
            }
        }
    }
}

fn sub_complex(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (lhs.0 - rhs.0, lhs.1 - rhs.1)
}

fn scale_complex(value: (f64, f64), scale: f64) -> (f64, f64) {
    (value.0 * scale, value.1 * scale)
}

fn product(dims: &[usize]) -> usize {
    dims.iter()
        .copied()
        .fold(1usize, |acc, value| acc.saturating_mul(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntegerStorage, NumericDType, NumericStorage, Tensor};

    fn gradient_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::gradient_builtin(value, rest))
    }

    #[test]
    fn gradient_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = GRADIENT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"G = gradient(F)"));
        assert!(labels.contains(&"G = gradient(F, h)"));
        assert!(labels.contains(&"[G1, G2, ...] = gradient(F)"));
        assert!(labels.contains(&"[G1, G2, ...] = gradient(F, h1, h2, ...)"));
    }

    #[test]
    fn gradient_descriptor_errors_have_stable_codes() {
        assert!(GRADIENT_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == GRADIENT_ERROR_INVALID_ARGUMENT.code));
        assert!(GRADIENT_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == GRADIENT_ERROR_INVALID_INPUT.code));
        assert!(GRADIENT_DESCRIPTOR
            .errors
            .iter()
            .any(|error| error.code == GRADIENT_ERROR_INTERNAL.code));
    }

    #[test]
    fn gradient_row_vector_returns_horizontal_derivative() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        assert_eq!(
            result,
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap())
        );
    }

    #[test]
    fn gradient_one_dimensional_tensor_is_treated_as_row_vector() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![3]).unwrap();
        let result =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Num(2.0)]).expect("gradient");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.materialize_f64(), vec![1.5, 2.0, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_matrix_outputs_follow_matlab_order() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                let fx = test_support::gather(outputs[0].clone()).expect("fx");
                let fy = test_support::gather(outputs[1].clone()).expect("fy");
                assert_eq!(fx.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
                assert_eq!(fy.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]);
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn gradient_scalar_spacing_scales_output() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let result =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Num(2.0)]).expect("gradient");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.5, 2.0, 2.5]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_scalar_spacing_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing =
            Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("spacing");

        let result = gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)])
            .expect("gradient");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![1.5, 2.0, 2.5]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_input_reads_typed_integer_storage_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 4, 9]), vec![1, 3]).expect("input");

        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.numeric_dtype(), NumericDType::F64);
                assert!(out.integer_storage().is_none());
                assert_eq!(out.materialize_f64(), vec![3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_multi_output_uses_typed_integer_storage_length_without_mirror() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 3, 2, 4]), vec![2, 2]).expect("input");
        let _guard = crate::output_count::push_output_count(Some(2));

        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                let fx = test_support::gather(outputs[0].clone()).expect("fx");
                let fy = test_support::gather(outputs[1].clone()).expect("fy");
                assert_eq!(fx.shape, vec![2, 2]);
                assert_eq!(fx.numeric_dtype(), NumericDType::F64);
                assert!(fx.integer_storage().is_none());
                assert_eq!(fx.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
                assert_eq!(fy.shape, vec![2, 2]);
                assert_eq!(fy.numeric_dtype(), NumericDType::F64);
                assert!(fy.integer_storage().is_none());
                assert_eq!(fy.materialize_f64(), vec![2.0, 2.0, 2.0, 2.0]);
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn gradient_preserves_single_precision_host_tensor() {
        let tensor =
            Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
        let result = gradient_builtin(Value::Tensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::Tensor(out) => assert_eq!(
                out.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![3.0, 4.0, 5.0])
            ),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_complex_host_supported() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (4.0, 3.0), (9.0, 6.0)], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("gradient");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(
                    out.materialize_f64(),
                    vec![(3.0, 2.0), (4.0, 2.5), (5.0, 3.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_coordinate_vector_spacing_for_row_vector() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)])
            .expect("gradient");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.materialize_f64(), vec![3.0, 8.0 / 3.0, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_coordinate_vector_spacing_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing =
            Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 3]), vec![1, 3]).expect("spacing");

        let result = gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)])
            .expect("gradient");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(out.materialize_f64(), vec![3.0, 8.0 / 3.0, 2.5]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_mixed_scalar_and_coordinate_vector_spacing_for_matrix() {
        let tensor = Tensor::new(vec![0.0, 20.0, 1.0, 21.0, 9.0, 29.0], vec![2, 3]).unwrap();
        let x = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let result = gradient_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(x), Value::Num(2.0)],
        )
        .expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                let fx = test_support::gather(outputs[0].clone()).expect("fx");
                let fy = test_support::gather(outputs[1].clone()).expect("fy");
                assert_eq!(fx.shape, vec![2, 3]);
                assert_eq!(fx.materialize_f64(), vec![1.0, 1.0, 3.0, 3.0, 4.0, 4.0]);
                assert_eq!(fy.shape, vec![2, 3]);
                assert_eq!(
                    fy.materialize_f64(),
                    vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
                );
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn gradient_complex_coordinate_vector_spacing() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (4.0, 3.0), (9.0, 7.0)], vec![1, 3]).unwrap();
        let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::ComplexTensor(tensor), vec![Value::Tensor(spacing)])
            .expect("gradient");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![1, 3]);
                assert_eq!(
                    out.materialize_f64(),
                    vec![(3.0, 2.0), (8.0 / 3.0, 2.0), (2.5, 2.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_rejects_coordinate_vector_length_mismatch() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let err =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)]).unwrap_err();
        assert_eq!(err.identifier(), GRADIENT_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("length"));
    }

    #[test]
    fn gradient_allows_nonmonotonic_coordinate_vector_spacing() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing = Tensor::new(vec![0.0, 1.0, 0.5], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)])
            .expect("gradient");
        match result {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![3.0, 16.0, -10.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gradient_rejects_zero_coordinate_denominator() {
        let tensor = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
        let spacing = Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap();
        let err =
            gradient_builtin(Value::Tensor(tensor), vec![Value::Tensor(spacing)]).unwrap_err();
        assert_eq!(err.identifier(), GRADIENT_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("denominator"));
    }

    #[test]
    fn gradient_rejects_too_many_outputs() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let _guard = crate::output_count::push_output_count(Some(2));
        let err = gradient_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), GRADIENT_ERROR_INVALID_ARGUMENT.identifier);
        assert!(err.message().contains("requested 2 outputs"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_scalar_spacing_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let host =
            Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
        let view = HostTensorView {
            data: &host.materialize_f64(),
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            gradient_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("gradient");
        match result {
            Value::GpuTensor(out) => {
                let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
                assert_eq!(gathered.materialize_f64(), vec![1.5, 2.0, 2.5]);
                assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_coordinate_spacing_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let host =
            Tensor::new_with_dtype(vec![1.0, 4.0, 9.0], vec![1, 3], NumericDType::F32).unwrap();
        let view = HostTensorView {
            data: &host.materialize_f64(),
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
        let result = gradient_builtin(Value::GpuTensor(handle), vec![Value::Tensor(spacing)])
            .expect("gradient");
        match result {
            Value::GpuTensor(out) => {
                let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
                assert_eq!(gathered.shape, vec![1, 3]);
                assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
                let expected = [3.0, 8.0 / 3.0, 2.5];
                for (idx, (actual, expected)) in
                    gathered.materialize_f64().iter().zip(expected).enumerate()
                {
                    assert!(
                        (*actual - expected).abs() < 1.0e-5,
                        "gradient mismatch at {idx}: actual={actual} expected={expected}"
                    );
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_one_dimensional_shape_matches_matlab_row_vector_semantics() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let data = [1.0, 4.0, 9.0];
        let shape = [3usize];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result =
            gradient_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("gradient");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 3]);
        assert_eq!(gathered.materialize_f64(), vec![1.5, 2.0, 2.5]);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_multi_output_uses_output_list() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let host = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &host.materialize_f64(),
            shape: &host.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let _out_guard = crate::output_count::push_output_count(Some(2));
        let result = gradient_builtin(Value::GpuTensor(handle), Vec::new()).expect("gradient");
        match result {
            Value::OutputList(outputs) => {
                assert!(matches!(outputs[0], Value::GpuTensor(_)));
                assert!(matches!(outputs[1], Value::GpuTensor(_)));
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn gradient_gpu_coordinate_vector_spacing_stays_resident() {
        test_support::with_test_provider(|provider| {
            let host = Tensor::new(vec![1.0, 4.0, 9.0], vec![1, 3]).unwrap();
            let view = HostTensorView {
                data: &host.materialize_f64(),
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let spacing = Tensor::new(vec![0.0, 1.0, 3.0], vec![1, 3]).unwrap();
            let result = gradient_builtin(Value::GpuTensor(handle), vec![Value::Tensor(spacing)])
                .expect("gradient");
            match result {
                Value::GpuTensor(out_handle) => {
                    let out = test_support::gather(Value::GpuTensor(out_handle)).expect("gather");
                    assert_eq!(out.shape, vec![1, 3]);
                    assert_eq!(out.materialize_f64(), vec![3.0, 8.0 / 3.0, 2.5]);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn gradient_gpu_mixed_scalar_and_coordinate_outputs_stay_resident() {
        test_support::with_test_provider(|provider| {
            let host = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &host.materialize_f64(),
                shape: &host.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let spacing = Tensor::new(vec![0.0, 2.0], vec![2, 1]).unwrap();
            let _out_guard = crate::output_count::push_output_count(Some(2));
            let result = gradient_builtin(
                Value::GpuTensor(handle),
                vec![Value::Tensor(spacing), Value::Num(2.0)],
            )
            .expect("gradient");
            match result {
                Value::OutputList(outputs) => {
                    assert!(matches!(outputs[0], Value::GpuTensor(_)));
                    assert!(matches!(outputs[1], Value::GpuTensor(_)));
                    let first = test_support::gather(outputs[0].clone()).expect("gather first");
                    let second = test_support::gather(outputs[1].clone()).expect("gather second");
                    assert_eq!(first.shape, vec![2, 2]);
                    assert_eq!(first.materialize_f64(), vec![0.5, 0.5, 0.5, 0.5]);
                    assert_eq!(second.shape, vec![2, 2]);
                    assert_eq!(second.materialize_f64(), vec![1.0, 1.0, 1.0, 1.0]);
                }
                other => panic!("expected output list, got {other:?}"),
            }
        });
    }

    #[test]
    fn gradient_inprocess_complex_gpu_matches_cpu_and_stays_resident() {
        test_support::with_test_provider(|provider| {
            let host = ComplexTensor::new(
                vec![
                    (1.0, 1.0),
                    (2.0, -1.0),
                    (4.0, 3.0),
                    (6.0, 2.0),
                    (9.0, 6.0),
                    (12.0, 4.0),
                ],
                vec![2, 3],
            )
            .unwrap();
            let expected =
                gradient_complex_tensor_host(host.clone(), 2, 2.0).expect("cpu gradient");
            let handle = gpu_helpers::upload_complex_tensor(provider, &host).expect("upload");
            let result = gradient_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)])
                .expect("gradient");
            let Value::GpuTensor(out_handle) = result else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&out_handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
            )
            .expect("gather complex gradient");
            assert_eq!(gathered.shape, expected.shape);
            assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn gradient_gpu_complex_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let host = ComplexTensor::new(
            vec![
                (1.0, 1.0),
                (2.0, -1.0),
                (4.0, 3.0),
                (6.0, 2.0),
                (9.0, 6.0),
                (12.0, 4.0),
            ],
            vec![2, 3],
        )
        .unwrap();
        let expected = gradient_complex_tensor_host(host.clone(), 2, 2.0).expect("cpu gradient");
        let handle = gpu_helpers::upload_complex_tensor(provider, &host).expect("upload");
        let result =
            gradient_builtin(Value::GpuTensor(handle), vec![Value::Num(2.0)]).expect("gradient");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected complex gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(&out_handle, NAME),
        )
        .expect("gather complex gradient");
        assert_eq!(gathered.shape, expected.shape);
        for (idx, (actual, expected)) in gathered
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
            .enumerate()
        {
            assert!(
                (actual.0 - expected.0).abs() <= 1.0e-5,
                "real mismatch at {idx}: actual={} expected={}",
                actual.0,
                expected.0
            );
            assert!(
                (actual.1 - expected.1).abs() <= 1.0e-5,
                "imag mismatch at {idx}: actual={} expected={}",
                actual.1,
                expected.1
            );
        }
    }
}
