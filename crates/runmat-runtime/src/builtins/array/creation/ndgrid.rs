//! MATLAB-compatible `ndgrid` builtin.

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostTensorView, ProviderNdgridAxis, ProviderNdgridRequest,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericDType,
    NumericStorage, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::array::type_resolvers::size_vector_len;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "ndgrid";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::ndgrid")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: BUILTIN_NAME,
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ndgrid")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Resident real/logical gpuArray vector inputs use the provider ndgrid hook when available; unsupported providers or complex axes fall back to host materialisation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::ndgrid")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: BUILTIN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ndgrid explicitly materialises dense coordinate arrays and therefore bypasses fusion.",
};

fn ndgrid_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Unknown;
    }
    let shape = if args.len() == 1 {
        vec![size_vector_len(&args[0]), Some(1)]
    } else {
        args.iter().map(size_vector_len).collect::<Vec<_>>()
    };
    Type::Tensor { shape: Some(shape) }
}

const NDGRID_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full grid coordinates for the first dimension.",
    },
    BuiltinParamDescriptor {
        name: "X2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Full grid coordinates for the second dimension.",
    },
    BuiltinParamDescriptor {
        name: "Xn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Full grid coordinates for the nth dimension.",
    },
];

const NDGRID_SINGLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "xg",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grid vector reused for every requested output dimension.",
}];

const NDGRID_MULTI_INPUT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid vector for the first dimension.",
    },
    BuiltinParamDescriptor {
        name: "x2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grid vector for the second dimension.",
    },
    BuiltinParamDescriptor {
        name: "xn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grid vector for the nth dimension.",
    },
];

const NDGRID_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[X1, X2, ..., Xn] = ndgrid(xg)",
        inputs: &NDGRID_SINGLE_INPUT,
        outputs: &NDGRID_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[X1, X2, ..., Xn] = ndgrid(x1, x2, ..., xn)",
        inputs: &NDGRID_MULTI_INPUT,
        outputs: &NDGRID_OUTPUTS,
    },
];

const NDGRID_ERROR_MISSING_AXIS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.MISSING_AXIS",
    identifier: Some("RunMat:ndgrid:MissingAxis"),
    when: "No grid vector inputs are provided.",
    message: "ndgrid: at least one input vector is required",
};

const NDGRID_ERROR_INVALID_AXIS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.INVALID_AXIS",
    identifier: Some("RunMat:ndgrid:InvalidAxis"),
    when: "An input is not numeric vector data.",
    message: "ndgrid: inputs must be numeric vectors",
};

const NDGRID_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ndgrid:TooManyOutputs"),
    when: "More outputs are requested than available input grid vectors.",
    message: "ndgrid: too many output arguments for the supplied grid vectors",
};

const NDGRID_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.INTERNAL",
    identifier: Some("RunMat:ndgrid:Internal"),
    when: "Output allocation, GPU gather, or GPU upload fails.",
    message: "ndgrid: internal error",
};

const NDGRID_ERRORS: [BuiltinErrorDescriptor; 4] = [
    NDGRID_ERROR_MISSING_AXIS,
    NDGRID_ERROR_INVALID_AXIS,
    NDGRID_ERROR_TOO_MANY_OUTPUTS,
    NDGRID_ERROR_INTERNAL,
];

pub const NDGRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NDGRID_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NDGRID_ERRORS,
};

#[runtime_builtin(
    name = "ndgrid",
    category = "array/creation",
    summary = "Generate full coordinate arrays for an N-D rectangular grid.",
    keywords = "ndgrid,grid,n-dimensional,coordinate arrays,gpu",
    accel = "array_construct",
    type_resolver(ndgrid_type),
    descriptor(crate::builtins::array::creation::ndgrid::NDGRID_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::ndgrid"
)]
async fn ndgrid_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    let out_count = crate::output_count::current_output_count();
    if matches!(out_count, Some(0)) {
        return Ok(Value::OutputList(Vec::new()));
    }

    let eval = evaluate(&rest, out_count).await?;
    if let Some(count) = out_count {
        let mut outputs = Vec::with_capacity(count);
        for dim in 0..count {
            outputs.push(eval.output(dim).await?);
        }
        return Ok(Value::OutputList(outputs));
    }

    eval.output(0).await
}

pub async fn evaluate(
    args: &[Value],
    requested_outputs: Option<usize>,
) -> BuiltinResult<NdgridEval> {
    let parsed = ParsedNdgrid::parse(args, requested_outputs).await?;
    let output_shape = output_shape_from_lengths(&parsed.axis_lengths);
    let output_count = parsed.output_count;

    if let Some(outputs) = try_provider_outputs(&parsed, &output_shape)? {
        return Ok(NdgridEval {
            outputs,
            residency: parsed.residency,
        });
    }

    let mut host_axes = Vec::with_capacity(parsed.axes.len());
    for axis in &parsed.axes {
        host_axes.push(axis_to_host_async(axis).await?);
    }
    let outputs = build_host_outputs(&host_axes, &output_shape, output_count)?
        .into_iter()
        .map(NdgridOutput::Host)
        .collect();

    Ok(NdgridEval {
        outputs,
        residency: parsed.residency,
    })
}

fn try_provider_outputs(
    parsed: &ParsedNdgrid,
    output_shape: &[usize],
) -> BuiltinResult<Option<Vec<NdgridOutput>>> {
    let OutputResidency::Gpu { device_id } = parsed.residency else {
        return Ok(None);
    };
    if parsed.output_count == 0 {
        return Ok(Some(Vec::new()));
    }

    let Some(provider) = runmat_accelerate_api::provider_for_device(device_id)
        .or_else(runmat_accelerate_api::provider)
    else {
        return Ok(None);
    };

    let mut axes = Vec::with_capacity(parsed.output_count);
    for axis in parsed.axes.iter().take(parsed.output_count) {
        if matches!(axis.class, OutputClass::Complex)
            || matches!(
                axis.class,
                OutputClass::Real(
                    NumericDType::I8
                        | NumericDType::I16
                        | NumericDType::I32
                        | NumericDType::I64
                        | NumericDType::U8
                        | NumericDType::U16
                        | NumericDType::U32
                        | NumericDType::U64
                )
            )
        {
            return Ok(None);
        }
        let Some(handle) = axis.gpu_real.as_ref() else {
            return Ok(None);
        };
        axes.push(ProviderNdgridAxis { handle });
    }

    let request = ProviderNdgridRequest {
        axes: &axes,
        output_shape,
        output_count: parsed.output_count,
    };
    let result = match provider.ndgrid(&request) {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };
    if result.outputs.len() != parsed.output_count {
        return Err(ndgrid_error_with_detail(
            &NDGRID_ERROR_INTERNAL,
            "provider returned wrong output count",
        ));
    }

    let outputs = result
        .outputs
        .into_iter()
        .enumerate()
        .map(|(dim, handle)| {
            annotate_provider_output(&handle, parsed.axes[dim].class);
            NdgridOutput::Gpu(handle)
        })
        .collect();
    Ok(Some(outputs))
}

fn annotate_provider_output(handle: &GpuTensorHandle, class: OutputClass) {
    match class {
        OutputClass::Real(dtype) => {
            runmat_accelerate_api::set_handle_logical(handle, false);
            let precision = match dtype {
                NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                NumericDType::F64
                | NumericDType::I8
                | NumericDType::I16
                | NumericDType::I32
                | NumericDType::I64
                | NumericDType::U8
                | NumericDType::U16
                | NumericDType::U32
                | NumericDType::U64 => runmat_accelerate_api::ProviderPrecision::F64,
            };
            runmat_accelerate_api::set_handle_precision(handle, precision);
        }
        OutputClass::Logical => {
            runmat_accelerate_api::set_handle_logical(handle, true);
            runmat_accelerate_api::set_handle_precision(
                handle,
                runmat_accelerate_api::ProviderPrecision::F64,
            );
        }
        OutputClass::Complex => {}
    }
}

struct ParsedNdgrid {
    axes: Vec<AxisData>,
    axis_lengths: Vec<usize>,
    output_count: usize,
    residency: OutputResidency,
}

impl ParsedNdgrid {
    async fn parse(args: &[Value], requested_outputs: Option<usize>) -> BuiltinResult<Self> {
        if args.is_empty() {
            return Err(ndgrid_error(&NDGRID_ERROR_MISSING_AXIS));
        }

        let mut prefer_gpu = false;
        let mut gpu_device_id = None;
        let mut axes = Vec::with_capacity(args.len());
        for (index, value) in args.iter().cloned().enumerate() {
            let axis = axis_from_value(value, index, &mut prefer_gpu, &mut gpu_device_id).await?;
            axes.push(axis);
        }

        let output_count = match requested_outputs {
            Some(count) if args.len() == 1 || count <= axes.len() => count,
            Some(_) => return Err(ndgrid_error(&NDGRID_ERROR_TOO_MANY_OUTPUTS)),
            None => 1,
        };

        let expanded_axes = if args.len() == 1 && output_count > 1 {
            vec![axes[0].clone(); output_count]
        } else {
            axes
        };

        let axis_lengths = expanded_axes
            .iter()
            .map(|axis| axis.len)
            .collect::<Vec<_>>();
        let residency = if prefer_gpu {
            OutputResidency::Gpu {
                device_id: gpu_device_id.unwrap_or(0),
            }
        } else {
            OutputResidency::Host
        };

        Ok(Self {
            axes: expanded_axes,
            axis_lengths,
            output_count,
            residency,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputClass {
    Real(NumericDType),
    Logical,
    Complex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputResidency {
    Host,
    Gpu { device_id: u32 },
}

#[derive(Clone)]
struct AxisData {
    values: Vec<(f64, f64)>,
    len: usize,
    class: OutputClass,
    numeric_storage: Option<NumericStorage>,
    integer_data: Option<IntegerComplexStorage>,
    gpu_real: Option<GpuTensorHandle>,
}

async fn axis_from_value(
    value: Value,
    index: usize,
    prefer_gpu: &mut bool,
    gpu_device_id: &mut Option<u32>,
) -> BuiltinResult<AxisData> {
    match value {
        Value::Tensor(tensor) => axis_from_tensor(tensor, index),
        Value::LogicalArray(logical) => axis_from_logical(logical, index),
        Value::Num(value) => Ok(real_scalar_axis(value, NumericDType::F64)),
        Value::Int(value) => Ok(integer_scalar_axis(value)),
        Value::Bool(value) => Ok(logical_scalar_axis(value)),
        Value::Complex(re, im) => Ok(AxisData {
            values: vec![(re, im)],
            len: 1,
            class: OutputClass::Complex,
            numeric_storage: None,
            integer_data: None,
            gpu_real: None,
        }),
        Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor, index),
        Value::GpuTensor(handle) => {
            *prefer_gpu = true;
            gpu_device_id.get_or_insert(handle.device_id);
            if is_vector_shape(&handle.shape)
                && runmat_accelerate_api::handle_storage(&handle)
                    != GpuTensorStorage::ComplexInterleaved
            {
                let class = if runmat_accelerate_api::handle_is_logical(&handle) {
                    OutputClass::Logical
                } else {
                    OutputClass::Real(
                        runmat_accelerate_api::handle_precision(&handle)
                            .map(precision_to_dtype)
                            .unwrap_or(NumericDType::F64),
                    )
                };
                return Ok(AxisData {
                    values: Vec::new(),
                    len: vector_len_from_shape(&handle.shape),
                    class,
                    numeric_storage: None,
                    integer_data: None,
                    gpu_real: Some(handle),
                });
            }

            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
            match gathered {
                Value::Tensor(tensor) => axis_from_tensor(tensor, index),
                Value::LogicalArray(logical) => axis_from_logical(logical, index),
                Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor, index),
                Value::Num(value) => Ok(real_scalar_axis(value, NumericDType::F64)),
                Value::Bool(value) => Ok(logical_scalar_axis(value)),
                Value::Complex(re, im) => Ok(AxisData {
                    values: vec![(re, im)],
                    len: 1,
                    class: OutputClass::Complex,
                    numeric_storage: None,
                    integer_data: None,
                    gpu_real: None,
                }),
                other => Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INVALID_AXIS,
                    format!("argument {} got {other:?}", index + 1),
                )),
            }
        }
        other => Err(ndgrid_error_with_detail(
            &NDGRID_ERROR_INVALID_AXIS,
            format!("argument {} got {other:?}", index + 1),
        )),
    }
}

fn precision_to_dtype(precision: runmat_accelerate_api::ProviderPrecision) -> NumericDType {
    match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
    }
}

fn real_scalar_axis(value: f64, dtype: NumericDType) -> AxisData {
    let storage = match dtype {
        NumericDType::F64 => NumericStorage::F64(vec![value]),
        NumericDType::F32 => NumericStorage::F32(vec![value as f32]),
        _ => unreachable!("real scalar axes use a floating dtype"),
    };
    AxisData {
        values: Vec::new(),
        len: 1,
        class: OutputClass::Real(dtype),
        numeric_storage: Some(storage),
        integer_data: None,
        gpu_real: None,
    }
}

fn integer_scalar_axis(value: IntValue) -> AxisData {
    let storage = NumericStorage::from(IntegerStorage::from_scalar(value));
    let dtype = storage.numeric_dtype();
    AxisData {
        values: Vec::new(),
        len: 1,
        class: OutputClass::Real(dtype),
        numeric_storage: Some(storage),
        integer_data: None,
        gpu_real: None,
    }
}

fn logical_scalar_axis(value: bool) -> AxisData {
    AxisData {
        values: vec![(if value { 1.0 } else { 0.0 }, 0.0)],
        len: 1,
        class: OutputClass::Logical,
        numeric_storage: None,
        integer_data: None,
        gpu_real: None,
    }
}

fn axis_from_tensor(tensor: Tensor, index: usize) -> BuiltinResult<AxisData> {
    if !is_vector_shape(&tensor.shape) {
        return Err(ndgrid_error_with_detail(
            &NDGRID_ERROR_INVALID_AXIS,
            format!(
                "argument {} must be a vector, got shape {:?}",
                index + 1,
                tensor.shape
            ),
        ));
    }
    let storage = tensor
        .into_numeric_storage()
        .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
    let len = storage.len();
    let dtype = storage.numeric_dtype();
    Ok(AxisData {
        len,
        values: Vec::new(),
        class: OutputClass::Real(dtype),
        numeric_storage: Some(storage),
        integer_data: None,
        gpu_real: None,
    })
}

fn axis_from_logical(logical: LogicalArray, index: usize) -> BuiltinResult<AxisData> {
    if !is_vector_shape(&logical.shape) {
        return Err(ndgrid_error_with_detail(
            &NDGRID_ERROR_INVALID_AXIS,
            format!(
                "argument {} must be a vector, got shape {:?}",
                index + 1,
                logical.shape
            ),
        ));
    }
    let values = logical
        .data
        .into_iter()
        .map(|value| (if value != 0 { 1.0 } else { 0.0 }, 0.0))
        .collect::<Vec<_>>();
    Ok(AxisData {
        len: values.len(),
        values,
        class: OutputClass::Logical,
        numeric_storage: None,
        integer_data: None,
        gpu_real: None,
    })
}

fn axis_from_complex_tensor(tensor: ComplexTensor, index: usize) -> BuiltinResult<AxisData> {
    if !is_vector_shape(&tensor.shape) {
        return Err(ndgrid_error_with_detail(
            &NDGRID_ERROR_INVALID_AXIS,
            format!(
                "argument {} must be a vector, got shape {:?}",
                index + 1,
                tensor.shape
            ),
        ));
    }
    let len = tensor::complex_tensor_element_len(&tensor);
    let values = if tensor.integer_storage().is_some() {
        Vec::new()
    } else {
        tensor.materialize_f64()
    };
    Ok(AxisData {
        len,
        values,
        class: OutputClass::Complex,
        numeric_storage: None,
        integer_data: tensor.integer_storage().cloned(),
        gpu_real: None,
    })
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    shape.iter().filter(|&&extent| extent > 1).count() <= 1
}

fn vector_len_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else if shape.contains(&0) {
        0
    } else {
        shape.iter().copied().max().unwrap_or(0)
    }
}

async fn axis_to_host_async(axis: &AxisData) -> BuiltinResult<AxisData> {
    let Some(handle) = axis.gpu_real.as_ref() else {
        return Ok(axis.clone());
    };
    match gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await? {
        Value::Tensor(tensor) => axis_from_tensor(tensor, 0),
        Value::LogicalArray(logical) => axis_from_logical(logical, 0),
        Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor, 0),
        Value::Num(value) => Ok(real_scalar_axis(value, NumericDType::F64)),
        Value::Bool(value) => Ok(logical_scalar_axis(value)),
        Value::Complex(re, im) => Ok(AxisData {
            values: vec![(re, im)],
            len: 1,
            class: OutputClass::Complex,
            numeric_storage: None,
            integer_data: None,
            gpu_real: None,
        }),
        other => Err(ndgrid_error_with_detail(
            &NDGRID_ERROR_INTERNAL,
            format!("expected numeric GPU axis, got {other:?}"),
        )),
    }
}

fn output_shape_from_lengths(lengths: &[usize]) -> Vec<usize> {
    if lengths.len() == 1 {
        vec![lengths[0], 1]
    } else {
        lengths.to_vec()
    }
}

fn checked_element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().copied().try_fold(1usize, |acc, extent| {
        acc.checked_mul(extent).ok_or_else(|| {
            ndgrid_error_with_detail(
                &NDGRID_ERROR_INTERNAL,
                "output size exceeds platform limits",
            )
        })
    })
}

fn build_host_outputs(
    axes: &[AxisData],
    output_shape: &[usize],
    output_count: usize,
) -> BuiltinResult<Vec<GridOutput>> {
    let total = checked_element_count(output_shape)?;
    let strides = column_major_strides(output_shape)?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(output_count)
        .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err.to_string()))?;
    for axis in axes.iter().take(output_count) {
        let mut data = Vec::new();
        data.try_reserve_exact(total)
            .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err.to_string()))?;
        outputs.push(GridOutput {
            shape: output_shape.to_vec(),
            data,
            class: axis.class,
            numeric_storage: axis
                .numeric_storage
                .as_ref()
                .map(|storage| storage.zeros_like(total)),
            integer_data: axis.integer_data.clone(),
            integer_values: axis
                .integer_data
                .as_ref()
                .map(|_| (Vec::with_capacity(total), Vec::with_capacity(total))),
        });
    }

    for linear_idx in 0..total {
        for dim in 0..output_count {
            let coord = coordinate_for_dim(linear_idx, output_shape, &strides, dim);
            let output = &mut outputs[dim];
            if let (Some(storage), Some((real, imag))) = (
                axes[dim].integer_data.as_ref(),
                output.integer_values.as_mut(),
            ) {
                real.push(
                    storage
                        .real
                        .value_at(coord)
                        .expect("axis coordinate is in bounds"),
                );
                imag.push(
                    storage
                        .imag
                        .value_at(coord)
                        .expect("axis coordinate is in bounds"),
                );
            } else if let (Some(storage), Some(output_storage)) = (
                axes[dim].numeric_storage.as_ref(),
                output.numeric_storage.as_mut(),
            ) {
                output_storage
                    .set_value(
                        linear_idx,
                        storage
                            .value_at(coord)
                            .expect("axis coordinate is in bounds"),
                    )
                    .expect("axis and output numeric classes match");
            } else {
                let value = axes[dim].values.get(coord).copied().unwrap_or((0.0, 0.0));
                output.data.push(value);
            }
        }
    }

    Ok(outputs)
}

fn column_major_strides(shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut strides = Vec::new();
    strides
        .try_reserve_exact(shape.len())
        .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err.to_string()))?;
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride.checked_mul(extent).ok_or_else(|| {
            ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, "stride exceeds platform limits")
        })?;
    }
    Ok(strides)
}

fn coordinate_for_dim(idx: usize, shape: &[usize], strides: &[usize], dim: usize) -> usize {
    let extent = shape.get(dim).copied().unwrap_or(1);
    if extent == 0 {
        0
    } else {
        (idx / strides[dim]) % extent
    }
}

struct GridOutput {
    shape: Vec<usize>,
    data: Vec<(f64, f64)>,
    class: OutputClass,
    numeric_storage: Option<NumericStorage>,
    integer_data: Option<IntegerComplexStorage>,
    integer_values: Option<(Vec<IntValue>, Vec<IntValue>)>,
}

impl GridOutput {
    fn to_value(&self, residency: OutputResidency) -> BuiltinResult<Value> {
        match self.class {
            OutputClass::Real(dtype) => self.to_numeric_value(dtype, residency),
            OutputClass::Logical => self.to_logical_value(residency),
            OutputClass::Complex => self.to_complex_value(residency),
        }
    }

    fn to_numeric_value(
        &self,
        dtype: NumericDType,
        residency: OutputResidency,
    ) -> BuiltinResult<Value> {
        let storage = self.numeric_storage.as_ref().ok_or_else(|| {
            ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, "missing numeric output storage")
        })?;
        if storage.numeric_dtype() != dtype {
            return Err(ndgrid_error_with_detail(
                &NDGRID_ERROR_INTERNAL,
                "numeric output storage class mismatch",
            ));
        }
        let tensor = Tensor::from_numeric_storage(storage.clone(), self.shape.clone())
            .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
        match residency {
            OutputResidency::Host if dtype != NumericDType::F32 => {
                Ok(tensor::tensor_into_value(tensor))
            }
            OutputResidency::Host => Ok(Value::Tensor(tensor)),
            OutputResidency::Gpu { .. }
                if matches!(
                    dtype,
                    NumericDType::I8
                        | NumericDType::I16
                        | NumericDType::I32
                        | NumericDType::I64
                        | NumericDType::U8
                        | NumericDType::U16
                        | NumericDType::U32
                        | NumericDType::U64
                ) =>
            {
                Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INTERNAL,
                    "GPU-resident typed integer outputs are not supported",
                ))
            }
            OutputResidency::Gpu { device_id } => to_gpu_tensor_value(tensor, device_id),
        }
    }

    fn to_logical_value(&self, residency: OutputResidency) -> BuiltinResult<Value> {
        let mut data = Vec::new();
        data.try_reserve_exact(self.data.len())
            .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err.to_string()))?;
        for &(re, im) in &self.data {
            if im != 0.0 {
                return Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INTERNAL,
                    "cannot represent complex values in a logical output",
                ));
            }
            data.push(if re != 0.0 { 1 } else { 0 });
        }
        let logical = LogicalArray::new(data, self.shape.clone())
            .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
        match residency {
            OutputResidency::Host if logical.data.len() == 1 => {
                Ok(Value::Bool(logical.data[0] != 0))
            }
            OutputResidency::Host => Ok(Value::LogicalArray(logical)),
            OutputResidency::Gpu { device_id } => to_logical_gpu_value(logical, device_id),
        }
    }

    fn to_complex_value(&self, residency: OutputResidency) -> BuiltinResult<Value> {
        if let (Some(prototype), Some((real, imag))) =
            (self.integer_data.as_ref(), self.integer_values.as_ref())
        {
            let real = prototype
                .real
                .from_exact_values_like(real.clone())
                .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
            let imag = prototype
                .imag
                .from_exact_values_like(imag.clone())
                .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
            let storage = IntegerComplexStorage::new(real, imag)
                .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
            let tensor = ComplexTensor::new_integer(storage, self.shape.clone())
                .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
            return match residency {
                OutputResidency::Host => Ok(Value::ComplexTensor(tensor)),
                OutputResidency::Gpu { .. } => Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INTERNAL,
                    "GPU-resident typed complex integer outputs are not supported",
                )),
            };
        }
        let mut data = Vec::new();
        data.try_reserve_exact(self.data.len())
            .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err.to_string()))?;
        data.extend_from_slice(&self.data);
        let tensor = ComplexTensor::new(data, self.shape.clone())
            .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
        match residency {
            OutputResidency::Host => Ok(complex_tensor_into_value(tensor)),
            OutputResidency::Gpu { device_id } => to_complex_gpu_tensor_value(tensor, device_id),
        }
    }
}

fn to_gpu_tensor_value(tensor: Tensor, device_id: u32) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_device(device_id)
        .or_else(runmat_accelerate_api::provider)
    {
        let dtype = tensor.numeric_dtype();
        let data = tensor.materialize_f64();
        let view = HostTensorView {
            data: &data,
            shape: &tensor.shape,
        };
        match provider.upload(&view) {
            Ok(handle) => {
                let precision = match dtype {
                    NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
                    NumericDType::F64
                    | NumericDType::I8
                    | NumericDType::I16
                    | NumericDType::I32
                    | NumericDType::I64
                    | NumericDType::U8
                    | NumericDType::U16
                    | NumericDType::U32
                    | NumericDType::U64 => runmat_accelerate_api::ProviderPrecision::F64,
                };
                runmat_accelerate_api::set_handle_precision(&handle, precision);
                return Ok(Value::GpuTensor(handle));
            }
            Err(err) => {
                return Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INTERNAL,
                    err.to_string(),
                ))
            }
        }
    }
    Err(ndgrid_error_with_detail(
        &NDGRID_ERROR_INTERNAL,
        "no acceleration provider registered for GPU output",
    ))
}

fn to_logical_gpu_value(logical: LogicalArray, device_id: u32) -> BuiltinResult<Value> {
    let tensor = tensor::logical_to_tensor(&logical)
        .map_err(|err| ndgrid_error_with_detail(&NDGRID_ERROR_INTERNAL, err))?;
    if let Some(provider) = runmat_accelerate_api::provider_for_device(device_id)
        .or_else(runmat_accelerate_api::provider)
    {
        let data = tensor.as_f64_slice().ok_or_else(|| {
            ndgrid_error_with_detail(
                &NDGRID_ERROR_INTERNAL,
                "logical GPU upload requires double storage",
            )
        })?;
        let view = HostTensorView {
            data,
            shape: &tensor.shape,
        };
        match provider.upload(&view) {
            Ok(handle) => return Ok(gpu_helpers::logical_gpu_value(handle)),
            Err(err) => {
                return Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INTERNAL,
                    err.to_string(),
                ))
            }
        }
    }
    Err(ndgrid_error_with_detail(
        &NDGRID_ERROR_INTERNAL,
        "no acceleration provider registered for GPU output",
    ))
}

fn to_complex_gpu_tensor_value(tensor: ComplexTensor, device_id: u32) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_device(device_id)
        .or_else(runmat_accelerate_api::provider)
    {
        match gpu_helpers::upload_complex_tensor(provider, &tensor) {
            Ok(handle) => return Ok(gpu_helpers::complex_gpu_value(handle)),
            Err(err) => {
                return Err(ndgrid_error_with_detail(
                    &NDGRID_ERROR_INTERNAL,
                    err.to_string(),
                ))
            }
        }
    }
    Err(ndgrid_error_with_detail(
        &NDGRID_ERROR_INTERNAL,
        "no acceleration provider registered for GPU output",
    ))
}

pub struct NdgridEval {
    outputs: Vec<NdgridOutput>,
    residency: OutputResidency,
}

impl NdgridEval {
    pub async fn output(&self, dim: usize) -> BuiltinResult<Value> {
        let Some(output) = self.outputs.get(dim) else {
            return Err(ndgrid_error(&NDGRID_ERROR_TOO_MANY_OUTPUTS));
        };
        match output {
            NdgridOutput::Host(output) => output.to_value(self.residency),
            NdgridOutput::Gpu(handle) => Ok(Value::GpuTensor(handle.clone())),
        }
    }
}

enum NdgridOutput {
    Host(GridOutput),
    Gpu(GpuTensorHandle),
}

fn ndgrid_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn ndgrid_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn eval(args: &[Value], outputs: Option<usize>) -> BuiltinResult<NdgridEval> {
        block_on(evaluate(args, outputs))
    }

    fn output(eval: &NdgridEval, dim: usize) -> BuiltinResult<Value> {
        block_on(eval.output(dim))
    }

    fn call(args: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = crate::output_count::push_output_count(outputs);
        block_on(ndgrid_builtin(args))
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_single_output_is_column_vector() {
        let x = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let eval = eval(&[Value::Tensor(x)], None).expect("ndgrid");
        assert_eq!(eval.outputs.len(), 1);
        let x_out = test_support::gather(output(&eval, 0).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 1]);
        assert_eq!(x_out.materialize_f64(), vec![1.0, 2.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_single_input_uses_requested_output_count_as_rank() {
        let x = tensor(vec![10.0, 20.0], vec![1, 2]);
        let value = call(vec![Value::Tensor(x)], Some(3)).expect("ndgrid");
        let Value::OutputList(outputs) = value else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 3);

        let x1 = test_support::gather(outputs[0].clone()).expect("x1");
        assert_eq!(x1.shape, vec![2, 2, 2]);
        assert_eq!(
            x1.materialize_f64(),
            vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        );
        let x2 = test_support::gather(outputs[1].clone()).expect("x2");
        assert_eq!(
            x2.materialize_f64(),
            vec![10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0]
        );
        let x3 = test_support::gather(outputs[2].clone()).expect("x3");
        assert_eq!(
            x3.materialize_f64(),
            vec![10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_rectangular_two_inputs() {
        let x = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let y = tensor(vec![10.0, 20.0], vec![2, 1]);
        let eval = eval(&[Value::Tensor(x), Value::Tensor(y)], Some(2)).expect("ndgrid");
        assert_eq!(eval.outputs.len(), 2);

        let x_out = test_support::gather(output(&eval, 0).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 2]);
        assert_eq!(x_out.materialize_f64(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let y_out = test_support::gather(output(&eval, 1).expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![3, 2]);
        assert_eq!(
            y_out.materialize_f64(),
            vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_three_inputs_volume() {
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let y = tensor(vec![5.0, 6.0, 7.0], vec![3, 1]);
        let z = tensor(vec![0.0, 1.0], vec![1, 2]);
        let eval = eval(
            &[Value::Tensor(x), Value::Tensor(y), Value::Tensor(z)],
            Some(3),
        )
        .expect("ndgrid");
        assert_eq!(eval.outputs.len(), 3);

        let x_out = test_support::gather(output(&eval, 0).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![2, 3, 2]);
        assert_eq!(
            x_out.materialize_f64(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
        let z_out = test_support::gather(output(&eval, 2).expect("Z")).expect("host");
        assert_eq!(
            z_out.materialize_f64(),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_complex_inputs_produce_complex_outputs() {
        let x = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let y = tensor(vec![10.0], vec![1, 1]);
        let eval = eval(&[Value::ComplexTensor(x), Value::Tensor(y)], Some(2)).expect("ndgrid");
        let Value::ComplexTensor(out) = output(&eval, 0).expect("X") else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.materialize_f64(), vec![(1.0, 2.0), (3.0, -4.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_preserves_output_class_per_axis() {
        let x = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 0.0)], vec![1, 2]).unwrap();
        let y = tensor(vec![10.0, 20.0], vec![1, 2]);
        let eval = eval(&[Value::ComplexTensor(x), Value::Tensor(y)], Some(2)).expect("ndgrid");

        let Value::ComplexTensor(x_out) = output(&eval, 0).expect("X") else {
            panic!("expected complex X");
        };
        assert_eq!(x_out.shape, vec![2, 2]);
        let Value::Tensor(y_out) = output(&eval, 1).expect("Y") else {
            panic!("expected real Y");
        };
        assert_eq!(y_out.shape, vec![2, 2]);
        assert_eq!(y_out.numeric_dtype(), NumericDType::F64);
        assert_eq!(y_out.materialize_f64(), vec![10.0, 10.0, 20.0, 20.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_preserves_native_f32_and_exact_wide_u64_per_axis() {
        let x = Tensor::from_f32(vec![1.25, 2.5], vec![1, 2]).unwrap();
        let y_values = vec![9_007_199_254_740_993, u64::MAX];
        let y = Tensor::new_integer(IntegerStorage::U64(y_values.clone()), vec![1, 2]).unwrap();
        let eval = eval(&[Value::Tensor(x), Value::Tensor(y)], Some(2)).expect("ndgrid");

        let Value::Tensor(x_out) = output(&eval, 0).expect("X") else {
            panic!("expected single X");
        };
        assert_eq!(x_out.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            x_out.into_numeric_storage().unwrap(),
            NumericStorage::F32(vec![1.25, 2.5, 1.25, 2.5])
        );
        let Value::Tensor(y_out) = output(&eval, 1).expect("Y") else {
            panic!("expected uint64 Y");
        };
        assert_eq!(y_out.numeric_dtype(), NumericDType::U64);
        assert_eq!(
            y_out.into_numeric_storage().unwrap(),
            NumericStorage::U64(vec![y_values[0], y_values[0], y_values[1], y_values[1],])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_preserves_all_exact_integer_axis_classes() {
        let storages = [
            IntegerStorage::I8(vec![-2, 7]),
            IntegerStorage::I16(vec![-300, 400]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let source_values = storage.exact_values();
            let expected = storage
                .from_exact_values_like(vec![
                    source_values[0].clone(),
                    source_values[1].clone(),
                    source_values[0].clone(),
                    source_values[1].clone(),
                ])
                .expect("expected storage");
            let expected_second = storage
                .from_exact_values_like(vec![
                    source_values[0].clone(),
                    source_values[0].clone(),
                    source_values[1].clone(),
                    source_values[1].clone(),
                ])
                .expect("second expected storage");
            let axis = Tensor::new_integer(storage, vec![1, 2]).expect("integer axis");
            let eval = eval(&[Value::Tensor(axis)], Some(2)).expect("ndgrid");
            let Value::Tensor(first_output) = output(&eval, 0).expect("grid output") else {
                panic!("expected exact integer tensor");
            };
            assert_eq!(first_output.shape, vec![2, 2]);
            assert_eq!(first_output.integer_storage(), Some(&expected));
            let Value::Tensor(second_output) = output(&eval, 1).expect("second grid output") else {
                panic!("expected second exact integer tensor");
            };
            assert_eq!(second_output.integer_storage(), Some(&expected_second));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_reads_typed_integer_axis_length_from_storage_without_mirror() {
        let axis = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("axis");

        let eval = eval(&[Value::Tensor(axis)], Some(2)).expect("ndgrid");
        let Value::Tensor(first) = output(&eval, 0).expect("first grid") else {
            panic!("expected typed integer first grid");
        };
        let Value::Tensor(second) = output(&eval, 1).expect("second grid") else {
            panic!("expected typed integer second grid");
        };
        assert_eq!(first.shape, vec![2, 2]);
        assert_eq!(
            first.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                u64::MAX,
                9_007_199_254_740_993,
                u64::MAX,
            ]))
        );
        assert_eq!(
            second.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_993,
                u64::MAX,
                u64::MAX,
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_reads_typed_complex_integer_axis_length_from_storage_without_mirror() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![-3, 5]),
            IntegerStorage::I16(vec![7, -11]),
        )
        .unwrap();
        let axis = ComplexTensor::new_integer(storage, vec![1, 2]).expect("axis");

        let eval = eval(&[Value::ComplexTensor(axis)], Some(2)).expect("ndgrid");
        let Value::ComplexTensor(first) = output(&eval, 0).expect("first grid") else {
            panic!("expected typed complex integer first grid");
        };
        assert_eq!(first.shape, vec![2, 2]);
        assert_eq!(
            first.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![-3, 5, -3, 5]),
                    IntegerStorage::I16(vec![7, -11, 7, -11]),
                )
                .unwrap()
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_preserves_exact_integer_scalars() {
        let eval = eval(&[Value::Int(IntValue::U64(u64::MAX))], None).expect("ndgrid");
        assert_eq!(
            output(&eval, 0).expect("grid output"),
            Value::Int(IntValue::U64(u64::MAX))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_preserves_logical_axis_outputs() {
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        let y = tensor(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let eval =
            eval(&[Value::LogicalArray(logical), Value::Tensor(y)], Some(2)).expect("ndgrid");

        let Value::LogicalArray(x_out) = output(&eval, 0).expect("X") else {
            panic!("expected logical X");
        };
        assert_eq!(x_out.shape, vec![2, 3]);
        assert_eq!(x_out.data, vec![1, 0, 1, 0, 1, 0]);
        let y_out = test_support::gather(output(&eval, 1).expect("Y")).expect("Y host");
        assert_eq!(
            y_out.materialize_f64(),
            vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_default_multi_input_materializes_only_first_output() {
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let y = tensor(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let z = tensor(vec![100.0, 200.0], vec![1, 2]);
        let eval = eval(
            &[Value::Tensor(x), Value::Tensor(y), Value::Tensor(z)],
            None,
        )
        .expect("ndgrid");
        assert_eq!(eval.outputs.len(), 1);

        let x_out = test_support::gather(output(&eval, 0).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![2, 3, 2]);
        assert_eq!(
            x_out.materialize_f64(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_complex_class_is_preserved_for_zero_imaginary_inputs() {
        let x = ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![1, 2]).unwrap();
        let y = tensor(vec![10.0], vec![1, 1]);
        let eval = eval(&[Value::ComplexTensor(x), Value::Tensor(y)], Some(2)).expect("ndgrid");
        let Value::ComplexTensor(out) = output(&eval, 0).expect("X") else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.materialize_f64(), vec![(1.0, 0.0), (2.0, 0.0)]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_complex_class_is_preserved_for_nan_imaginary_inputs() {
        let x = ComplexTensor::new(vec![(1.0, f64::NAN)], vec![1, 1]).unwrap();
        let eval = eval(&[Value::ComplexTensor(x)], None).expect("ndgrid");
        let Value::Complex(re, im) = output(&eval, 0).expect("X") else {
            panic!("expected complex scalar");
        };
        assert_eq!(re, 1.0);
        assert!(im.is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_rejects_non_vector_input() {
        let matrix = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let err = match eval(&[Value::Tensor(matrix)], None) {
            Ok(_) => panic!("expected invalid axis"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), NDGRID_ERROR_INVALID_AXIS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_rejects_too_many_outputs_for_multiple_inputs() {
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let y = tensor(vec![3.0, 4.0], vec![1, 2]);
        let err = match eval(&[Value::Tensor(x), Value::Tensor(y)], Some(3)) {
            Ok(_) => panic!("expected too many outputs"),
            Err(err) => err,
        };
        assert_eq!(err.identifier(), NDGRID_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_zero_requested_outputs_returns_empty_list() {
        let x = tensor(vec![1.0, 2.0], vec![1, 2]);
        let value = call(vec![Value::Tensor(x)], Some(0)).expect("ndgrid");
        assert_eq!(value, Value::OutputList(Vec::new()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_gpu_inputs_use_provider_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let x = tensor(vec![1.0, 2.0], vec![1, 2]);
            let y = tensor(vec![10.0, 20.0, 30.0], vec![1, 3]);
            let x_handle = provider
                .upload(&HostTensorView {
                    data: &x.materialize_f64(),
                    shape: &x.shape,
                })
                .expect("upload x");
            let y_handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");
            provider.reset_telemetry();
            let eval = eval(
                &[
                    Value::GpuTensor(x_handle.clone()),
                    Value::GpuTensor(y_handle.clone()),
                ],
                Some(2),
            )
            .expect("ndgrid");
            assert!(matches!(output(&eval, 0).expect("X"), Value::GpuTensor(_)));
            let y_out = output(&eval, 1).expect("Y");
            assert!(matches!(y_out, Value::GpuTensor(_)));
            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.download_bytes, 0);
            assert_eq!(telemetry.upload_bytes, 0);
            let gathered = test_support::gather(y_out).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
            assert_eq!(
                gathered.materialize_f64(),
                vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ndgrid_gpu_inputs_keep_original_handle_shapes() {
        test_support::with_test_provider(|provider| {
            let x = tensor(vec![1.0, 2.0], vec![1, 2]);
            let y = tensor(vec![10.0, 20.0, 30.0], vec![3, 1]);
            let x_handle = provider
                .upload(&HostTensorView {
                    data: &x.materialize_f64(),
                    shape: &x.shape,
                })
                .expect("upload x");
            let y_handle = provider
                .upload(&HostTensorView {
                    data: &y.materialize_f64(),
                    shape: &y.shape,
                })
                .expect("upload y");

            let eval = eval(
                &[
                    Value::GpuTensor(x_handle.clone()),
                    Value::GpuTensor(y_handle.clone()),
                ],
                None,
            )
            .expect("ndgrid");
            assert!(matches!(output(&eval, 0).expect("X"), Value::GpuTensor(_)));

            let x_after =
                test_support::gather(Value::GpuTensor(x_handle)).expect("gather original x");
            let y_after =
                test_support::gather(Value::GpuTensor(y_handle)).expect("gather original y");
            assert_eq!(x_after.shape, vec![1, 2]);
            assert_eq!(x_after.materialize_f64(), vec![1.0, 2.0]);
            assert_eq!(y_after.shape, vec![3, 1]);
            assert_eq!(y_after.materialize_f64(), vec![10.0, 20.0, 30.0]);
        });
    }
}
