//! MATLAB-compatible `meshgrid` builtin with GPU-aware semantics.

use std::cmp::max;

use log::warn;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, ResolveContext, Tensor, Type,
    Value,
};

use crate::builtins::array::type_resolvers::size_vector_len;
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::meshgrid")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "meshgrid",
    op_kind: GpuOpKind::Custom("array_construct"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("meshgrid")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may supply a dedicated meshgrid hook; until then the runtime builds grids on the host and uploads them when GPU residency is requested.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("meshgrid")
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::meshgrid")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "meshgrid",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Meshgrid explicitly materialises dense coordinate arrays and therefore bypasses fusion.",
};

fn meshgrid_type(args: &[Type], _context: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Unknown;
    }
    let mut axis_count = args.len();
    if axis_count >= 2 && matches!(args[axis_count - 2], Type::String) {
        axis_count = axis_count.saturating_sub(2);
    }
    if axis_count == 0 {
        return Type::Unknown;
    }
    let axis_args = &args[..axis_count];
    let len_x = axis_args.get(0).and_then(size_vector_len);
    let len_y = axis_args.get(1).and_then(size_vector_len).or(len_x);
    let len_z = axis_args.get(2).and_then(size_vector_len);
    let shape = if axis_count >= 3 {
        vec![len_y, len_x, len_z]
    } else {
        vec![len_y, len_x]
    };
    Type::Tensor { shape: Some(shape) }
}

const MESHGRID_OUTPUT_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along X-axis.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along Y-axis.",
    },
];

const MESHGRID_OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along X-axis.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid coordinates along Y-axis.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grid coordinates along Z-axis.",
    },
];

const MESHGRID_SIG_X_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-axis vector.",
}];

const MESHGRID_SIG_XY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
];

const MESHGRID_SIG_XYZ_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Z-axis vector.",
    },
];

const MESHGRID_SIG_X_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling class/device residency.",
    },
];

const MESHGRID_SIG_XY_LIKE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling class/device residency.",
    },
];

const MESHGRID_SIG_XYZ_LIKE_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Z-axis vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling class/device residency.",
    },
];

const MESHGRID_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x)",
        inputs: &MESHGRID_SIG_X_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x, y)",
        inputs: &MESHGRID_SIG_XY_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = meshgrid(x, y, z)",
        inputs: &MESHGRID_SIG_XYZ_INPUTS,
        outputs: &MESHGRID_OUTPUT_XYZ,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x, \"like\", prototype)",
        inputs: &MESHGRID_SIG_X_LIKE_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y] = meshgrid(x, y, \"like\", prototype)",
        inputs: &MESHGRID_SIG_XY_LIKE_INPUTS,
        outputs: &MESHGRID_OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,Z] = meshgrid(x, y, z, \"like\", prototype)",
        inputs: &MESHGRID_SIG_XYZ_LIKE_INPUTS,
        outputs: &MESHGRID_OUTPUT_XYZ,
    },
];

const MESHGRID_ERRORS: [BuiltinErrorDescriptor; 11] = [
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.MISSING_AXIS",
        identifier: None,
        when: "No axis vectors are provided.",
        message: "meshgrid: at least one input vector is required",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.TOO_MANY_AXES",
        identifier: None,
        when: "More than three axis vectors are provided.",
        message: "meshgrid: expected at most three input vectors",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "meshgrid: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "meshgrid: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.LIKE_POSITION",
        identifier: None,
        when: "The 'like' keyword is in an invalid position or not final.",
        message: "meshgrid: 'like' must be the final argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not recognized.",
        message: "meshgrid: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.INVALID_AXIS_INPUT",
        identifier: None,
        when: "Axis inputs are non-numeric or non-vector shapes.",
        message: "meshgrid: input argument must be numeric vector data",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.INVALID_PROTOTYPE",
        identifier: None,
        when: "The 'like' prototype is unsupported.",
        message: "meshgrid: prototypes must be numeric arrays",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.OUTPUT_COUNT_EXCEEDED",
        identifier: None,
        when: "Requested outputs exceed available outputs for provided axes.",
        message:
            "meshgrid: supports at most two outputs for 2-axis inputs and three for 3-axis inputs",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.THIRD_OUTPUT_UNAVAILABLE",
        identifier: None,
        when: "A third output is requested without supplying a Z-axis vector.",
        message: "meshgrid: third output requested but no Z vector was supplied",
    },
    BuiltinErrorDescriptor {
        code: "RM.MESHGRID.COMPLEX_REAL_CONVERSION",
        identifier: None,
        when: "Complex axis values cannot be represented in requested real output class.",
        message: "meshgrid: cannot represent complex values in a real output",
    },
];

pub const MESHGRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MESHGRID_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MESHGRID_ERRORS,
};

#[runtime_builtin(
    name = "meshgrid",
    category = "array/creation",
    summary = "Generate coordinate matrices for 2-D and 3-D grids.",
    keywords = "meshgrid,grid,gpu,like,3d",
    accel = "array_construct",
    type_resolver(meshgrid_type),
    descriptor(crate::builtins::array::creation::meshgrid::MESHGRID_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::meshgrid"
)]
async fn meshgrid_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let available = eval.output_count();
        if out_count > available {
            let msg = if available == 2 {
                "meshgrid with two inputs supports at most two outputs"
            } else {
                "meshgrid supports at most three outputs"
            };
            return Err(builtin_error(msg));
        }
        let mut outputs = Vec::with_capacity(out_count);
        let first = eval.first().await?;
        outputs.push(first);
        if out_count >= 2 {
            outputs.push(eval.second().await?);
        }
        if out_count >= 3 {
            outputs.push(eval.third().await?);
        }
        return Ok(Value::OutputList(outputs));
    }
    eval.first().await
}

/// Evaluate the `meshgrid` builtin once and reuse the result for multiple outputs.
pub async fn evaluate(args: &[Value]) -> crate::BuiltinResult<MeshgridEval> {
    let parsed = ParsedMeshgrid::parse(args).await?;
    let (x_axis, y_axis, z_axis) = normalise_axes(&parsed.axes);

    let require_complex = parsed.axes.iter().any(|axis| axis.is_complex);

    let target_class = match &parsed.template {
        OutputTemplate::Default => {
            if require_complex {
                PrototypeClass::Complex
            } else {
                PrototypeClass::Real
            }
        }
        OutputTemplate::Like(spec) => {
            if require_complex {
                PrototypeClass::Complex
            } else {
                spec.class
            }
        }
    };

    let target_residency = match &parsed.template {
        OutputTemplate::Default => {
            if parsed.prefer_gpu {
                DevicePreference::Gpu
            } else {
                DevicePreference::Host
            }
        }
        OutputTemplate::Like(spec) => spec.residency,
    };

    let mut outputs: Vec<MeshgridOutput> = Vec::new();

    if matches!(target_residency, DevicePreference::Gpu) {
        if let Some(gpu) = try_meshgrid_gpu_from_vector_axes(&x_axis, &y_axis, z_axis.as_ref())? {
            outputs = gpu;
        }
    }

    if outputs.is_empty() {
        // Host fallback: ensure we have host axis values materialized.
        let x_host = axis_to_host_async(&x_axis).await?;
        let y_host = axis_to_host_async(&y_axis).await?;
        let z_host = match z_axis.as_ref() {
            Some(axis) => Some(axis_to_host_async(axis).await?),
            None => None,
        };
        outputs = build_outputs(&x_host, &y_host, z_host.as_ref())
            .into_iter()
            .map(MeshgridOutput::Host)
            .collect();
    }

    Ok(MeshgridEval {
        outputs,
        target_class,
        target_residency,
    })
}

#[derive(Clone)]
struct ParsedMeshgrid {
    axes: Vec<AxisData>,
    template: OutputTemplate,
    prefer_gpu: bool,
}

impl ParsedMeshgrid {
    async fn parse(args: &[Value]) -> crate::BuiltinResult<Self> {
        if args.is_empty() {
            return Err(builtin_error(
                "meshgrid: at least one input vector is required",
            ));
        }
        let mut axis_values: Vec<Value> = Vec::new();
        let mut like_proto: Option<Value> = None;
        let mut prefer_gpu = false;
        let mut idx = 0;
        while idx < args.len() {
            let value = args[idx].clone();
            if let Some(keyword) = keyword_of(&value) {
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "meshgrid: multiple 'like' specifications are not supported",
                            ));
                        }
                        if axis_values.is_empty() {
                            return Err(builtin_error(
                                "meshgrid: 'like' must follow at least one input vector",
                            ));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("meshgrid: expected prototype after 'like'"));
                        };
                        like_proto = Some(proto);
                        idx += 2;
                        if idx < args.len() {
                            return Err(builtin_error(
                                "meshgrid: 'like' must be the final argument",
                            ));
                        }
                        break;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "meshgrid: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if let Value::GpuTensor(_) = value {
                prefer_gpu = true;
            }
            axis_values.push(value);
            idx += 1;
        }

        if axis_values.is_empty() {
            return Err(builtin_error(
                "meshgrid: at least one input vector is required",
            ));
        }
        if axis_values.len() > 3 {
            return Err(builtin_error(
                "meshgrid: expected at most three input vectors",
            ));
        }

        let mut axes = Vec::with_capacity(max(axis_values.len(), 2));
        for (i, value) in axis_values.into_iter().enumerate() {
            let mut consumed_gpu = false;
            let data = axis_from_value(value, i, &mut consumed_gpu).await?;
            if consumed_gpu {
                prefer_gpu = true;
            }
            axes.push(data);
        }

        if !prefer_gpu {
            if let Some(max_len) = axes.iter().map(|axis| axis.len).max() {
                if max_len > 0
                    && !axes
                        .iter()
                        .any(|axis| axis.real_integer_data.is_some() || axis.integer_data.is_some())
                    && sequence_gpu_preference(max_len, SequenceIntent::MeshAxis, false).prefer_gpu
                {
                    prefer_gpu = true;
                }
            }
        }

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(analyse_like_prototype(&proto)?)
        } else {
            OutputTemplate::Default
        };

        Ok(Self {
            axes,
            template,
            prefer_gpu,
        })
    }
}

#[derive(Clone)]
enum OutputTemplate {
    Default,
    Like(PrototypeSpec),
}

#[derive(Clone)]
struct PrototypeSpec {
    residency: DevicePreference,
    class: PrototypeClass,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PrototypeClass {
    Real,
    Complex,
}

#[derive(Clone, Copy)]
enum DevicePreference {
    Host,
    Gpu,
}

fn analyse_like_prototype(proto: &Value) -> crate::BuiltinResult<PrototypeSpec> {
    match proto {
        Value::GpuTensor(handle) => {
            let class = if runmat_accelerate_api::handle_storage(handle)
                == GpuTensorStorage::ComplexInterleaved
            {
                PrototypeClass::Complex
            } else {
                PrototypeClass::Real
            };
            Ok(PrototypeSpec {
                residency: DevicePreference::Gpu,
                class,
            })
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => Ok(PrototypeSpec {
            residency: DevicePreference::Host,
            class: PrototypeClass::Complex,
        }),
        Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_) => Ok(PrototypeSpec {
            residency: DevicePreference::Host,
            class: PrototypeClass::Real,
        }),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => Err(builtin_error(
            "meshgrid: prototypes must be numeric or gpuArray values",
        )),
        Value::Symbolic(_) => Err(builtin_error(
            "meshgrid: prototypes must be numeric or gpuArray values",
        )),
        Value::Cell(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(builtin_error("meshgrid: prototypes must be numeric arrays")),
    }
}

#[derive(Clone)]
struct AxisData {
    values: Vec<(f64, f64)>,
    len: usize,
    is_complex: bool,
    real_integer_data: Option<IntegerStorage>,
    integer_data: Option<IntegerComplexStorage>,
    gpu_real: Option<GpuTensorHandle>,
}

async fn axis_from_value(
    value: Value,
    index: usize,
    prefer_gpu: &mut bool,
) -> crate::BuiltinResult<AxisData> {
    match value {
        Value::Tensor(tensor) => axis_from_tensor(tensor, index),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)?;
            axis_from_tensor(tensor, index)
        }
        Value::Num(n) => Ok(AxisData {
            values: vec![(n, 0.0)],
            len: 1,
            is_complex: false,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }),
        Value::Int(i) => Ok(AxisData {
            values: Vec::new(),
            len: 1,
            is_complex: false,
            real_integer_data: Some(IntegerStorage::from_scalar(i)),
            integer_data: None,
            gpu_real: None,
        }),
        Value::Bool(b) => Ok(AxisData {
            values: vec![(if b { 1.0 } else { 0.0 }, 0.0)],
            len: 1,
            is_complex: false,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }),
        Value::Complex(re, im) => Ok(AxisData {
            values: vec![(re, im)],
            len: 1,
            is_complex: im != 0.0,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }),
        Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor, index),
        Value::GpuTensor(handle) => {
            let is_complex = runmat_accelerate_api::handle_storage(&handle)
                == GpuTensorStorage::ComplexInterleaved;
            // Fast path: if the gpuArray is vector-like, keep it on-device and avoid a download.
            // We'll validate any non-vector shapes by gathering below.
            if is_vector_shape(&handle.shape) && !is_complex {
                *prefer_gpu = true;
                return Ok(AxisData {
                    values: Vec::new(),
                    len: vector_len_from_shape(&handle.shape),
                    is_complex,
                    real_integer_data: None,
                    integer_data: None,
                    gpu_real: Some(handle),
                });
            }

            // Fallback: gather to validate / recover axes from meshgrid matrices.
            *prefer_gpu = true;
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?;
            match gathered {
                Value::Tensor(tensor) => {
                    if is_vector_shape(&tensor.shape) {
                        *prefer_gpu = true;
                    }
                    axis_from_tensor(tensor, index)
                }
                Value::ComplexTensor(tensor) => {
                    if is_vector_shape(&tensor.shape) {
                        *prefer_gpu = true;
                    }
                    axis_from_complex_tensor(tensor, index)
                }
                other => Err(builtin_error(format!(
                    "meshgrid: input argument {} must be numeric, got {other:?}",
                    index + 1
                ))),
            }
        }
        other => Err(builtin_error(format!(
            "meshgrid: input argument {} must be numeric, got {other:?}",
            index + 1
        ))),
    }
}

fn axis_from_tensor(tensor: Tensor, index: usize) -> crate::BuiltinResult<AxisData> {
    if is_vector_shape(&tensor.shape) {
        let values = if tensor.integer_data.is_some() {
            Vec::new()
        } else {
            let values = tensor::tensor_values_f64_cow(&tensor);
            values.iter().map(|&value| (value, 0.0)).collect()
        };
        return Ok(AxisData {
            len: tensor.data.len(),
            values,
            is_complex: false,
            real_integer_data: tensor.integer_data,
            integer_data: None,
            gpu_real: None,
        });
    }

    // Be slightly more permissive than MATLAB: if the input is already a meshgrid-style
    // coordinate matrix, accept it and recover the original axis vector.
    //
    // This is a pragmatic compatibility shim for cases where callers already have
    // coordinate matrices (X/Y) and pass them through `meshgrid` again.
    if let Some(axis) = axis_from_meshgrid_matrix_real(&tensor, index)? {
        return Ok(axis);
    }

    Err(builtin_error(format!(
        "meshgrid: input argument {} must be a vector (1xN or Nx1), got shape {:?}",
        index + 1,
        tensor.shape
    )))
}

fn axis_from_complex_tensor(tensor: ComplexTensor, index: usize) -> crate::BuiltinResult<AxisData> {
    if is_vector_shape(&tensor.shape) {
        let is_complex = tensor
            .data
            .iter()
            .any(|&(_, imag)| !imag.is_nan() && imag != 0.0);
        let ComplexTensor {
            data, integer_data, ..
        } = tensor;
        return Ok(AxisData {
            len: data.len(),
            values: data,
            is_complex,
            real_integer_data: None,
            integer_data,
            gpu_real: None,
        });
    }

    if tensor.integer_data.is_some() {
        return Err(builtin_error(format!(
            "meshgrid: input argument {} must be a vector (1xN or Nx1), got shape {:?}",
            index + 1,
            tensor.shape
        )));
    }

    if let Some(axis) = axis_from_meshgrid_matrix_complex(&tensor, index)? {
        return Ok(axis);
    }

    Err(builtin_error(format!(
        "meshgrid: input argument {} must be a vector (1xN or Nx1), got shape {:?}",
        index + 1,
        tensor.shape
    )))
}

fn axis_from_meshgrid_matrix_real(
    tensor: &Tensor,
    index: usize,
) -> crate::BuiltinResult<Option<AxisData>> {
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return Ok(None),
    };
    if rows <= 1 || cols <= 1 {
        return Ok(None);
    }

    if let Some(storage) = tensor.integer_storage() {
        return Ok(axis_from_integer_meshgrid_matrix(
            storage, rows, cols, index,
        ));
    }

    // Index 0 is expected to be the X-axis: a meshgrid X matrix has identical rows.
    // Index 1 is expected to be the Y-axis: a meshgrid Y matrix has identical columns.
    let expect_rows_constant = index == 0;

    if expect_rows_constant {
        if !matrix_rows_are_identical_real(tensor, rows, cols) {
            return Ok(None);
        }
        // Extract the first row as the axis vector (length = cols).
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            let idx = rows * col;
            values.push((tensor.data[idx], 0.0));
        }
        return Ok(Some(AxisData {
            len: values.len(),
            values,
            is_complex: false,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }));
    }

    if !matrix_cols_are_identical_real(tensor, rows, cols) {
        return Ok(None);
    }
    // Extract the first column as the axis vector (length = rows).
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        values.push((tensor.data[row], 0.0));
    }
    Ok(Some(AxisData {
        len: values.len(),
        values,
        is_complex: false,
        real_integer_data: None,
        integer_data: None,
        gpu_real: None,
    }))
}

fn axis_from_integer_meshgrid_matrix(
    storage: &IntegerStorage,
    rows: usize,
    cols: usize,
    index: usize,
) -> Option<AxisData> {
    let expect_rows_constant = index == 0;
    let values = storage.exact_values();
    if expect_rows_constant {
        for row in 1..rows {
            for col in 0..cols {
                if values[row + rows * col] != values[rows * col] {
                    return None;
                }
            }
        }
        let axis_values = (0..cols).map(|col| values[rows * col].clone()).collect();
        let storage = storage.from_exact_values_like(axis_values).ok()?;
        return Some(AxisData {
            values: Vec::new(),
            len: cols,
            is_complex: false,
            real_integer_data: Some(storage),
            integer_data: None,
            gpu_real: None,
        });
    }

    for col in 1..cols {
        for row in 0..rows {
            if values[row + rows * col] != values[row] {
                return None;
            }
        }
    }
    let axis_values = (0..rows).map(|row| values[row].clone()).collect();
    let storage = storage.from_exact_values_like(axis_values).ok()?;
    Some(AxisData {
        values: Vec::new(),
        len: rows,
        is_complex: false,
        real_integer_data: Some(storage),
        integer_data: None,
        gpu_real: None,
    })
}

fn axis_from_meshgrid_matrix_complex(
    tensor: &ComplexTensor,
    index: usize,
) -> crate::BuiltinResult<Option<AxisData>> {
    let (rows, cols) = match tensor.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => return Ok(None),
    };
    if rows <= 1 || cols <= 1 {
        return Ok(None);
    }

    let expect_rows_constant = index == 0;
    if expect_rows_constant {
        if !matrix_rows_are_identical_complex(tensor, rows, cols) {
            return Ok(None);
        }
        let mut values = Vec::with_capacity(cols);
        for col in 0..cols {
            let idx = rows * col;
            values.push(tensor.data[idx]);
        }
        let is_complex = values.iter().any(|&(_, im)| !im.is_nan() && im != 0.0);
        return Ok(Some(AxisData {
            len: values.len(),
            values,
            is_complex,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }));
    }

    if !matrix_cols_are_identical_complex(tensor, rows, cols) {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        values.push(tensor.data[row]);
    }
    let is_complex = values.iter().any(|&(_, im)| !im.is_nan() && im != 0.0);
    Ok(Some(AxisData {
        len: values.len(),
        values,
        is_complex,
        real_integer_data: None,
        integer_data: None,
        gpu_real: None,
    }))
}

fn matrix_rows_are_identical_real(tensor: &Tensor, rows: usize, cols: usize) -> bool {
    for row in 1..rows {
        for col in 0..cols {
            let idx0 = rows * col;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_cols_are_identical_real(tensor: &Tensor, rows: usize, cols: usize) -> bool {
    for col in 1..cols {
        for row in 0..rows {
            let idx0 = row;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_rows_are_identical_complex(tensor: &ComplexTensor, rows: usize, cols: usize) -> bool {
    for row in 1..rows {
        for col in 0..cols {
            let idx0 = rows * col;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn matrix_cols_are_identical_complex(tensor: &ComplexTensor, rows: usize, cols: usize) -> bool {
    for col in 1..cols {
        for row in 0..rows {
            let idx0 = row;
            let idx = row + rows * col;
            if tensor.data[idx] != tensor.data[idx0] {
                return false;
            }
        }
    }
    true
}

fn is_vector_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    let mut non_singleton = 0usize;
    for &dim in shape {
        if dim > 1 {
            non_singleton += 1;
        }
    }
    non_singleton <= 1
}

fn vector_len_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape.iter().copied().max().unwrap_or(0)
}

async fn axis_to_host_async(axis: &AxisData) -> crate::BuiltinResult<AxisData> {
    if axis.gpu_real.is_none() {
        return Ok(axis.clone());
    }
    let handle = axis.gpu_real.as_ref().expect("checked gpu_real is_some");
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    // Index is only used for error messages; value came from a validated vector-like handle.
    match gathered {
        Value::Tensor(tensor) => axis_from_tensor(tensor, 0),
        Value::ComplexTensor(tensor) => axis_from_complex_tensor(tensor, 0),
        Value::Num(n) => Ok(AxisData {
            values: vec![(n, 0.0)],
            len: 1,
            is_complex: false,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }),
        Value::Complex(re, im) => Ok(AxisData {
            values: vec![(re, im)],
            len: 1,
            is_complex: im != 0.0,
            real_integer_data: None,
            integer_data: None,
            gpu_real: None,
        }),
        other => Err(builtin_error(format!(
            "meshgrid: expected numeric GPU axis, got {other:?}"
        ))),
    }
}

fn try_meshgrid_gpu_from_vector_axes(
    x_axis: &AxisData,
    y_axis: &AxisData,
    z_axis: Option<&AxisData>,
) -> crate::BuiltinResult<Option<Vec<MeshgridOutput>>> {
    let Some(x_handle) = x_axis.gpu_real.as_ref() else {
        return Ok(None);
    };
    let Some(y_handle) = y_axis.gpu_real.as_ref() else {
        return Ok(None);
    };

    let z_handle = match z_axis {
        Some(axis) => match axis.gpu_real.as_ref() {
            Some(h) => Some(h),
            None => return Ok(None),
        },
        None => None,
    };

    let Some(provider) = runmat_accelerate_api::provider_for_handle(x_handle) else {
        return Ok(None);
    };
    let Some(y_provider) = runmat_accelerate_api::provider_for_handle(y_handle) else {
        return Ok(None);
    };
    if y_provider.device_id() != provider.device_id() {
        return Ok(None);
    }
    if let Some(z) = z_handle {
        let Some(z_provider) = runmat_accelerate_api::provider_for_handle(z) else {
            return Ok(None);
        };
        if z_provider.device_id() != provider.device_id() {
            return Ok(None);
        }
    }

    let nx = x_axis.len;
    let ny = y_axis.len;
    let nz = z_axis.map(|axis| axis.len).unwrap_or(1);

    // Reshape axis vectors (metadata-only) so repmat can build full grids on-device.
    let x_row = provider
        .reshape(x_handle, &[1, nx])
        .map_err(|e| builtin_error(format!("meshgrid: reshape X failed: {e}")))?;
    let y_col = provider
        .reshape(y_handle, &[ny, 1])
        .map_err(|e| builtin_error(format!("meshgrid: reshape Y failed: {e}")))?;

    let mut outputs = Vec::with_capacity(if z_handle.is_some() { 3 } else { 2 });
    if let Some(z) = z_handle {
        let x_base = provider
            .reshape(&x_row, &[1, nx, 1])
            .map_err(|e| builtin_error(format!("meshgrid: reshape X(3d) failed: {e}")))?;
        let y_base = provider
            .reshape(&y_col, &[ny, 1, 1])
            .map_err(|e| builtin_error(format!("meshgrid: reshape Y(3d) failed: {e}")))?;

        let x_grid = provider
            .repmat(&x_base, &[ny, 1, nz])
            .map_err(|e| builtin_error(format!("meshgrid: repmat X failed: {e}")))?;
        let y_grid = provider
            .repmat(&y_base, &[1, nx, nz])
            .map_err(|e| builtin_error(format!("meshgrid: repmat Y failed: {e}")))?;

        outputs.push(MeshgridOutput::Gpu(x_grid));
        outputs.push(MeshgridOutput::Gpu(y_grid));
        let z_axis_row = provider
            .reshape(z, &[1, nz])
            .map_err(|e| builtin_error(format!("meshgrid: reshape Z failed: {e}")))?;
        let z_base = provider
            .reshape(&z_axis_row, &[1, 1, nz])
            .map_err(|e| builtin_error(format!("meshgrid: reshape Z(3d) failed: {e}")))?;
        let z_grid = provider
            .repmat(&z_base, &[ny, nx, 1])
            .map_err(|e| builtin_error(format!("meshgrid: repmat Z failed: {e}")))?;
        outputs.push(MeshgridOutput::Gpu(z_grid));
    } else {
        let x_grid = provider
            .repmat(&x_row, &[ny, 1])
            .map_err(|e| builtin_error(format!("meshgrid: repmat X failed: {e}")))?;
        let y_grid = provider
            .repmat(&y_col, &[1, nx])
            .map_err(|e| builtin_error(format!("meshgrid: repmat Y failed: {e}")))?;
        outputs.push(MeshgridOutput::Gpu(x_grid));
        outputs.push(MeshgridOutput::Gpu(y_grid));
    }

    Ok(Some(outputs))
}

fn normalise_axes(axes: &[AxisData]) -> (AxisData, AxisData, Option<AxisData>) {
    match axes.len() {
        1 => {
            let x = axes[0].clone();
            (x.clone(), x, None)
        }
        2 => {
            let x = axes[0].clone();
            let y = axes[1].clone();
            (x, y, None)
        }
        3 => {
            let x = axes[0].clone();
            let y = axes[1].clone();
            let z = axes[2].clone();
            (x, y, Some(z))
        }
        _ => unreachable!(),
    }
}

fn build_outputs(
    x_axis: &AxisData,
    y_axis: &AxisData,
    z_axis: Option<&AxisData>,
) -> Vec<GridOutput> {
    let nx = x_axis.len;
    let ny = y_axis.len;
    let nz = z_axis.map(|axis| axis.len).unwrap_or(1);
    let total = nx * ny * nz;
    let mut x_data = Vec::with_capacity(total);
    let mut y_data = Vec::with_capacity(total);
    let mut z_data = z_axis.map(|_| Vec::with_capacity(total));
    let mut x_real_integer_values = x_axis
        .real_integer_data
        .as_ref()
        .map(|_| Vec::with_capacity(total));
    let mut y_real_integer_values = y_axis
        .real_integer_data
        .as_ref()
        .map(|_| Vec::with_capacity(total));
    let mut z_real_integer_values = z_axis.and_then(|axis| {
        axis.real_integer_data
            .as_ref()
            .map(|_| Vec::with_capacity(total))
    });
    let mut x_integer_values = x_axis
        .integer_data
        .as_ref()
        .map(|_| (Vec::with_capacity(total), Vec::with_capacity(total)));
    let mut y_integer_values = y_axis
        .integer_data
        .as_ref()
        .map(|_| (Vec::with_capacity(total), Vec::with_capacity(total)));
    let mut z_integer_values = z_axis.and_then(|axis| {
        axis.integer_data
            .as_ref()
            .map(|_| (Vec::with_capacity(total), Vec::with_capacity(total)))
    });

    for k in 0..nz {
        for col in 0..nx {
            for row in 0..ny {
                append_axis_value(
                    x_axis,
                    col,
                    &mut x_data,
                    x_real_integer_values.as_mut(),
                    x_integer_values.as_mut(),
                );
                append_axis_value(
                    y_axis,
                    row,
                    &mut y_data,
                    y_real_integer_values.as_mut(),
                    y_integer_values.as_mut(),
                );
                if let Some(ref mut z_vec) = z_data {
                    append_axis_value(
                        z_axis.expect("z output has a z axis"),
                        k,
                        z_vec,
                        z_real_integer_values.as_mut(),
                        z_integer_values.as_mut(),
                    );
                }
            }
        }
    }

    let mut outputs = Vec::new();
    let base_shape = if nz == 1 {
        vec![ny, nx]
    } else {
        vec![ny, nx, nz]
    };
    outputs.push(GridOutput {
        shape: base_shape.clone(),
        data: x_data,
        real_integer_data: x_axis.real_integer_data.clone(),
        real_integer_values: x_real_integer_values,
        integer_data: x_axis.integer_data.clone(),
        integer_values: x_integer_values,
    });
    outputs.push(GridOutput {
        shape: base_shape.clone(),
        data: y_data,
        real_integer_data: y_axis.real_integer_data.clone(),
        real_integer_values: y_real_integer_values,
        integer_data: y_axis.integer_data.clone(),
        integer_values: y_integer_values,
    });
    if let Some(z_vec) = z_data {
        outputs.push(GridOutput {
            shape: base_shape,
            data: z_vec,
            real_integer_data: z_axis.and_then(|axis| axis.real_integer_data.clone()),
            real_integer_values: z_real_integer_values,
            integer_data: z_axis.and_then(|axis| axis.integer_data.clone()),
            integer_values: z_integer_values,
        });
    }
    outputs
}

fn append_axis_value(
    axis: &AxisData,
    index: usize,
    data: &mut Vec<(f64, f64)>,
    real_integer_values: Option<&mut Vec<IntValue>>,
    integer_values: Option<&mut (Vec<IntValue>, Vec<IntValue>)>,
) {
    if let (Some(storage), Some(values)) = (axis.real_integer_data.as_ref(), real_integer_values) {
        values.push(storage.value_at(index).expect("axis index is in bounds"));
    } else if let (Some(storage), Some((real, imag))) = (axis.integer_data.as_ref(), integer_values)
    {
        real.push(
            storage
                .real
                .value_at(index)
                .expect("axis index is in bounds"),
        );
        imag.push(
            storage
                .imag
                .value_at(index)
                .expect("axis index is in bounds"),
        );
    } else {
        data.push(axis.values[index]);
    }
}

struct GridOutput {
    shape: Vec<usize>,
    data: Vec<(f64, f64)>,
    real_integer_data: Option<IntegerStorage>,
    real_integer_values: Option<Vec<IntValue>>,
    integer_data: Option<IntegerComplexStorage>,
    integer_values: Option<(Vec<IntValue>, Vec<IntValue>)>,
}

impl GridOutput {
    fn to_value(
        &self,
        class: PrototypeClass,
        residency: DevicePreference,
    ) -> crate::BuiltinResult<Value> {
        match class {
            PrototypeClass::Real => self.to_real_value(residency),
            PrototypeClass::Complex => self.to_complex_value(residency),
        }
    }

    fn to_real_value(&self, residency: DevicePreference) -> crate::BuiltinResult<Value> {
        if let (Some(prototype), Some(values)) = (
            self.real_integer_data.as_ref(),
            self.real_integer_values.as_ref(),
        ) {
            let storage = prototype
                .from_exact_values_like(values.clone())
                .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
            let tensor = Tensor::new_integer(storage, self.shape.clone())
                .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
            return match residency {
                DevicePreference::Host => Ok(Value::Tensor(tensor)),
                DevicePreference::Gpu => Err(builtin_error(
                    "meshgrid: GPU-resident typed integer outputs are not supported",
                )),
            };
        }
        let mut real = Vec::with_capacity(self.data.len());
        for &(re, im) in &self.data {
            if im != 0.0 {
                return Err(builtin_error(
                    "meshgrid: cannot represent complex values in a real output",
                ));
            }
            real.push(re);
        }
        let tensor = Tensor::new(real, self.shape.clone())
            .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
        match residency {
            DevicePreference::Host => Ok(tensor::tensor_into_value(tensor)),
            DevicePreference::Gpu => to_gpu_tensor_value(tensor),
        }
    }

    fn to_complex_value(&self, residency: DevicePreference) -> crate::BuiltinResult<Value> {
        if self.real_integer_data.is_some() {
            return Err(builtin_error(
                "meshgrid: complex output for typed integer axes is not supported",
            ));
        }
        if let (Some(prototype), Some((real, imag))) =
            (self.integer_data.as_ref(), self.integer_values.as_ref())
        {
            let real = prototype
                .real
                .from_exact_values_like(real.clone())
                .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
            let imag = prototype
                .imag
                .from_exact_values_like(imag.clone())
                .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
            let storage = IntegerComplexStorage::new(real, imag)
                .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
            let tensor = ComplexTensor::new_integer(storage, self.shape.clone())
                .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
            return match residency {
                DevicePreference::Host => Ok(Value::ComplexTensor(tensor)),
                DevicePreference::Gpu => Err(builtin_error(
                    "meshgrid: GPU-resident typed complex integer outputs are not supported",
                )),
            };
        }
        let tensor = ComplexTensor::new(self.data.clone(), self.shape.clone())
            .map_err(|e| builtin_error(format!("meshgrid: {e}")))?;
        match residency {
            DevicePreference::Host => Ok(complex_tensor_into_value(tensor)),
            DevicePreference::Gpu => to_complex_gpu_tensor_value(tensor),
        }
    }
}

fn to_gpu_tensor_value(tensor: Tensor) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        match provider.upload(&view) {
            Ok(handle) => return Ok(Value::GpuTensor(handle)),
            Err(err) => {
                warn!("meshgrid: failed to upload tensor to GPU, returning host array: {err}")
            }
        }
    }
    Ok(tensor::tensor_into_value(tensor))
}

fn to_complex_gpu_tensor_value(tensor: ComplexTensor) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        match gpu_helpers::upload_complex_tensor(provider, &tensor) {
            Ok(handle) => return Ok(gpu_helpers::complex_gpu_value(handle)),
            Err(err) => {
                warn!(
                    "meshgrid: failed to upload complex tensor to GPU, returning host array: {err}"
                )
            }
        }
    }
    Ok(complex_tensor_into_value(tensor))
}

fn tensor_to_complex_tensor(tensor: Tensor) -> crate::BuiltinResult<ComplexTensor> {
    let values = tensor::tensor_values_f64_cow(&tensor);
    let data: Vec<(f64, f64)> = values.iter().map(|&re| (re, 0.0)).collect();
    ComplexTensor::new(data, tensor.shape.clone())
        .map_err(|e| builtin_error(format!("meshgrid: {e}")))
}

fn tensor_to_complex_value(tensor: Tensor) -> crate::BuiltinResult<Value> {
    let complex = tensor_to_complex_tensor(tensor)?;
    Ok(complex_tensor_into_value(complex))
}

enum MeshgridOutput {
    Host(GridOutput),
    Gpu(GpuTensorHandle),
}

impl MeshgridOutput {
    async fn to_value(
        &self,
        class: PrototypeClass,
        residency: DevicePreference,
    ) -> crate::BuiltinResult<Value> {
        match self {
            MeshgridOutput::Host(host) => host.to_value(class, residency),
            MeshgridOutput::Gpu(handle) => match (class, residency) {
                (PrototypeClass::Real, DevicePreference::Gpu) => {
                    Ok(Value::GpuTensor(handle.clone()))
                }
                (PrototypeClass::Real, DevicePreference::Host) => {
                    let tensor = gpu_helpers::gather_tensor_async(handle).await?;
                    Ok(tensor::tensor_into_value(tensor))
                }
                (PrototypeClass::Complex, DevicePreference::Host) => {
                    match gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?
                    {
                        Value::ComplexTensor(tensor) => Ok(complex_tensor_into_value(tensor)),
                        Value::Complex(re, im) => Ok(Value::Complex(re, im)),
                        Value::Tensor(tensor) => tensor_to_complex_value(tensor),
                        Value::Num(n) => Ok(Value::Complex(n, 0.0)),
                        other => Err(builtin_error(format!(
                            "meshgrid: expected numeric GPU output, got {other:?}"
                        ))),
                    }
                }
                (PrototypeClass::Complex, DevicePreference::Gpu) => {
                    if runmat_accelerate_api::handle_storage(handle)
                        == GpuTensorStorage::ComplexInterleaved
                    {
                        Ok(gpu_helpers::complex_gpu_value(handle.clone()))
                    } else {
                        let tensor = gpu_helpers::gather_tensor_async(handle).await?;
                        to_complex_gpu_tensor_value(tensor_to_complex_tensor(tensor)?)
                    }
                }
            },
        }
    }
}

/// Holds the results of a `meshgrid` evaluation so multiple outputs can be
/// materialised without recomputing the grid.
pub struct MeshgridEval {
    outputs: Vec<MeshgridOutput>,
    target_class: PrototypeClass,
    target_residency: DevicePreference,
}

impl MeshgridEval {
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    pub async fn first(&self) -> crate::BuiltinResult<Value> {
        self.outputs[0]
            .to_value(self.target_class, self.target_residency)
            .await
    }

    pub async fn second(&self) -> crate::BuiltinResult<Value> {
        if self.outputs.len() < 2 {
            Err(builtin_error("meshgrid: second output unavailable"))
        } else {
            self.outputs[1]
                .to_value(self.target_class, self.target_residency)
                .await
        }
    }

    pub async fn third(&self) -> crate::BuiltinResult<Value> {
        if self.outputs.len() < 3 {
            Err(builtin_error(
                "meshgrid: third output requested but no Z vector was supplied",
            ))
        } else {
            self.outputs[2]
                .to_value(self.target_class, self.target_residency)
                .await
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;

    use runmat_accelerate_api::HostTensorView;

    fn evaluate(args: &[Value]) -> crate::BuiltinResult<MeshgridEval> {
        block_on(super::evaluate(args))
    }

    fn eval_first(eval: &MeshgridEval) -> crate::BuiltinResult<Value> {
        block_on(eval.first())
    }

    fn eval_second(eval: &MeshgridEval) -> crate::BuiltinResult<Value> {
        block_on(eval.second())
    }

    fn eval_third(eval: &MeshgridEval) -> crate::BuiltinResult<Value> {
        block_on(eval.third())
    }

    #[test]
    fn meshgrid_preserves_all_exact_real_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![-2, 7, 9]),
            IntegerStorage::I16(vec![-300, 400, 900]),
            IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
            IntegerStorage::U8(vec![0, 7, u8::MAX]),
            IntegerStorage::U16(vec![0, 700, u16::MAX]),
            IntegerStorage::U32(vec![0, 9_007_199, u32::MAX]),
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let values = storage.exact_values();
            let axis = Tensor::new_integer(storage.clone(), vec![1, 3]).expect("axis");
            let eval = evaluate(&[Value::Tensor(axis)]).expect("meshgrid");
            let Value::Tensor(x) = eval_first(&eval).expect("X") else {
                panic!("expected real integer X output");
            };
            let Value::Tensor(y) = eval_second(&eval).expect("Y") else {
                panic!("expected real integer Y output");
            };
            assert_eq!(x.shape, vec![3, 3]);
            assert_eq!(
                x.integer_storage(),
                Some(
                    &storage
                        .from_exact_values_like(vec![
                            values[0].clone(),
                            values[0].clone(),
                            values[0].clone(),
                            values[1].clone(),
                            values[1].clone(),
                            values[1].clone(),
                            values[2].clone(),
                            values[2].clone(),
                            values[2].clone(),
                        ])
                        .expect("expected X")
                )
            );
            assert_eq!(
                y.integer_storage(),
                Some(
                    &storage
                        .from_exact_values_like(vec![
                            values[0].clone(),
                            values[1].clone(),
                            values[2].clone(),
                            values[0].clone(),
                            values[1].clone(),
                            values[2].clone(),
                            values[0].clone(),
                            values[1].clone(),
                            values[2].clone(),
                        ])
                        .expect("expected Y")
                )
            );
        }

        let eval = evaluate(&[Value::Int(IntValue::U64(u64::MAX))]).expect("scalar meshgrid");
        let Value::Tensor(output) = eval_first(&eval).expect("scalar X") else {
            panic!("expected real integer scalar output");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX]))
        );
    }

    #[test]
    fn meshgrid_recovers_exact_integer_axes_from_coordinate_matrices() {
        let x = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("X axis");
        let y = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, 7]), vec![2, 1])
            .expect("Y axis");
        let initial = evaluate(&[Value::Tensor(x), Value::Tensor(y)]).expect("initial meshgrid");
        let x_grid = eval_first(&initial).expect("X grid");
        let y_grid = eval_second(&initial).expect("Y grid");
        let recovered = evaluate(&[x_grid, y_grid]).expect("recovered meshgrid");
        let Value::Tensor(x_out) = eval_first(&recovered).expect("recovered X") else {
            panic!("expected real integer X output");
        };
        let Value::Tensor(y_out) = eval_second(&recovered).expect("recovered Y") else {
            panic!("expected real integer Y output");
        };
        assert_eq!(
            x_out.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_993,
                u64::MAX,
                u64::MAX,
            ]))
        );
        assert_eq!(
            y_out.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, 7, i64::MIN, 7]))
        );
    }

    #[test]
    fn meshgrid_does_not_lossily_promote_typed_integer_axes_to_complex() {
        let axis = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .expect("axis");
        let prototype = ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).expect("prototype");
        let eval = evaluate(&[
            Value::Tensor(axis),
            Value::String("like".to_string()),
            Value::ComplexTensor(prototype),
        ])
        .expect("parse meshgrid");
        let error = eval_first(&eval).expect_err("typed integer complex output must reject");
        assert!(error
            .message()
            .contains("complex output for typed integer axes is not supported"));
    }

    #[test]
    fn meshgrid_tensor_to_complex_reads_typed_integer_storage_exactly() {
        let mut tensor =
            Tensor::new_integer(IntegerStorage::I64(vec![-3, 5]), vec![1, 2]).expect("tensor");
        tensor.data.fill(f64::NAN);

        let out = tensor_to_complex_tensor(tensor).expect("complex tensor");
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.data, vec![(-3.0, 0.0), (5.0, 0.0)]);
    }

    fn tensor_from_vec(data: Vec<f64>, rows: usize, cols: usize) -> Tensor {
        Tensor::new(data, vec![rows, cols]).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_single_input_duplicates_axis() {
        let x = tensor_from_vec(vec![-1.0, 0.0, 1.0], 1, 3);
        let eval = evaluate(&[Value::Tensor(x)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 2);
        let x_out = test_support::gather(eval_first(&eval).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 3]);
        assert_eq!(
            x_out.data,
            vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
        let y_out = test_support::gather(eval_second(&eval).expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![3, 3]);
        assert_eq!(
            y_out.data,
            vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn meshgrid_type_infers_rank_from_axis_count() {
        let ctx = ResolveContext::new(Vec::new());
        assert_eq!(
            meshgrid_type(&[Type::Num, Type::Num], &ctx),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1)])
            }
        );
        assert_eq!(
            meshgrid_type(&[Type::Num, Type::Num, Type::Num], &ctx),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(1), Some(1)])
            }
        );
    }

    #[test]
    fn meshgrid_type_uses_vector_lengths() {
        let ctx = ResolveContext::new(Vec::new());
        assert_eq!(
            meshgrid_type(
                &[
                    Type::Tensor {
                        shape: Some(vec![Some(1), Some(201)]),
                    },
                    Type::Tensor {
                        shape: Some(vec![Some(1), Some(101)]),
                    },
                ],
                &ctx,
            ),
            Type::Tensor {
                shape: Some(vec![Some(101), Some(201)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_rectangular_inputs() {
        let x = tensor_from_vec(vec![0.0, 0.5, 1.0], 1, 3);
        let y = tensor_from_vec(vec![10.0, 20.0], 2, 1);
        let eval = evaluate(&[Value::Tensor(x), Value::Tensor(y)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 2);
        let x_out = test_support::gather(eval_first(&eval).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![2, 3]);
        assert_eq!(x_out.data, vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
        let y_out = test_support::gather(eval_second(&eval).expect("Y")).expect("host");
        assert_eq!(y_out.shape, vec![2, 3]);
        assert_eq!(y_out.data, vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_three_inputs_volume() {
        let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
        let y = tensor_from_vec(vec![5.0, 6.0, 7.0], 3, 1);
        let z = tensor_from_vec(vec![0.0, 1.0], 1, 2);
        let eval =
            evaluate(&[Value::Tensor(x), Value::Tensor(y), Value::Tensor(z)]).expect("meshgrid");
        assert_eq!(eval.output_count(), 3);
        let x_out = test_support::gather(eval_first(&eval).expect("X")).expect("host");
        assert_eq!(x_out.shape, vec![3, 2, 2]);
        assert_eq!(
            x_out.data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
        let z_out = test_support::gather(eval_third(&eval).expect("Z")).expect("host");
        assert_eq!(z_out.shape, vec![3, 2, 2]);
        assert_eq!(
            z_out.data,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_like_keeps_gpu_residency() {
        test_support::with_test_provider(|provider| {
            let x = tensor_from_vec(vec![-1.0, 0.0, 1.0], 1, 3);
            let y = tensor_from_vec(vec![2.0, 4.0], 2, 1);
            let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let proto_handle = provider.upload(&proto_view).expect("upload");
            let eval = evaluate(&[
                Value::Tensor(x),
                Value::Tensor(y),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ])
            .expect("meshgrid");
            let x_value = eval_first(&eval).expect("X");
            assert!(matches!(x_value, Value::GpuTensor(_)));
            let gathered = test_support::gather(x_value).expect("gather");
            assert_eq!(gathered.shape, vec![2, 3]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_gpu_inputs_roundtrip() {
        test_support::with_test_provider(|provider| {
            let x = tensor_from_vec(vec![0.0, 0.5], 1, 2);
            let y = tensor_from_vec(vec![1.0, 2.0], 2, 1);
            let x_view = HostTensorView {
                data: &x.data,
                shape: &x.shape,
            };
            let y_view = HostTensorView {
                data: &y.data,
                shape: &y.shape,
            };
            let x_handle = provider.upload(&x_view).expect("upload");
            let y_handle = provider.upload(&y_view).expect("upload");
            let eval = evaluate(&[Value::GpuTensor(x_handle), Value::GpuTensor(y_handle)])
                .expect("meshgrid");
            assert!(matches!(eval_first(&eval).expect("X"), Value::GpuTensor(_)));
            assert!(matches!(
                eval_second(&eval).expect("Y"),
                Value::GpuTensor(_)
            ));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn meshgrid_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let x = tensor_from_vec(vec![-1.0, 0.0, 1.0, 2.0], 1, 4);
        let y = tensor_from_vec(vec![5.0, 6.0], 2, 1);

        let cpu_eval =
            evaluate(&[Value::Tensor(x.clone()), Value::Tensor(y.clone())]).expect("meshgrid cpu");
        let cpu_x =
            test_support::gather(eval_first(&cpu_eval).expect("X cpu")).expect("gather X cpu");
        let cpu_y =
            test_support::gather(eval_second(&cpu_eval).expect("Y cpu")).expect("gather Y cpu");

        let x_view = HostTensorView {
            data: &x.data,
            shape: &x.shape,
        };
        let y_view = HostTensorView {
            data: &y.data,
            shape: &y.shape,
        };
        let x_gpu = provider.upload(&x_view).expect("upload x");
        let y_gpu = provider.upload(&y_view).expect("upload y");

        let gpu_eval =
            evaluate(&[Value::GpuTensor(x_gpu), Value::GpuTensor(y_gpu)]).expect("meshgrid gpu");
        let gpu_x_value = eval_first(&gpu_eval).expect("X gpu");
        let gpu_y_value = eval_second(&gpu_eval).expect("Y gpu");

        assert!(matches!(gpu_x_value, Value::GpuTensor(_)));
        assert!(matches!(gpu_y_value, Value::GpuTensor(_)));

        let gathered_x = test_support::gather(gpu_x_value).expect("gather X gpu");
        let gathered_y = test_support::gather(gpu_y_value).expect("gather Y gpu");

        assert_eq!(gathered_x.shape, cpu_x.shape);
        assert_eq!(gathered_x.data, cpu_x.data);
        assert_eq!(gathered_y.shape, cpu_y.shape);
        assert_eq!(gathered_y.data, cpu_y.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_complex_inputs_produce_complex_outputs() {
        let complex = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let eval = evaluate(&[Value::ComplexTensor(complex)]).expect("meshgrid");
        let x_value = eval_first(&eval).expect("X");
        match x_value {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 2]);
            }
            Value::Complex(_, _) => {}
            other => panic!("expected complex output, got {other:?}"),
        }
    }

    #[test]
    fn meshgrid_like_complex_gpu_prototype_keeps_complex_residency() {
        test_support::with_test_provider(|provider| {
            let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
            let proto = ComplexTensor::new(vec![(0.0, 1.0)], vec![1, 1]).unwrap();
            let proto_handle =
                gpu_helpers::upload_complex_tensor(provider, &proto).expect("upload");

            let eval = evaluate(&[
                Value::Tensor(x),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ])
            .expect("meshgrid");
            let x_value = eval_first(&eval).expect("X");
            let Value::GpuTensor(handle) = x_value else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                .expect("gather");
            let Value::ComplexTensor(tensor) = gathered else {
                panic!("expected complex tensor");
            };
            assert_eq!(tensor.shape, vec![2, 2]);
            assert_eq!(
                tensor.data,
                vec![(1.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 0.0)]
            );
        });
    }

    #[test]
    fn meshgrid_complex_gpu_axis_stays_resident() {
        test_support::with_test_provider(|provider| {
            let axis = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
            let axis_handle = gpu_helpers::upload_complex_tensor(provider, &axis).expect("upload");

            let eval = evaluate(&[Value::GpuTensor(axis_handle)]).expect("meshgrid");
            let x_value = eval_first(&eval).expect("X");
            let Value::GpuTensor(handle) = x_value else {
                panic!("expected complex gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&handle),
                GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle)))
                .expect("gather");
            let Value::ComplexTensor(tensor) = gathered else {
                panic!("expected complex tensor");
            };
            assert_eq!(tensor.shape, vec![2, 2]);
            assert_eq!(
                tensor.data,
                vec![(1.0, 1.0), (1.0, 1.0), (2.0, -1.0), (2.0, -1.0)]
            );
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn meshgrid_wgpu_complex_axis_matches_cpu_and_stays_resident() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let axis = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let cpu_eval = evaluate(&[Value::ComplexTensor(axis.clone())]).expect("meshgrid cpu");
        let cpu_x = match eval_first(&cpu_eval).expect("X cpu") {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected cpu complex tensor, got {other:?}"),
        };

        let axis_handle = gpu_helpers::upload_complex_tensor(provider, &axis).expect("upload");
        let gpu_eval = evaluate(&[Value::GpuTensor(axis_handle)]).expect("meshgrid gpu");
        let gpu_x = eval_first(&gpu_eval).expect("X gpu");
        let Value::GpuTensor(handle) = gpu_x else {
            panic!("expected complex gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).expect("gather");
        let Value::ComplexTensor(gpu_tensor) = gathered else {
            panic!("expected complex tensor");
        };
        assert_eq!(gpu_tensor.shape, cpu_x.shape);
        assert_eq!(gpu_tensor.data, cpu_x.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn meshgrid_like_host_prototype() {
        let x = tensor_from_vec(vec![1.0, 2.0], 1, 2);
        let eval =
            evaluate(&[Value::Tensor(x), Value::from("like"), Value::Num(0.0)]).expect("meshgrid");
        let x_out = eval_first(&eval).expect("X");
        assert!(matches!(x_out, Value::Tensor(_) | Value::Num(_)));
    }
}
