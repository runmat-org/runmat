//! MATLAB-compatible `cummax` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{
    GpuTensorHandle, ProviderCummaxResult, ProviderNanMode, ProviderScanDirection,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use super::complex_cumulative_extrema;
use super::floating_cumulative_extrema::{
    self, CumulativeDirection, CumulativeExtrema, CumulativeNanMode,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cummax";

const GPU_NANFLAG_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cummax-gpu-nanflag",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cummax with an explicit missing-value flag on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CummaxGpuNanflagExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_NANFLAG_EXTENSION];

fn cummax_type(args: &[Type], ctx: &ResolveContext) -> Type {
    cumulative_numeric_type(args, ctx)
}

const CUMMAX_OUTPUT_M: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cumulative maximum values.",
}];

const CUMMAX_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
};

const CUMMAX_PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("[]"),
    description: "Dimension selector (placeholder [] keeps default dimension).",
};

const CUMMAX_PARAM_DIRECTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "direction",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"forward\""),
    description: "Scan direction: \"forward\" or \"reverse\".",
};

const CUMMAX_PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description:
        "Missing-value mode: \"includenan\"/\"includemissing\" or \"omitnan\"/\"omitmissing\".",
};

const CUMMAX_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [CUMMAX_PARAM_A];
const CUMMAX_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [CUMMAX_PARAM_A, CUMMAX_PARAM_DIM];
const CUMMAX_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIRECTION];
const CUMMAX_INPUTS_NANFLAG: [BuiltinParamDescriptor; 2] = [CUMMAX_PARAM_A, CUMMAX_PARAM_NANFLAG];
const CUMMAX_INPUTS_DIM_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIM, CUMMAX_PARAM_DIRECTION];
const CUMMAX_INPUTS_DIRECTION_DIM: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIRECTION, CUMMAX_PARAM_DIM];
const CUMMAX_INPUTS_DIM_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIM, CUMMAX_PARAM_NANFLAG];
const CUMMAX_INPUTS_NANFLAG_DIM: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_NANFLAG, CUMMAX_PARAM_DIM];
const CUMMAX_INPUTS_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_DIRECTION, CUMMAX_PARAM_NANFLAG];
const CUMMAX_INPUTS_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 3] =
    [CUMMAX_PARAM_A, CUMMAX_PARAM_NANFLAG, CUMMAX_PARAM_DIRECTION];
const CUMMAX_INPUTS_DIM_DIRECTION_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_NANFLAG,
];
const CUMMAX_INPUTS_DIM_NANFLAG_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIRECTION,
];
const CUMMAX_INPUTS_DIRECTION_DIM_NANFLAG: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_NANFLAG,
];
const CUMMAX_INPUTS_DIRECTION_NANFLAG_DIM: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIM,
];
const CUMMAX_INPUTS_NANFLAG_DIM_DIRECTION: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIM,
    CUMMAX_PARAM_DIRECTION,
];
const CUMMAX_INPUTS_NANFLAG_DIRECTION_DIM: [BuiltinParamDescriptor; 4] = [
    CUMMAX_PARAM_A,
    CUMMAX_PARAM_NANFLAG,
    CUMMAX_PARAM_DIRECTION,
    CUMMAX_PARAM_DIM,
];

const CUMMAX_SIGNATURES: [BuiltinSignatureDescriptor; 16] = [
    BuiltinSignatureDescriptor {
        label: "M = cummax(A)",
        inputs: &CUMMAX_INPUTS_CORE,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, dim)",
        inputs: &CUMMAX_INPUTS_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, direction)",
        inputs: &CUMMAX_INPUTS_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, nanflag)",
        inputs: &CUMMAX_INPUTS_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, dim, direction)",
        inputs: &CUMMAX_INPUTS_DIM_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, direction, dim)",
        inputs: &CUMMAX_INPUTS_DIRECTION_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, dim, nanflag)",
        inputs: &CUMMAX_INPUTS_DIM_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, nanflag, dim)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, direction, nanflag)",
        inputs: &CUMMAX_INPUTS_DIRECTION_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, nanflag, direction)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, dim, direction, nanflag)",
        inputs: &CUMMAX_INPUTS_DIM_DIRECTION_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, dim, nanflag, direction)",
        inputs: &CUMMAX_INPUTS_DIM_NANFLAG_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, direction, dim, nanflag)",
        inputs: &CUMMAX_INPUTS_DIRECTION_DIM_NANFLAG,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, direction, nanflag, dim)",
        inputs: &CUMMAX_INPUTS_DIRECTION_NANFLAG_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, nanflag, dim, direction)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIM_DIRECTION,
        outputs: &CUMMAX_OUTPUT_M,
    },
    BuiltinSignatureDescriptor {
        label: "M = cummax(A, nanflag, direction, dim)",
        inputs: &CUMMAX_INPUTS_NANFLAG_DIRECTION_DIM,
        outputs: &CUMMAX_OUTPUT_M,
    },
];

const CUMMAX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMAX.INVALID_ARGUMENT",
    identifier: Some("RunMat:cummax:InvalidArgument"),
    when: "Dimension, direction, or missing-value argument grammar is invalid.",
    message: "cummax: invalid argument",
};

const CUMMAX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMAX.INVALID_INPUT",
    identifier: Some("RunMat:cummax:InvalidInput"),
    when: "Input value type is unsupported for cumulative maximum reduction.",
    message: "cummax: invalid input",
};

const CUMMAX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMAX.INTERNAL",
    identifier: Some("RunMat:cummax:Internal"),
    when: "Reduction execution fails due to conversion, provider, or allocation operations.",
    message: "cummax: internal reduction failure",
};

const CUMMAX_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CUMMAX_ERROR_INVALID_ARGUMENT,
    CUMMAX_ERROR_INVALID_INPUT,
    CUMMAX_ERROR_INTERNAL,
];

pub const CUMMAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMMAX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMMAX_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Ordinary real arrays accept all eight integer classes; complex-integer ordering remains a separately tracked conformance question.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional positive scalar dimension is decoded exactly from typed integer or integer-valued floating storage.",
    },
];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "M = cummax(A, dim, direction, nanflag)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Cumulative maximum preserves integer class and exact shape in forward or reverse direction; the compatibility target exposes one public value output and rejects GPU nanflag, so internal/provider indices remain private and RunMat GPU nanflag support is mode-gated.",
    }];

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::reduction::type_resolvers::cumulative_numeric_type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::reduction::cummax")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cummax",
    op_kind: GpuOpKind::Custom("scan"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("cummax_scan")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes:
        "Providers may compute internal running-selection indices, but the public builtin exposes only cumulative values; the runtime gathers to host when hooks or options are unsupported.",
};

fn cummax_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cummax_error_with_message(error.message, error)
}

fn cummax_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    cummax_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn cummax_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cummax_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    cummax_error_with_detail(&CUMMAX_ERROR_INTERNAL, detail)
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::reduction::cummax")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cummax",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner currently lowers cummax to the runtime implementation; providers can substitute specialised scan kernels when available.",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CummaxDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CummaxNanMode {
    Include,
    Omit,
}

/// Evaluation artifact returned by `cummax` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct CummaxEvaluation {
    values: Value,
    indices: Value,
}

impl CummaxEvaluation {
    /// Consume the evaluation and return only the running maxima (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both maxima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming the evaluation.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

#[runtime_builtin(
    name = "cummax",
    category = "math/reduction",
    summary = "Compute cumulative maxima.",
    keywords = "cummax,cumulative maximum,running maximum,reverse,omitnan,gpu",
    accel = "reduction",
    type_resolver(cummax_type),
    descriptor(crate::builtins::math::reduction::cummax::CUMMAX_DESCRIPTOR),
    extensions(crate::builtins::math::reduction::cummax::EXTENSIONS),
    integer_capabilities(crate::builtins::math::reduction::cummax::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::reduction::cummax"
)]
async fn cummax_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count > 1 {
            return Err(cummax_error_with_detail(
                &CUMMAX_ERROR_INVALID_ARGUMENT,
                "cummax returns exactly one output",
            ));
        }
        return Ok(Value::OutputList(vec![eval.into_value()]));
    }
    Ok(eval.into_value())
}

/// Evaluate the builtin once and expose both outputs (value + indices).
pub async fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CummaxEvaluation> {
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(cummax_error_with_detail(
            &CUMMAX_ERROR_INVALID_INPUT,
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    let (dim, direction, nan_mode, nanflag_explicit) = parse_arguments(rest)?;
    if matches!(&value, Value::GpuTensor(_)) && nanflag_explicit {
        crate::compatibility::ensure_builtin_extension_enabled(&GPU_NANFLAG_EXTENSION, NAME)?;
    }
    match value {
        Value::GpuTensor(handle) => cummax_gpu(handle, dim, direction, nan_mode).await,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| cummax_internal_error(&e))?;
            let target_dim = dim.unwrap_or(1);
            let (values, indices) =
                cummax_complex_tensor(&tensor, target_dim, direction, nan_mode)?;
            Ok(CummaxEvaluation {
                values: complex_tensor_into_value(values),
                indices: tensor::tensor_into_value(indices),
            })
        }
        Value::ComplexTensor(ct) => {
            let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&ct.shape));
            let (values, indices) = cummax_complex_tensor(&ct, target_dim, direction, nan_mode)?;
            Ok(CummaxEvaluation {
                values: complex_tensor_into_value(values),
                indices: tensor::tensor_into_value(indices),
            })
        }
        other => cummax_host(other, dim, direction, nan_mode),
    }
}

fn parse_arguments(
    args: &[Value],
) -> BuiltinResult<(Option<usize>, CummaxDirection, CummaxNanMode, bool)> {
    if args.len() > 4 {
        return Err(cummax_error(&CUMMAX_ERROR_INVALID_ARGUMENT));
    }

    let mut dim: Option<usize> = None;
    let mut direction = CummaxDirection::Forward;
    let mut direction_set = false;
    let mut nan_mode = CummaxNanMode::Include;
    let mut nan_set = false;

    for value in args {
        match value {
            Value::Int(_) | Value::Num(_) => {
                if dim.is_some() {
                    return Err(cummax_error_with_detail(
                        &CUMMAX_ERROR_INVALID_ARGUMENT,
                        "dimension specified more than once",
                    ));
                }
                dim = Some(tensor::parse_dimension(value, "cummax").map_err(|err| {
                    cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, err)
                })?);
            }
            Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
                if dim.is_some() {
                    return Err(cummax_error_with_detail(
                        &CUMMAX_ERROR_INVALID_ARGUMENT,
                        "dimension specified more than once",
                    ));
                }
                dim = Some(tensor::parse_dimension(value, "cummax").map_err(|err| {
                    cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, err)
                })?);
            }
            Value::Tensor(t) if tensor::tensor_element_len(t) == 0 => {
                // MATLAB allows [] placeholders; ignore them.
            }
            Value::LogicalArray(l) if l.data.is_empty() => {}
            _ => {
                if let Some(text) = tensor::value_to_string(value) {
                    let keyword = text.trim().to_ascii_lowercase();
                    match keyword.as_str() {
                        "forward" => {
                            if direction_set {
                                return Err(cummax_error_with_detail(
                                    &CUMMAX_ERROR_INVALID_ARGUMENT,
                                    "direction specified more than once",
                                ));
                            }
                            direction = CummaxDirection::Forward;
                            direction_set = true;
                        }
                        "reverse" => {
                            if direction_set {
                                return Err(cummax_error_with_detail(
                                    &CUMMAX_ERROR_INVALID_ARGUMENT,
                                    "direction specified more than once",
                                ));
                            }
                            direction = CummaxDirection::Reverse;
                            direction_set = true;
                        }
                        "omitnan" | "omitmissing" => {
                            if nan_set {
                                return Err(cummax_error_with_detail(
                                    &CUMMAX_ERROR_INVALID_ARGUMENT,
                                    "missing-value handling specified more than once",
                                ));
                            }
                            nan_mode = CummaxNanMode::Omit;
                            nan_set = true;
                        }
                        "includenan" | "includemissing" => {
                            if nan_set {
                                return Err(cummax_error_with_detail(
                                    &CUMMAX_ERROR_INVALID_ARGUMENT,
                                    "missing-value handling specified more than once",
                                ));
                            }
                            nan_mode = CummaxNanMode::Include;
                            nan_set = true;
                        }
                        "" => {
                            return Err(cummax_error_with_detail(
                                &CUMMAX_ERROR_INVALID_ARGUMENT,
                                "empty string option is not supported",
                            ));
                        }
                        other => {
                            return Err(cummax_error_with_detail(
                                &CUMMAX_ERROR_INVALID_ARGUMENT,
                                format!("unrecognised option '{other}'"),
                            ));
                        }
                    }
                } else {
                    return Err(cummax_error_with_detail(
                        &CUMMAX_ERROR_INVALID_ARGUMENT,
                        format!("unsupported argument type {value:?}"),
                    ));
                }
            }
        }
    }

    Ok((dim, direction, nan_mode, nan_set))
}

fn cummax_host(
    value: Value,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<CummaxEvaluation> {
    match value {
        Value::Int(value) => {
            let storage =
                crate::builtins::math::reduction::integer_native::storage_from_scalar(&value);
            integer_cummax(&storage, vec![1, 1], dim.unwrap_or(1), direction)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
            integer_cummax(
                tensor.integer_storage().expect("checked integer storage"),
                tensor.shape.clone(),
                target_dim,
                direction,
            )
        }
        other => cummax_host_floating(other, dim, direction, nan_mode),
    }
}

fn cummax_host_floating(
    value: Value,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<CummaxEvaluation> {
    let tensor = tensor::value_into_tensor_for("cummax", value)
        .map_err(|err| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_INPUT, err))?;
    let target_dim = dim.unwrap_or_else(|| default_dimension(&tensor));
    let (values, indices) = cummax_tensor(tensor, target_dim, direction, nan_mode)?;
    Ok(CummaxEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn integer_cummax(
    storage: &runmat_builtins::IntegerStorage,
    shape: Vec<usize>,
    dim: usize,
    direction: CummaxDirection,
) -> BuiltinResult<CummaxEvaluation> {
    let result = crate::builtins::math::reduction::integer_native::cumulative_extrema(
        storage,
        &shape,
        dim,
        match direction {
            CummaxDirection::Forward => {
                crate::builtins::math::reduction::integer_native::CumulativeDirection::Forward
            }
            CummaxDirection::Reverse => {
                crate::builtins::math::reduction::integer_native::CumulativeDirection::Reverse
            }
        },
        crate::builtins::math::reduction::integer_native::CumulativeExtremaDirection::Max,
    )
    .map_err(|error| cummax_internal_error(&error))?;
    Ok(CummaxEvaluation {
        values: result.values,
        indices: result.indices,
    })
}

async fn cummax_gpu(
    handle: GpuTensorHandle,
    dim: Option<usize>,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<CummaxEvaluation> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(target) = dim {
        if target == 0 {
            return Err(cummax_error_with_detail(
                &CUMMAX_ERROR_INVALID_ARGUMENT,
                "dimension must be >= 1",
            ));
        }
    }

    let target_dim = dim.unwrap_or_else(|| default_dimension_from_shape(&handle.shape));
    if target_dim == 0 {
        return Err(cummax_error_with_detail(
            &CUMMAX_ERROR_INVALID_ARGUMENT,
            "dimension must be >= 1",
        ));
    }

    if target_dim > handle.shape.len() {
        let indices = ones_indices(&handle.shape)?;
        return Ok(CummaxEvaluation {
            values: Value::GpuTensor(handle),
            indices: tensor::tensor_into_value(indices),
        });
    }

    if runmat_accelerate_api::handle_integer_type(&handle).is_some() {
        let provider_direction = match direction {
            CummaxDirection::Forward => ProviderScanDirection::Forward,
            CummaxDirection::Reverse => ProviderScanDirection::Reverse,
        };
        let provider = runmat_accelerate_api::provider().ok_or_else(|| {
            cummax_error_with_detail(
                &CUMMAX_ERROR_INVALID_INPUT,
                "cummax: native integer gpuArray requires an acceleration provider",
            )
        })?;
        let ProviderCummaxResult { values, indices } = provider
            .integer_cummax_scan(&handle, target_dim - 1, provider_direction)
            .map_err(|err| cummax_internal_error(format!("cummax: {err}")))?;
        return Ok(CummaxEvaluation {
            values: Value::GpuTensor(values),
            indices: Value::GpuTensor(indices),
        });
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let zero_based_dim = target_dim.saturating_sub(1);
        if zero_based_dim < handle.shape.len() {
            let provider_direction = match direction {
                CummaxDirection::Forward => ProviderScanDirection::Forward,
                CummaxDirection::Reverse => ProviderScanDirection::Reverse,
            };
            let provider_nan_mode = match nan_mode {
                CummaxNanMode::Include => ProviderNanMode::Include,
                CummaxNanMode::Omit => ProviderNanMode::Omit,
            };
            if let Ok(ProviderCummaxResult { values, indices }) = provider.cummax_scan(
                &handle,
                zero_based_dim,
                provider_direction,
                provider_nan_mode,
            ) {
                return Ok(CummaxEvaluation {
                    values: Value::GpuTensor(values),
                    indices: Value::GpuTensor(indices),
                });
            }
        }
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(|err| cummax_internal_error(err.message()))?;
    let (values, indices) = cummax_tensor(tensor, target_dim, direction, nan_mode)?;
    Ok(CummaxEvaluation {
        values: tensor::tensor_into_value(values),
        indices: tensor::tensor_into_value(indices),
    })
}

fn cummax_tensor(
    tensor: Tensor,
    dim: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<(Tensor, Tensor)> {
    let shape = tensor.shape.clone();
    let storage = tensor
        .into_numeric_storage()
        .map_err(|error| cummax_internal_error(&error))?;
    floating_cumulative_extrema::cumulative_extrema(
        storage,
        shape,
        dim,
        match direction {
            CummaxDirection::Forward => CumulativeDirection::Forward,
            CummaxDirection::Reverse => CumulativeDirection::Reverse,
        },
        match nan_mode {
            CummaxNanMode::Include => CumulativeNanMode::Include,
            CummaxNanMode::Omit => CumulativeNanMode::Omit,
        },
        CumulativeExtrema::Max,
    )
    .map_err(|error| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, error))
}

fn cummax_complex_tensor(
    tensor: &ComplexTensor,
    dim: usize,
    direction: CummaxDirection,
    nan_mode: CummaxNanMode,
) -> BuiltinResult<(ComplexTensor, Tensor)> {
    complex_cumulative_extrema::cumulative_extrema(
        tensor.clone().into_complex_storage(),
        tensor.shape.clone(),
        dim,
        match direction {
            CummaxDirection::Forward => complex_cumulative_extrema::Direction::Forward,
            CummaxDirection::Reverse => complex_cumulative_extrema::Direction::Reverse,
        },
        match nan_mode {
            CummaxNanMode::Include => complex_cumulative_extrema::NanMode::Include,
            CummaxNanMode::Omit => complex_cumulative_extrema::NanMode::Omit,
        },
        complex_cumulative_extrema::Extrema::Max,
    )
    .map_err(|error| cummax_error_with_detail(&CUMMAX_ERROR_INVALID_ARGUMENT, error))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if let Some([value]) = tensor.as_f64_slice() {
        Value::Complex(value.0, value.1)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn ones_indices(shape: &[usize]) -> BuiltinResult<Tensor> {
    let len = tensor::element_count(shape);
    let data = if len == 0 {
        Vec::new()
    } else {
        vec![1.0f64; len]
    };
    Tensor::new(data, shape.to_vec()).map_err(|e| cummax_internal_error(&e))
}

fn default_dimension(tensor: &Tensor) -> usize {
    default_dimension_from_shape(&tensor.shape)
}

fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape
        .iter()
        .position(|&extent| extent != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, NumericStorage};

    #[test]
    fn cummax_type_keeps_shape() {
        let out = cummax_type(
            &[Type::Tensor {
                shape: Some(vec![Some(2), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }

    fn evaluate(value: Value, rest: &[Value]) -> BuiltinResult<CummaxEvaluation> {
        block_on(super::evaluate(value, rest))
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn cummax_descriptor_signatures_and_errors() {
        let labels: Vec<&str> = CUMMAX_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"M = cummax(A)"));
        assert!(labels.contains(&"M = cummax(A, dim)"));
        assert!(labels.contains(&"M = cummax(A, direction)"));
        assert!(labels.contains(&"M = cummax(A, nanflag)"));
        assert!(labels.contains(&"M = cummax(A, dim, direction, nanflag)"));
        assert_eq!(labels.len(), 16);
        assert!(labels.iter().all(|label| !label.contains("[M, I]")));
        assert_eq!(CUMMAX_DESCRIPTOR.output_mode, BuiltinOutputMode::Fixed);
        assert!(CUMMAX_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMAX_ERROR_INVALID_ARGUMENT.code));
        assert!(CUMMAX_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMAX_ERROR_INVALID_INPUT.code));
        assert!(CUMMAX_DESCRIPTOR
            .errors
            .iter()
            .any(|err| err.code == CUMMAX_ERROR_INTERNAL.code));
    }

    #[test]
    fn cummax_rejects_a_second_public_output() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let error = block_on(super::cummax_builtin(Value::Num(7.0), Vec::new())).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:cummax:InvalidArgument")
        );
    }

    #[test]
    fn cummax_complex_single_preserves_native_storage() {
        let input =
            ComplexTensor::from_f32(vec![(3.0, 0.0), (1.0, 0.0), (0.0, 2.0)], vec![3, 1]).unwrap();
        let values = evaluate(Value::ComplexTensor(input), &[])
            .expect("cummax")
            .into_value();
        let Value::ComplexTensor(values) = values else {
            panic!("expected native complex tensor");
        };
        assert_eq!(
            values.complex_storage(),
            &runmat_builtins::ComplexStorage::F32(vec![(3.0, 0.0), (3.0, 0.0), (3.0, 0.0)])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_scalar_returns_value_and_index() {
        let eval = evaluate(Value::Num(7.0), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        assert_eq!(values, Value::Num(7.0));
        assert_eq!(indices, Value::Num(1.0));
    }

    #[test]
    fn cummax_preserves_native_single_in_reverse() {
        let input = Tensor::from_f32(vec![1.0, 5.0, f32::NAN, 3.0], vec![4, 1]).expect("input");
        let (values, indices) = evaluate(
            Value::Tensor(input),
            &[Value::from("reverse"), Value::from("omitnan")],
        )
        .expect("cummax")
        .into_pair();
        let Value::Tensor(values) = values else {
            panic!("expected values tensor");
        };
        assert_eq!(
            values.into_numeric_storage().expect("native storage"),
            NumericStorage::F32(vec![5.0, 5.0, 3.0, 3.0])
        );
        let Value::Tensor(indices) = indices else {
            panic!("expected indices tensor");
        };
        assert_eq!(indices.materialize_f64(), vec![2.0, 2.0, 4.0, 4.0]);
    }

    #[test]
    fn cummax_integer_storage_and_reverse_indices_remain_exact() {
        let input = Tensor::new_integer(
            runmat_builtins::IntegerStorage::I64(vec![i64::MIN, -4, -5, -3]),
            vec![2, 2],
        )
        .unwrap();
        let (values, indices) = evaluate(Value::Tensor(input), &[Value::from("reverse")])
            .unwrap()
            .into_pair();
        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(
                    runmat_builtins::IntegerStorage::I64(vec![-4, -4, -3, -3]),
                    vec![2, 2]
                )
                .unwrap()
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]).unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_matrix_default_dimension() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.materialize_f64(), vec![4.0, 4.0, 2.0, 5.0, 7.0, 7.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.shape, vec![2, 3]);
                assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 1.0, 2.0, 1.0, 1.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_two_tracks_rows() {
        let tensor = Tensor::new(vec![4.0, 3.0, 2.0, 5.0, 7.0, 1.0], vec![2, 3]).unwrap();
        let args = vec![Value::Int(IntValue::I32(2))];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![4.0, 3.0, 4.0, 5.0, 7.0, 5.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 1.0, 2.0, 3.0, 2.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[test]
    fn cummax_parses_typed_integer_dimension_without_mirror() {
        let input =
            Tensor::new_integer(IntegerStorage::I16(vec![4, 3, 2, 5]), vec![2, 2]).expect("input");
        let dim = Tensor::new_integer(IntegerStorage::I32(vec![2]), vec![1, 1]).expect("dimension");

        let eval = evaluate(Value::Tensor(input), &[Value::Tensor(dim)])
            .expect("cummax dimension from typed integer tensor");
        let (values, indices) = eval.into_pair();

        assert_eq!(
            values,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![4, 3, 4, 5]), vec![2, 2])
                    .expect("dimension two values"),
            )
        );
        assert_eq!(
            indices,
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 2.0], vec![2, 2]).expect("indices"))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_reverse_direction() {
        let tensor = Tensor::new(vec![8.0, 3.0, 6.0, 2.0], vec![4, 1]).unwrap();
        let args = vec![Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![8.0, 6.0, 6.0, 2.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 3.0, 3.0, 4.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_reverse_omitnan_combo() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, f64::NAN, 2.0, 3.0, 4.0], vec![2, 3]).expect("tensor");
        let args = vec![
            Value::Int(IntValue::I32(2)),
            Value::from("reverse"),
            Value::from("omitnan"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![3.0, 5.0, 3.0, 4.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![3.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_placeholder_allows_four_arguments() {
        let tensor =
            Tensor::new(vec![1.0, 5.0, f64::NAN, 2.0, 3.0, 4.0], vec![2, 3]).expect("tensor");
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).expect("placeholder");
        let args = vec![
            Value::Tensor(placeholder),
            Value::Int(IntValue::I32(2)),
            Value::from("reverse"),
            Value::from("omitnan"),
        ];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64(), vec![3.0, 5.0, 3.0, 4.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![3.0, 1.0, 3.0, 3.0, 3.0, 3.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_omit_nan_behaviour() {
        let tensor = Tensor::new(vec![f64::NAN, 5.0, f64::NAN, 3.0], vec![4, 1]).expect("tensor");
        let args = vec![Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert!(out.materialize_f64()[0].is_nan());
                assert_eq!(out.materialize_f64()[1], 5.0);
                assert_eq!(out.materialize_f64()[2], 5.0);
                assert_eq!(out.materialize_f64()[3], 5.0);
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert!(idx.materialize_f64()[0].is_nan());
                assert_eq!(idx.materialize_f64()[1], 2.0);
                assert_eq!(idx.materialize_f64()[2], 2.0);
                assert_eq!(idx.materialize_f64()[3], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_duplicate_direction_errors() {
        let err = evaluate(
            Value::Num(1.0),
            &[Value::from("reverse"), Value::from("forward")],
        );
        match err {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMAX_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err.message().contains("direction specified more than once"));
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_duplicate_nanflag_errors() {
        let err = evaluate(
            Value::Num(1.0),
            &[Value::from("omitnan"), Value::from("includenan")],
        );
        match err {
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMAX_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err
                    .message()
                    .contains("missing-value handling specified more than once"));
            }
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_include_nan_propagates() {
        let tensor = Tensor::new(vec![1.0, f64::NAN, 3.0], vec![3, 1]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => {
                assert_eq!(out.materialize_f64()[0], 1.0);
                assert!(out.materialize_f64()[1].is_nan());
                assert!(out.materialize_f64()[2].is_nan());
            }
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64()[0], 1.0);
                assert_eq!(idx.materialize_f64()[1], 2.0);
                assert_eq!(idx.materialize_f64()[2], 2.0);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_greater_than_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let args = vec![Value::Int(IntValue::I32(4))];
        let eval = evaluate(Value::Tensor(tensor.clone()), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), tensor.materialize_f64()),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert!(idx.materialize_f64().iter().all(|v| *v == 1.0)),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_allows_empty_dimension_placeholder() {
        let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
        let placeholder = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = [Value::Tensor(placeholder), Value::from("reverse")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![3.0, 1.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 2.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_dimension_zero_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let args = [Value::Int(IntValue::I32(0))];
        match evaluate(Value::Tensor(tensor), &args) {
            Ok(_) => panic!("expected dimension error"),
            Err(err) => {
                assert_eq!(
                    error_identifier(&err),
                    CUMMAX_ERROR_INVALID_ARGUMENT.identifier
                );
                assert!(err
                    .message()
                    .contains(CUMMAX_ERROR_INVALID_ARGUMENT.message));
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_reverse_omitnan_combination() {
        let tensor =
            Tensor::new(vec![f64::NAN, 4.0, 2.0, f64::NAN, 3.0], vec![5, 1]).expect("tensor");
        let args = [Value::from("reverse"), Value::from("omitnan")];
        let eval = evaluate(Value::Tensor(tensor), &args).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::Tensor(out) => assert_eq!(out.materialize_f64(), vec![4.0, 4.0, 3.0, 3.0, 3.0]),
            other => panic!("expected tensor values, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => {
                assert_eq!(idx.materialize_f64(), vec![2.0, 2.0, 5.0, 5.0, 5.0]);
            }
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_complex_vector() {
        let tensor =
            ComplexTensor::new(vec![(3.0, 0.0), (2.0, 0.0), (2.0, 1.0)], vec![3, 1]).unwrap();
        let eval = evaluate(Value::ComplexTensor(tensor), &[]).expect("cummax");
        let (values, indices) = eval.into_pair();
        match values {
            Value::ComplexTensor(out) => {
                assert_eq!(out.materialize_f64()[0], (3.0, 0.0));
                assert_eq!(out.materialize_f64()[1], (3.0, 0.0));
                assert_eq!(out.materialize_f64()[2], (3.0, 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
        match indices {
            Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 1.0]),
            other => panic!("expected tensor indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0], vec![4, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummax");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.materialize_f64(), vec![4.0, 4.0, 7.0, 7.0]);
            assert_eq!(gathered_indices.materialize_f64(), vec![1.0, 1.0, 3.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_native_integer_gpu_reverse_values_and_indices_stay_resident() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[4, 9, 9, 7]),
                    shape: &[4, 1],
                })
                .expect("upload native integer");
            let eval = evaluate(Value::GpuTensor(handle), &[Value::from("reverse")])
                .expect("cummax reverse");
            let (values, indices) = eval.into_pair();
            let Value::GpuTensor(value_handle) = values else {
                panic!("expected GPU value tensor");
            };
            let Value::GpuTensor(index_handle) = indices else {
                panic!("expected GPU index tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&value_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(
                block_on(provider.download_integer(&value_handle))
                    .expect("download native integer values")
                    .data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![9, 9, 9, 7])
            );
            let gathered_indices =
                test_support::gather(Value::GpuTensor(index_handle)).expect("gather indices");
            assert_eq!(gathered_indices.materialize_f64(), vec![3.0, 3.0, 3.0, 4.0]);
        });
    }

    #[test]
    fn cummax_gpu_nanflag_follows_compatibility_mode() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 3.0], vec![2, 1]).expect("input");
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let args = vec![Value::from("omitnan")];
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = evaluate(Value::GpuTensor(handle.clone()), &args)
                    .expect_err("MATLAB-compatible mode");
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:compatibility:CummaxGpuNanflagExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                evaluate(Value::GpuTensor(handle), &args).expect("RunMat extension mode");
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cummax_gpu_dimension_exceeds_rank_returns_indices() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![Value::Int(IntValue::I32(5))];
            let eval = evaluate(Value::GpuTensor(handle), &args).expect("cummax");
            let (values, indices) = eval.into_pair();
            let gathered_values = test_support::gather(values).expect("gather values");
            let gathered_indices = test_support::gather(indices).expect("gather indices");
            assert_eq!(gathered_values.materialize_f64(), tensor.materialize_f64());
            assert!(gathered_indices.materialize_f64().iter().all(|v| *v == 1.0));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cummax_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 2.0, 7.0, 1.0, 5.0, 0.0], vec![3, 2]).unwrap();
        let cpu_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cummax cpu");
        let (cpu_vals, cpu_idx) = cpu_eval.into_pair();
        let expected_vals = match cpu_vals {
            Value::Tensor(t) => t,
            other => panic!("expected tensor values from cpu eval, got {other:?}"),
        };
        let expected_idx = match cpu_idx {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices from cpu eval, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("cummax gpu");
        let (gpu_vals, gpu_idx) = gpu_eval.into_pair();

        match (&gpu_vals, &gpu_idx) {
            (Value::GpuTensor(_), Value::GpuTensor(_)) => {}
            other => panic!("expected GPU tensors, got {other:?}"),
        }

        let gathered_vals = test_support::gather(gpu_vals).expect("gather values");
        let gathered_idx = test_support::gather(gpu_idx).expect("gather indices");

        assert_eq!(gathered_vals.shape, expected_vals.shape);
        assert_eq!(
            gathered_vals.materialize_f64(),
            expected_vals.materialize_f64()
        );
        assert_eq!(gathered_idx.shape, expected_idx.shape);
        assert_eq!(
            gathered_idx.materialize_f64(),
            expected_idx.materialize_f64()
        );
    }
}
