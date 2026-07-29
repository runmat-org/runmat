//! MATLAB-compatible moving-window numeric reductions.

use std::cmp::Ordering;

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostTensorView, ProviderMovingWindowEndpoints,
    ProviderMovingWindowOp, ProviderMovingWindowRequest, ProviderNanMode, ProviderStdNormalization,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::spec::{BuiltinFusionSpec, ConstantStrategy, ShapeRequirements};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

fn moving_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Moving-window reduction result.",
}];

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input numeric, logical, or gpuArray data.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Window length or two-element [kb kf] backward/forward window.",
};

const PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension along which to move the window.",
};

const PARAM_W: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "w",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Normalization flag: 0 for sample normalization, 1 for population normalization.",
};

const PARAM_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nanflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"includenan\""),
    description: "NaN handling mode: \"includenan\" or \"omitnan\".",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "NameValue",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options, including \"Endpoints\".",
};

const INPUTS_A_K: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_K];
const INPUTS_A_K_DIM: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_K, PARAM_DIM];
const INPUTS_A_K_NANFLAG: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_K, PARAM_NANFLAG];
const INPUTS_A_K_NV: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_K, PARAM_OPTIONS];
const INPUTS_A_K_NANFLAG_NV: [BuiltinParamDescriptor; 4] =
    [PARAM_A, PARAM_K, PARAM_NANFLAG, PARAM_OPTIONS];
const INPUTS_A_K_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_A, PARAM_K, PARAM_DIM, PARAM_OPTIONS];
const INPUTS_A_K_W: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_K, PARAM_W];
const INPUTS_A_K_W_DIM: [BuiltinParamDescriptor; 4] = [PARAM_A, PARAM_K, PARAM_W, PARAM_DIM];
const INPUTS_A_K_W_NANFLAG: [BuiltinParamDescriptor; 4] =
    [PARAM_A, PARAM_K, PARAM_W, PARAM_NANFLAG];
const INPUTS_A_K_W_NANFLAG_NV: [BuiltinParamDescriptor; 5] =
    [PARAM_A, PARAM_K, PARAM_W, PARAM_NANFLAG, PARAM_OPTIONS];
const INPUTS_A_K_W_DIM_NANFLAG: [BuiltinParamDescriptor; 5] =
    [PARAM_A, PARAM_K, PARAM_W, PARAM_DIM, PARAM_NANFLAG];
const INPUTS_A_K_W_NV: [BuiltinParamDescriptor; 4] = [PARAM_A, PARAM_K, PARAM_W, PARAM_OPTIONS];
const INPUTS_A_K_W_DIM_NV: [BuiltinParamDescriptor; 5] =
    [PARAM_A, PARAM_K, PARAM_W, PARAM_DIM, PARAM_OPTIONS];
const INPUTS_A_K_W_DIM_NANFLAG_NV: [BuiltinParamDescriptor; 6] = [
    PARAM_A,
    PARAM_K,
    PARAM_W,
    PARAM_DIM,
    PARAM_NANFLAG,
    PARAM_OPTIONS,
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING.INVALID_ARGUMENT",
    identifier: None,
    when: "Window length, dimension, nanflag, or name-value options are invalid.",
    message: "moving window: invalid argument",
};

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING.INVALID_INPUT",
    identifier: None,
    when: "Input values cannot be converted to a supported moving-window domain.",
    message: "moving window: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING.INTERNAL",
    identifier: None,
    when: "Moving-window execution fails due to allocation, provider, or shape operations.",
    message: "moving window: internal failure",
};

macro_rules! moving_descriptor {
    ($name:literal) => {
        const SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k)"),
                inputs: &super::INPUTS_A_K,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, dim)"),
                inputs: &super::INPUTS_A_K_DIM,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, nanflag)"),
                inputs: &super::INPUTS_A_K_NANFLAG,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, Name, Value)"),
                inputs: &super::INPUTS_A_K_NV,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, nanflag, Name, Value)"),
                inputs: &super::INPUTS_A_K_NANFLAG_NV,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, dim, Name, Value)"),
                inputs: &super::INPUTS_A_K_OPTIONS,
                outputs: &super::OUTPUT_Y,
            },
        ];

        const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
            when: super::ERROR_INVALID_ARGUMENT.when,
            message: super::ERROR_INVALID_ARGUMENT.message,
        };

        const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: super::ERROR_INVALID_INPUT.when,
            message: super::ERROR_INVALID_INPUT.message,
        };

        const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: super::ERROR_INTERNAL.when,
            message: super::ERROR_INTERNAL.message,
        };

        const ERRORS: [BuiltinErrorDescriptor; 3] =
            [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

macro_rules! moving_stdvar_descriptor {
    ($name:literal) => {
        const SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k)"),
                inputs: &super::INPUTS_A_K,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w)"),
                inputs: &super::INPUTS_A_K_W,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim)"),
                inputs: &super::INPUTS_A_K_W_DIM,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, nanflag)"),
                inputs: &super::INPUTS_A_K_NANFLAG,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, nanflag)"),
                inputs: &super::INPUTS_A_K_W_NANFLAG,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim, nanflag)"),
                inputs: &super::INPUTS_A_K_W_DIM_NANFLAG,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, Name, Value)"),
                inputs: &super::INPUTS_A_K_NV,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, Name, Value)"),
                inputs: &super::INPUTS_A_K_W_NV,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, nanflag, Name, Value)"),
                inputs: &super::INPUTS_A_K_W_NANFLAG_NV,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim, Name, Value)"),
                inputs: &super::INPUTS_A_K_W_DIM_NV,
                outputs: &super::OUTPUT_Y,
            },
            BuiltinSignatureDescriptor {
                label: concat!("M = ", $name, "(A, k, w, dim, nanflag, Name, Value)"),
                inputs: &super::INPUTS_A_K_W_DIM_NANFLAG_NV,
                outputs: &super::OUTPUT_Y,
            },
        ];

        const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
            when: super::ERROR_INVALID_ARGUMENT.when,
            message: super::ERROR_INVALID_ARGUMENT.message,
        };

        const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INVALID_INPUT"),
            identifier: Some(concat!("RunMat:", $name, ":InvalidInput")),
            when: super::ERROR_INVALID_INPUT.when,
            message: super::ERROR_INVALID_INPUT.message,
        };

        const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
            code: concat!("RM.", $name, ".INTERNAL"),
            identifier: Some(concat!("RunMat:", $name, ":Internal")),
            when: super::ERROR_INTERNAL.when,
            message: super::ERROR_INTERNAL.message,
        };

        const ERRORS: [BuiltinErrorDescriptor; 3] =
            [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &SIGNATURES,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

macro_rules! moving_fusion_spec {
    ($name:literal, $path:literal) => {
        #[runmat_macros::register_fusion_spec(builtin_path = $path)]
        pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
            name: $name,
            shape: ShapeRequirements::BroadcastCompatible,
            constant_strategy: ConstantStrategy::InlineLiteral,
            elementwise: None,
            reduction: None,
            emits_nan: true,
            notes: "Moving-window reductions materialise overlapping slices and are not fused yet.",
        };
    };
}

pub mod movsum {
    use super::*;
    moving_descriptor!("movsum");
    moving_fusion_spec!("movsum", "crate::builtins::math::reduction::moving::movsum");

    #[runtime_builtin(
        name = "movsum",
        category = "math/reduction",
        summary = "Moving sum over numeric arrays.",
        keywords = "movsum,moving window,sum,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movsum"
    )]
    pub(crate) async fn movsum_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movsum", MovingOp::Sum, args).await
    }
}

pub mod movmean {
    use super::*;
    moving_descriptor!("movmean");
    moving_fusion_spec!(
        "movmean",
        "crate::builtins::math::reduction::moving::movmean"
    );

    #[runtime_builtin(
        name = "movmean",
        category = "math/reduction",
        summary = "Moving mean over numeric arrays.",
        keywords = "movmean,moving window,mean,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movmean"
    )]
    pub(crate) async fn movmean_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movmean", MovingOp::Mean, args).await
    }
}

pub mod movprod {
    use super::*;
    moving_descriptor!("movprod");
    moving_fusion_spec!(
        "movprod",
        "crate::builtins::math::reduction::moving::movprod"
    );

    #[runtime_builtin(
        name = "movprod",
        category = "math/reduction",
        summary = "Moving product over numeric arrays.",
        keywords = "movprod,moving window,product,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movprod"
    )]
    pub(crate) async fn movprod_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movprod", MovingOp::Prod, args).await
    }
}

pub mod movmin {
    use super::*;
    moving_descriptor!("movmin");
    moving_fusion_spec!("movmin", "crate::builtins::math::reduction::moving::movmin");

    #[runtime_builtin(
        name = "movmin",
        category = "math/reduction",
        summary = "Moving minimum over numeric arrays.",
        keywords = "movmin,moving window,min,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movmin"
    )]
    pub(crate) async fn movmin_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movmin", MovingOp::Min, args).await
    }
}

pub mod movmax {
    use super::*;
    moving_descriptor!("movmax");
    moving_fusion_spec!("movmax", "crate::builtins::math::reduction::moving::movmax");

    #[runtime_builtin(
        name = "movmax",
        category = "math/reduction",
        summary = "Moving maximum over numeric arrays.",
        keywords = "movmax,moving window,max,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movmax"
    )]
    pub(crate) async fn movmax_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movmax", MovingOp::Max, args).await
    }
}

pub mod movmedian {
    use super::*;
    moving_descriptor!("movmedian");
    moving_fusion_spec!(
        "movmedian",
        "crate::builtins::math::reduction::moving::movmedian"
    );

    #[runtime_builtin(
        name = "movmedian",
        category = "math/reduction",
        summary = "Moving median over numeric arrays.",
        keywords = "movmedian,moving window,median,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movmedian"
    )]
    pub(crate) async fn movmedian_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movmedian", MovingOp::Median, args).await
    }
}

pub mod movstd {
    use super::*;
    moving_stdvar_descriptor!("movstd");
    moving_fusion_spec!("movstd", "crate::builtins::math::reduction::moving::movstd");

    #[runtime_builtin(
        name = "movstd",
        category = "math/reduction",
        summary = "Moving standard deviation over numeric arrays.",
        keywords = "movstd,moving window,standard deviation,std,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movstd"
    )]
    pub(crate) async fn movstd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movstd", MovingOp::Std, args).await
    }
}

pub mod movvar {
    use super::*;
    moving_stdvar_descriptor!("movvar");
    moving_fusion_spec!("movvar", "crate::builtins::math::reduction::moving::movvar");

    #[runtime_builtin(
        name = "movvar",
        category = "math/reduction",
        summary = "Moving variance over numeric arrays.",
        keywords = "movvar,moving window,variance,var,omitnan,endpoints",
        accel = "reduction",
        type_resolver(super::moving_type),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::math::reduction::moving::movvar"
    )]
    pub(crate) async fn movvar_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        super::moving_builtin("movvar", MovingOp::Var, args).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MovingOp {
    Sum,
    Mean,
    Prod,
    Min,
    Max,
    Median,
    Std,
    Var,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NanMode {
    Include,
    Omit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Endpoints {
    Shrink,
    Discard,
    FillDefault,
    Fill(f64),
}

#[derive(Clone, Debug)]
struct ParsedMoving {
    window: WindowSpec,
    dim: Option<usize>,
    nan_mode: NanMode,
    endpoints: Endpoints,
    sample_points: Option<Vec<f64>>,
    normalization: VarianceNormalization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarianceNormalization {
    Sample,
    Population,
}

#[derive(Clone, Copy, Debug)]
enum WindowSpec {
    Counts { before: usize, after: usize },
    Span { before: f64, after: f64 },
}

#[derive(Clone, Copy, Debug)]
struct CountWindow {
    start: usize,
    end: usize,
    fill_count: usize,
}

impl WindowSpec {
    fn count_radius(self) -> Option<(usize, usize)> {
        match self {
            WindowSpec::Counts { before, after } => Some((before, after)),
            WindowSpec::Span { .. } => None,
        }
    }

    fn checked_count_len(self, name: &'static str) -> BuiltinResult<Option<usize>> {
        match self {
            WindowSpec::Counts { before, after } => before
                .checked_add(after)
                .and_then(|sum| sum.checked_add(1))
                .map(Some)
                .ok_or_else(|| invalid_argument(name, format!("{name}: window length overflow"))),
            WindowSpec::Span { .. } => Ok(None),
        }
    }
}

impl MovingOp {
    fn default_nan_mode(self) -> NanMode {
        match self {
            MovingOp::Min | MovingOp::Max => NanMode::Omit,
            MovingOp::Sum
            | MovingOp::Mean
            | MovingOp::Prod
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => NanMode::Include,
        }
    }

    fn default_fill(self) -> f64 {
        match self {
            MovingOp::Min => f64::INFINITY,
            MovingOp::Max => f64::NEG_INFINITY,
            MovingOp::Sum
            | MovingOp::Mean
            | MovingOp::Prod
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => f64::NAN,
        }
    }

    fn uses_variance_normalization(self) -> bool {
        matches!(self, MovingOp::Std | MovingOp::Var)
    }

    fn provider_op(self) -> Option<ProviderMovingWindowOp> {
        match self {
            MovingOp::Sum => Some(ProviderMovingWindowOp::Sum),
            MovingOp::Mean => Some(ProviderMovingWindowOp::Mean),
            MovingOp::Prod => Some(ProviderMovingWindowOp::Prod),
            MovingOp::Min => Some(ProviderMovingWindowOp::Min),
            MovingOp::Max => Some(ProviderMovingWindowOp::Max),
            MovingOp::Median => Some(ProviderMovingWindowOp::Median),
            MovingOp::Std => Some(ProviderMovingWindowOp::Std),
            MovingOp::Var => Some(ProviderMovingWindowOp::Var),
        }
    }
}

async fn moving_builtin(
    name: &'static str,
    op: MovingOp,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(invalid_argument(
            name,
            format!("{name}: expected at least A and k"),
        ));
    }
    let mut iter = args.into_iter();
    let value = iter.next().expect("checked");
    let k = iter.next().expect("checked");
    let rest: Vec<Value> = iter.collect();
    crate::builtins::common::validation::reject_typed_complex_integer(&value, name)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&k, name)?;
    for argument in &rest {
        crate::builtins::common::validation::reject_typed_complex_integer(argument, name)?;
    }
    let parsed = parse_moving_args(name, op, &k, &rest).await?;
    match value {
        Value::GpuTensor(handle) => moving_gpu(name, op, handle, &parsed).await,
        Value::Complex(re, im) => {
            let tensor = ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|err| internal(name, err))?;
            moving_complex(name, op, tensor, &parsed).map(complex_tensor_into_value)
        }
        Value::ComplexTensor(tensor) => {
            moving_complex(name, op, tensor, &parsed).map(complex_tensor_into_value)
        }
        other => {
            let tensor = tensor::value_into_tensor_for(name, other)
                .map_err(|err| invalid_input(name, err))?;
            moving_real(name, op, tensor, &parsed).map(tensor::tensor_into_value)
        }
    }
}

async fn moving_gpu(
    name: &'static str,
    op: MovingOp,
    handle: GpuTensorHandle,
    parsed: &ParsedMoving,
) -> BuiltinResult<Value> {
    if let Some(output) = moving_gpu_provider(name, op, &handle, parsed).await? {
        return Ok(Value::GpuTensor(output));
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let result = moving_real(name, op, tensor, parsed)?;
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(tensor::tensor_into_value(result));
    };
    let view = HostTensorView {
        data: &result.data,
        shape: &result.shape,
    };
    let uploaded = provider
        .upload(&view)
        .map_err(|err| internal(name, format!("{name}: failed to upload GPU result: {err}")))?;
    Ok(Value::GpuTensor(uploaded))
}

async fn moving_gpu_provider(
    name: &'static str,
    op: MovingOp,
    handle: &GpuTensorHandle,
    parsed: &ParsedMoving,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider_op) = op.provider_op() else {
        return Ok(None);
    };
    if parsed.sample_points.is_some() {
        return Ok(None);
    }
    let Some((before, after)) = parsed.window.count_radius() else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Ok(None);
    }
    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };

    let dim = parsed
        .dim
        .unwrap_or_else(|| default_dimension(&handle.shape));
    if dim == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: dimension must be >= 1"),
        ));
    }
    let input_shape = shape_with_trailing_singletons(&handle.shape, dim);
    let output_shape = output_shape(name, &input_shape, dim, parsed)?;
    validate_sample_points_len(name, input_shape[dim - 1], parsed)?;
    let request = ProviderMovingWindowRequest {
        input: handle,
        output_shape: &output_shape,
        dim: dim - 1,
        before,
        after,
        op: provider_op,
        endpoints: provider_endpoints(op, parsed.endpoints),
        nan_mode: provider_nan_mode(parsed.nan_mode),
        normalization: provider_normalization(parsed.normalization),
    };
    match provider.moving_window(&request).await {
        Ok(output) => {
            runmat_accelerate_api::set_handle_logical(&output, false);
            Ok(Some(output))
        }
        Err(_) => Ok(None),
    }
}

fn provider_endpoints(op: MovingOp, endpoints: Endpoints) -> ProviderMovingWindowEndpoints {
    match endpoints {
        Endpoints::Shrink => ProviderMovingWindowEndpoints::Shrink,
        Endpoints::Discard => ProviderMovingWindowEndpoints::Discard,
        Endpoints::FillDefault => ProviderMovingWindowEndpoints::Fill(op.default_fill()),
        Endpoints::Fill(value) => ProviderMovingWindowEndpoints::Fill(value),
    }
}

fn provider_nan_mode(mode: NanMode) -> ProviderNanMode {
    match mode {
        NanMode::Include => ProviderNanMode::Include,
        NanMode::Omit => ProviderNanMode::Omit,
    }
}

fn provider_normalization(normalization: VarianceNormalization) -> ProviderStdNormalization {
    match normalization {
        VarianceNormalization::Sample => ProviderStdNormalization::Sample,
        VarianceNormalization::Population => ProviderStdNormalization::Population,
    }
}

fn moving_real(
    name: &'static str,
    op: MovingOp,
    tensor: Tensor,
    parsed: &ParsedMoving,
) -> BuiltinResult<Tensor> {
    let dim = parsed
        .dim
        .unwrap_or_else(|| default_dimension(&tensor.shape));
    if dim == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: dimension must be >= 1"),
        ));
    }
    let input_shape = shape_with_trailing_singletons(&tensor.shape, dim);
    let shape = output_shape(name, &input_shape, dim, parsed)?;
    let count = checked_element_count(name, &shape)?;
    let dim_index = dim - 1;
    let axis_len = input_shape[dim_index];
    validate_sample_points_len(name, axis_len, parsed)?;
    let stride_before = checked_product(name, &input_shape[..dim_index])?;
    let stride_after = checked_product(name, &input_shape[dim..])?;
    let out_axis = shape[dim_index];
    let mut output = vec![0.0; count];

    for after in 0..stride_after {
        for out_pos in 0..out_axis {
            let center = source_center(name, out_pos, parsed)?;
            let sample_positions = sample_window_positions(name, center, axis_len, parsed)?;
            let count_window = if sample_positions.is_none() {
                Some(count_window(name, center, axis_len, parsed)?)
            } else {
                None
            };
            let capacity = sample_positions
                .as_ref()
                .map(Vec::len)
                .or_else(|| count_window.map(|window| window.end - window.start))
                .unwrap_or(0);
            let mut values = Vec::with_capacity(capacity);
            for before in 0..stride_before {
                let out_idx = before + out_pos * stride_before + after * stride_before * out_axis;
                values.clear();
                let mut saw_nan = false;
                if let Some(positions) = sample_positions.as_ref() {
                    for &pos in positions {
                        let idx = before + pos * stride_before + after * stride_before * axis_len;
                        let value = tensor.data[idx];
                        if value.is_nan() {
                            if parsed.nan_mode == NanMode::Include {
                                saw_nan = true;
                                break;
                            }
                            continue;
                        }
                        values.push(value);
                    }
                    output[out_idx] =
                        reduce_real_window(op, &values, saw_nan, parsed.normalization);
                    continue;
                }

                let window = count_window.expect("count window computed");
                for pos in window.start..window.end {
                    let idx = before + pos * stride_before + after * stride_before * axis_len;
                    let value = tensor.data[idx];
                    if value.is_nan() {
                        if parsed.nan_mode == NanMode::Include {
                            saw_nan = true;
                            break;
                        }
                        continue;
                    }
                    values.push(value);
                }
                output[out_idx] = reduce_real_window_with_fill(
                    op,
                    &values,
                    saw_nan,
                    fill_value(op, parsed, window.fill_count),
                    parsed.nan_mode,
                    parsed.normalization,
                );
            }
        }
    }

    Tensor::new_with_dtype(output, shape, tensor.dtype).map_err(|err| internal(name, err))
}

fn moving_complex(
    name: &'static str,
    op: MovingOp,
    tensor: ComplexTensor,
    parsed: &ParsedMoving,
) -> BuiltinResult<ComplexTensor> {
    let dim = parsed
        .dim
        .unwrap_or_else(|| default_dimension(&tensor.shape));
    if dim == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: dimension must be >= 1"),
        ));
    }
    let input_shape = shape_with_trailing_singletons(&tensor.shape, dim);
    let shape = output_shape(name, &input_shape, dim, parsed)?;
    let count = checked_element_count(name, &shape)?;
    let dim_index = dim - 1;
    let axis_len = input_shape[dim_index];
    validate_sample_points_len(name, axis_len, parsed)?;
    let stride_before = checked_product(name, &input_shape[..dim_index])?;
    let stride_after = checked_product(name, &input_shape[dim..])?;
    let out_axis = shape[dim_index];
    let mut output = vec![(0.0, 0.0); count];

    for after in 0..stride_after {
        for out_pos in 0..out_axis {
            let center = source_center(name, out_pos, parsed)?;
            let sample_positions = sample_window_positions(name, center, axis_len, parsed)?;
            let count_window = if sample_positions.is_none() {
                Some(count_window(name, center, axis_len, parsed)?)
            } else {
                None
            };
            let capacity = sample_positions
                .as_ref()
                .map(Vec::len)
                .or_else(|| count_window.map(|window| window.end - window.start))
                .unwrap_or(0);
            let mut values = Vec::with_capacity(capacity);
            for before in 0..stride_before {
                let out_idx = before + out_pos * stride_before + after * stride_before * out_axis;
                values.clear();
                let mut saw_nan = false;
                if let Some(positions) = sample_positions.as_ref() {
                    for &pos in positions {
                        let idx = before + pos * stride_before + after * stride_before * axis_len;
                        let value = tensor.data[idx];
                        if value.0.is_nan() || value.1.is_nan() {
                            if parsed.nan_mode == NanMode::Include {
                                saw_nan = true;
                                break;
                            }
                            continue;
                        }
                        values.push(value);
                    }
                    output[out_idx] =
                        reduce_complex_window(op, &values, saw_nan, parsed.normalization);
                    continue;
                }

                let window = count_window.expect("count window computed");
                for pos in window.start..window.end {
                    let idx = before + pos * stride_before + after * stride_before * axis_len;
                    let value = tensor.data[idx];
                    if value.0.is_nan() || value.1.is_nan() {
                        if parsed.nan_mode == NanMode::Include {
                            saw_nan = true;
                            break;
                        }
                        continue;
                    }
                    values.push(value);
                }
                output[out_idx] = reduce_complex_window_with_fill(
                    op,
                    &values,
                    saw_nan,
                    fill_value(op, parsed, window.fill_count)
                        .map(|(fill, count)| ((fill, 0.0), count)),
                    parsed.nan_mode,
                    parsed.normalization,
                );
            }
        }
    }

    ComplexTensor::new(output, shape).map_err(|err| internal(name, err))
}

fn complex_tensor_into_value(tensor: ComplexTensor) -> Value {
    if tensor::is_scalar_complex_tensor(&tensor) && tensor.integer_data.is_none() {
        let value = tensor::complex_tensor_value_complex64(&tensor, 0);
        Value::Complex(value.re, value.im)
    } else {
        Value::ComplexTensor(tensor)
    }
}

fn reduce_real_window(
    op: MovingOp,
    values: &[f64],
    saw_nan: bool,
    normalization: VarianceNormalization,
) -> f64 {
    if saw_nan {
        return f64::NAN;
    }
    if values.is_empty() {
        return match op {
            MovingOp::Sum => 0.0,
            MovingOp::Prod => 1.0,
            MovingOp::Mean
            | MovingOp::Min
            | MovingOp::Max
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => f64::NAN,
        };
    }
    match op {
        MovingOp::Sum => values.iter().sum(),
        MovingOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
        MovingOp::Prod => values.iter().product(),
        MovingOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        MovingOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        MovingOp::Median => median(values),
        MovingOp::Var => variance_real(values, normalization),
        MovingOp::Std => variance_real(values, normalization).sqrt(),
    }
}

fn reduce_real_window_with_fill(
    op: MovingOp,
    values: &[f64],
    saw_nan: bool,
    fill: Option<(f64, usize)>,
    nan_mode: NanMode,
    normalization: VarianceNormalization,
) -> f64 {
    let Some((fill, fill_count)) = fill else {
        return reduce_real_window(op, values, saw_nan, normalization);
    };
    if fill_count == 0 {
        return reduce_real_window(op, values, saw_nan, normalization);
    }
    if saw_nan || (fill.is_nan() && nan_mode == NanMode::Include) {
        return f64::NAN;
    }
    if fill.is_nan() && nan_mode == NanMode::Omit {
        return reduce_real_window(op, values, false, normalization);
    }
    match op {
        MovingOp::Sum => values.iter().sum::<f64>() + fill * fill_count as f64,
        MovingOp::Mean => {
            (values.iter().sum::<f64>() + fill * fill_count as f64)
                / (values.len() + fill_count) as f64
        }
        MovingOp::Prod => values.iter().product::<f64>() * fill.powf(fill_count as f64),
        MovingOp::Min => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .fold(f64::INFINITY, f64::min),
        MovingOp::Max => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .fold(f64::NEG_INFINITY, f64::max),
        MovingOp::Median => median_with_fill(values, fill, fill_count),
        MovingOp::Var => variance_real_with_fill(values, fill, fill_count, normalization),
        MovingOp::Std => variance_real_with_fill(values, fill, fill_count, normalization).sqrt(),
    }
}

fn reduce_complex_window(
    op: MovingOp,
    values: &[(f64, f64)],
    saw_nan: bool,
    normalization: VarianceNormalization,
) -> (f64, f64) {
    if saw_nan {
        return (f64::NAN, f64::NAN);
    }
    if values.is_empty() {
        return match op {
            MovingOp::Sum => (0.0, 0.0),
            MovingOp::Prod => (1.0, 0.0),
            MovingOp::Mean
            | MovingOp::Min
            | MovingOp::Max
            | MovingOp::Median
            | MovingOp::Std
            | MovingOp::Var => (f64::NAN, f64::NAN),
        };
    }
    match op {
        MovingOp::Sum => values
            .iter()
            .fold((0.0, 0.0), |acc, value| (acc.0 + value.0, acc.1 + value.1)),
        MovingOp::Mean => {
            let sum = reduce_complex_window(MovingOp::Sum, values, false, normalization);
            (sum.0 / values.len() as f64, sum.1 / values.len() as f64)
        }
        MovingOp::Prod => values.iter().fold((1.0, 0.0), |acc, value| {
            (
                acc.0 * value.0 - acc.1 * value.1,
                acc.0 * value.1 + acc.1 * value.0,
            )
        }),
        MovingOp::Min => values
            .iter()
            .copied()
            .min_by(compare_complex_auto)
            .unwrap_or((f64::NAN, f64::NAN)),
        MovingOp::Max => values
            .iter()
            .copied()
            .max_by(compare_complex_auto)
            .unwrap_or((f64::NAN, f64::NAN)),
        MovingOp::Median => complex_median(values),
        MovingOp::Var => (variance_complex(values, normalization), 0.0),
        MovingOp::Std => (variance_complex(values, normalization).sqrt(), 0.0),
    }
}

fn reduce_complex_window_with_fill(
    op: MovingOp,
    values: &[(f64, f64)],
    saw_nan: bool,
    fill: Option<((f64, f64), usize)>,
    nan_mode: NanMode,
    normalization: VarianceNormalization,
) -> (f64, f64) {
    let Some((fill, fill_count)) = fill else {
        return reduce_complex_window(op, values, saw_nan, normalization);
    };
    if fill_count == 0 {
        return reduce_complex_window(op, values, saw_nan, normalization);
    }
    let fill_is_nan = fill.0.is_nan() || fill.1.is_nan();
    if saw_nan || (fill_is_nan && nan_mode == NanMode::Include) {
        return (f64::NAN, f64::NAN);
    }
    if fill_is_nan && nan_mode == NanMode::Omit {
        return reduce_complex_window(op, values, false, normalization);
    }
    match op {
        MovingOp::Sum => {
            let sum = reduce_complex_window(MovingOp::Sum, values, false, normalization);
            (
                sum.0 + fill.0 * fill_count as f64,
                sum.1 + fill.1 * fill_count as f64,
            )
        }
        MovingOp::Mean => {
            let sum = reduce_complex_window_with_fill(
                MovingOp::Sum,
                values,
                false,
                Some((fill, fill_count)),
                nan_mode,
                normalization,
            );
            (
                sum.0 / (values.len() + fill_count) as f64,
                sum.1 / (values.len() + fill_count) as f64,
            )
        }
        MovingOp::Prod => {
            let product = reduce_complex_window(MovingOp::Prod, values, false, normalization);
            let fill_product = complex_pow_usize(fill, fill_count);
            (
                product.0 * fill_product.0 - product.1 * fill_product.1,
                product.0 * fill_product.1 + product.1 * fill_product.0,
            )
        }
        MovingOp::Min => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .min_by(compare_complex_auto)
            .unwrap_or((f64::NAN, f64::NAN)),
        MovingOp::Max => values
            .iter()
            .copied()
            .chain(std::iter::once(fill))
            .max_by(compare_complex_auto)
            .unwrap_or((f64::NAN, f64::NAN)),
        MovingOp::Median => complex_median_with_fill(values, fill, fill_count),
        MovingOp::Var => (
            variance_complex_with_fill(values, fill, fill_count, normalization),
            0.0,
        ),
        MovingOp::Std => (
            variance_complex_with_fill(values, fill, fill_count, normalization).sqrt(),
            0.0,
        ),
    }
}

fn variance_denominator(n: usize, normalization: VarianceNormalization) -> f64 {
    match normalization {
        VarianceNormalization::Population => n as f64,
        VarianceNormalization::Sample if n > 1 => (n - 1) as f64,
        VarianceNormalization::Sample => n as f64,
    }
}

fn variance_real(values: &[f64], normalization: VarianceNormalization) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut acc = RealVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.finish(normalization)
}

fn variance_real_with_fill(
    values: &[f64],
    fill: f64,
    fill_count: usize,
    normalization: VarianceNormalization,
) -> f64 {
    let n = values.len() + fill_count;
    if n == 0 {
        return f64::NAN;
    }
    let mut acc = RealVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.merge_repeated(fill, fill_count);
    acc.finish(normalization)
}

fn variance_complex(values: &[(f64, f64)], normalization: VarianceNormalization) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut acc = ComplexVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.finish(normalization)
}

fn variance_complex_with_fill(
    values: &[(f64, f64)],
    fill: (f64, f64),
    fill_count: usize,
    normalization: VarianceNormalization,
) -> f64 {
    let n = values.len() + fill_count;
    if n == 0 {
        return f64::NAN;
    }
    let mut acc = ComplexVarianceAccumulator::default();
    for value in values {
        acc.push(*value);
    }
    acc.merge_repeated(fill, fill_count);
    acc.finish(normalization)
}

#[derive(Default)]
struct RealVarianceAccumulator {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RealVarianceAccumulator {
    fn push(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    fn merge_repeated(&mut self, value: f64, count: usize) {
        if count == 0 {
            return;
        }
        if self.n == 0 {
            self.n = count;
            self.mean = value;
            self.m2 = 0.0;
            return;
        }
        let total = self.n + count;
        let delta = value - self.mean;
        self.mean += delta * (count as f64 / total as f64);
        self.m2 += delta * delta * (self.n as f64 * count as f64 / total as f64);
        self.n = total;
    }

    fn finish(self, normalization: VarianceNormalization) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        self.m2 / variance_denominator(self.n, normalization)
    }
}

#[derive(Default)]
struct ComplexVarianceAccumulator {
    n: usize,
    mean: (f64, f64),
    m2: f64,
}

impl ComplexVarianceAccumulator {
    fn push(&mut self, value: (f64, f64)) {
        self.n += 1;
        let delta = (value.0 - self.mean.0, value.1 - self.mean.1);
        self.mean.0 += delta.0 / self.n as f64;
        self.mean.1 += delta.1 / self.n as f64;
        let delta2 = (value.0 - self.mean.0, value.1 - self.mean.1);
        self.m2 += delta.0 * delta2.0 + delta.1 * delta2.1;
    }

    fn merge_repeated(&mut self, value: (f64, f64), count: usize) {
        if count == 0 {
            return;
        }
        if self.n == 0 {
            self.n = count;
            self.mean = value;
            self.m2 = 0.0;
            return;
        }
        let total = self.n + count;
        let delta = (value.0 - self.mean.0, value.1 - self.mean.1);
        let scale = self.n as f64 * count as f64 / total as f64;
        self.mean.0 += delta.0 * (count as f64 / total as f64);
        self.mean.1 += delta.1 * (count as f64 / total as f64);
        self.m2 += (delta.0 * delta.0 + delta.1 * delta.1) * scale;
        self.n = total;
    }

    fn finish(self, normalization: VarianceNormalization) -> f64 {
        if self.n == 0 {
            return f64::NAN;
        }
        self.m2 / variance_denominator(self.n, normalization)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

fn median_with_fill(values: &[f64], fill: f64, fill_count: usize) -> f64 {
    if fill_count == 0 {
        return median(values);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let total = sorted.len() + fill_count;
    if total == 0 {
        return f64::NAN;
    }
    let mid = total / 2;
    if total % 2 == 1 {
        kth_with_repeated_fill(&sorted, fill, fill_count, mid)
    } else {
        (kth_with_repeated_fill(&sorted, fill, fill_count, mid - 1)
            + kth_with_repeated_fill(&sorted, fill, fill_count, mid))
            / 2.0
    }
}

fn kth_with_repeated_fill(sorted: &[f64], fill: f64, fill_count: usize, k: usize) -> f64 {
    let less = sorted.partition_point(|value| *value < fill);
    let equal = sorted[less..].partition_point(|value| *value == fill);
    if k < less {
        sorted[k]
    } else if k < less + equal + fill_count {
        fill
    } else {
        sorted[k - fill_count]
    }
}

fn complex_median(values: &[(f64, f64)]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| compare_complex_auto(a, b));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (
            (sorted[mid - 1].0 + sorted[mid].0) / 2.0,
            (sorted[mid - 1].1 + sorted[mid].1) / 2.0,
        )
    }
}

fn complex_median_with_fill(
    values: &[(f64, f64)],
    fill: (f64, f64),
    fill_count: usize,
) -> (f64, f64) {
    if fill_count == 0 {
        return complex_median(values);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| compare_complex_auto(a, b));
    let total = sorted.len() + fill_count;
    if total == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mid = total / 2;
    if total % 2 == 1 {
        kth_complex_with_repeated_fill(&sorted, fill, fill_count, mid)
    } else {
        let lo = kth_complex_with_repeated_fill(&sorted, fill, fill_count, mid - 1);
        let hi = kth_complex_with_repeated_fill(&sorted, fill, fill_count, mid);
        ((lo.0 + hi.0) / 2.0, (lo.1 + hi.1) / 2.0)
    }
}

fn kth_complex_with_repeated_fill(
    sorted: &[(f64, f64)],
    fill: (f64, f64),
    fill_count: usize,
    k: usize,
) -> (f64, f64) {
    let less = sorted.partition_point(|value| compare_complex_auto(value, &fill) == Ordering::Less);
    let equal = sorted[less..]
        .partition_point(|value| compare_complex_auto(value, &fill) == Ordering::Equal);
    if k < less {
        sorted[k]
    } else if k < less + equal + fill_count {
        fill
    } else {
        sorted[k - fill_count]
    }
}

fn complex_pow_usize(mut base: (f64, f64), mut exponent: usize) -> (f64, f64) {
    let mut result = (1.0, 0.0);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = complex_mul(result, base);
        }
        exponent >>= 1;
        if exponent > 0 {
            base = complex_mul(base, base);
        }
    }
    result
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn compare_complex_auto(a: &(f64, f64), b: &(f64, f64)) -> Ordering {
    let a_abs = a.0.hypot(a.1);
    let b_abs = b.0.hypot(b.1);
    a_abs
        .partial_cmp(&b_abs)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            a.1.atan2(a.0)
                .partial_cmp(&b.1.atan2(b.0))
                .unwrap_or(Ordering::Equal)
        })
}

fn source_center(
    name: &'static str,
    out_pos: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<usize> {
    match parsed.endpoints {
        Endpoints::Discard => {
            let Some((before, _)) = parsed.window.count_radius() else {
                return Ok(out_pos);
            };
            out_pos
                .checked_add(before)
                .ok_or_else(|| invalid_argument(name, format!("{name}: window index overflow")))
        }
        Endpoints::Shrink | Endpoints::FillDefault | Endpoints::Fill(_) => Ok(out_pos),
    }
}

fn fill_value(op: MovingOp, parsed: &ParsedMoving, fill_count: usize) -> Option<(f64, usize)> {
    if fill_count == 0 {
        return None;
    }
    match parsed.endpoints {
        Endpoints::FillDefault => Some((op.default_fill(), fill_count)),
        Endpoints::Fill(value) => Some((value, fill_count)),
        Endpoints::Shrink | Endpoints::Discard => None,
    }
}

fn count_window(
    name: &'static str,
    center: usize,
    axis_len: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<CountWindow> {
    let Some((before, after)) = parsed.window.count_radius() else {
        return Err(invalid_argument(
            name,
            format!("{name}: sample points are required for duration windows"),
        ));
    };
    let start = center as i128 - before as i128;
    let end = center as i128 + after as i128;
    let in_start = start.max(0).min(axis_len as i128) as usize;
    let in_end = end
        .checked_add(1)
        .ok_or_else(|| invalid_argument(name, format!("{name}: window index overflow")))?
        .max(0)
        .min(axis_len as i128) as usize;
    let fill_count = match parsed.endpoints {
        Endpoints::FillDefault | Endpoints::Fill(_) => {
            let before_missing = if start < 0 { (-start) as usize } else { 0 };
            let after_missing = if end >= axis_len as i128 {
                (end - axis_len as i128 + 1) as usize
            } else {
                0
            };
            before_missing
                .checked_add(after_missing)
                .ok_or_else(|| invalid_argument(name, format!("{name}: window length overflow")))?
        }
        Endpoints::Shrink | Endpoints::Discard => 0,
    };
    Ok(CountWindow {
        start: in_start,
        end: in_end,
        fill_count,
    })
}

fn sample_window_positions(
    name: &'static str,
    center: usize,
    axis_len: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<Option<Vec<usize>>> {
    let Some(sample_points) = parsed.sample_points.as_ref() else {
        return Ok(None);
    };
    if center >= axis_len || center >= sample_points.len() {
        return Err(invalid_argument(
            name,
            format!("{name}: sample point center is outside the selected dimension"),
        ));
    }
    let WindowSpec::Span { before, after } = parsed.window else {
        return Err(invalid_argument(
            name,
            format!("{name}: SamplePoints requires a distance window"),
        ));
    };
    let center_point = sample_points[center];
    let lower = center_point - before;
    let upper = center_point + after;
    Ok(Some(
        sample_points
            .iter()
            .enumerate()
            .filter_map(|(idx, point)| {
                if *point >= lower && *point <= upper {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect(),
    ))
}

fn shape_with_trailing_singletons(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    if dim > out.len() {
        out.resize(dim, 1);
    }
    out
}

fn validate_sample_points_len(
    name: &'static str,
    axis_len: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<()> {
    if let Some(points) = parsed.sample_points.as_ref() {
        if points.len() != axis_len {
            return Err(invalid_argument(
                name,
                format!("{name}: SamplePoints length must match the selected dimension length"),
            ));
        }
    }
    Ok(())
}

fn output_shape(
    name: &'static str,
    input_shape: &[usize],
    dim: usize,
    parsed: &ParsedMoving,
) -> BuiltinResult<Vec<usize>> {
    let mut shape = input_shape.to_vec();
    if dim == 0 {
        return Ok(shape);
    }
    if matches!(parsed.endpoints, Endpoints::Discard) {
        let Some((before, after)) = parsed.window.count_radius() else {
            return Err(invalid_argument(
                name,
                format!("{name}: Endpoints 'discard' is not supported with SamplePoints"),
            ));
        };
        let axis = shape[dim - 1];
        let trim = before
            .checked_add(after)
            .ok_or_else(|| invalid_argument(name, format!("{name}: window length overflow")))?;
        shape[dim - 1] = axis.saturating_sub(trim);
    }
    checked_element_count(name, &shape)?;
    Ok(shape)
}

async fn parse_moving_args(
    name: &'static str,
    op: MovingOp,
    k: &Value,
    rest: &[Value],
) -> BuiltinResult<ParsedMoving> {
    let mut dim = None;
    let mut nan_mode = op.default_nan_mode();
    let mut endpoints = Endpoints::Shrink;
    let mut sample_points: Option<Vec<f64>> = None;
    let mut normalization = VarianceNormalization::Sample;
    let mut normalization_seen = false;
    let mut idx = 0;
    while idx < rest.len() {
        if let Some(keyword) = value_as_keyword(&rest[idx]) {
            match keyword.as_str() {
                "omitnan" | "omitmissing" => {
                    nan_mode = NanMode::Omit;
                    idx += 1;
                    continue;
                }
                "includenan" | "includemissing" => {
                    nan_mode = NanMode::Include;
                    idx += 1;
                    continue;
                }
                "endpoints" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(invalid_argument(
                            name,
                            format!("{name}: Endpoints requires a value"),
                        ));
                    };
                    endpoints = parse_endpoints(name, value)?;
                    idx += 2;
                    continue;
                }
                "samplepoints" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(invalid_argument(
                            name,
                            format!("{name}: SamplePoints requires a value"),
                        ));
                    };
                    if sample_points.is_some() {
                        return Err(invalid_argument(
                            name,
                            format!("{name}: multiple SamplePoints options provided"),
                        ));
                    }
                    sample_points = Some(parse_sample_points(name, value).await?);
                    idx += 2;
                    continue;
                }
                _ => {
                    if rest.get(idx + 1).is_some() && !looks_like_dimension(&rest[idx]).await? {
                        return Err(invalid_argument(
                            name,
                            format!(
                                "{name}: unrecognised option '{}'",
                                value_as_string(&rest[idx]).unwrap_or(keyword)
                            ),
                        ));
                    }
                }
            }
        }

        if op.uses_variance_normalization() && !normalization_seen {
            if let Some(parsed_normalization) =
                parse_variance_normalization(name, &rest[idx]).await?
            {
                normalization = parsed_normalization;
                normalization_seen = true;
                idx += 1;
                continue;
            }
        }

        if dim.is_none() {
            if let Some(parsed_dim) = tensor::dimension_from_value_async(&rest[idx], name, false)
                .await
                .map_err(|err| invalid_argument(name, err))?
            {
                dim = Some(parsed_dim);
                idx += 1;
                continue;
            }
        }

        return Err(invalid_argument(
            name,
            format!("{name}: unrecognised argument {:?}", rest[idx]),
        ));
    }
    if sample_points.is_some() && !matches!(endpoints, Endpoints::Shrink) {
        return Err(invalid_argument(
            name,
            format!("{name}: SamplePoints currently requires Endpoints 'shrink'"),
        ));
    }
    let window = if sample_points.is_some() {
        parse_sample_window(name, k).await?
    } else {
        parse_window(name, k).await?
    };
    Ok(ParsedMoving {
        window,
        dim,
        nan_mode,
        endpoints,
        sample_points,
        normalization,
    })
}

async fn parse_variance_normalization(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Option<VarianceNormalization>> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return parse_integer_variance_normalization(name, &integer).map(Some);
    }
    let values = match value {
        Value::Tensor(tensor) if tensor_len(tensor) == 0 => {
            return Ok(Some(VarianceNormalization::Sample));
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            vec![tensor::tensor_value_f64(tensor, 0)]
        }
        Value::Tensor(_) => {
            return Err(invalid_argument(
                name,
                format!("{name}: normalization flag must be 0 or 1"),
            ));
        }
        Value::LogicalArray(logical) if logical.data.is_empty() => {
            return Ok(Some(VarianceNormalization::Sample));
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            vec![if logical.data[0] != 0 { 1.0 } else { 0.0 }]
        }
        Value::LogicalArray(_) => {
            return Err(invalid_argument(
                name,
                format!("{name}: normalization flag must be 0 or 1"),
            ));
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if tensor_len(&tensor) == 0 {
                return Ok(Some(VarianceNormalization::Sample));
            }
            if tensor::is_scalar_tensor(&tensor) {
                if let Some(integer) = tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0))
                {
                    return parse_integer_variance_normalization(name, &integer).map(Some);
                }
                vec![tensor::tensor_value_f64(&tensor, 0)]
            } else {
                return Err(invalid_argument(
                    name,
                    format!("{name}: normalization flag must be 0 or 1"),
                ));
            }
        }
        Value::Num(n) => vec![*n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Bool(b) => vec![if *b { 1.0 } else { 0.0 }],
        _ => {
            return Ok(None);
        }
    };
    let raw = values[0];
    if (raw - 0.0).abs() <= f64::EPSILON {
        Ok(Some(VarianceNormalization::Sample))
    } else if (raw - 1.0).abs() <= f64::EPSILON {
        Ok(Some(VarianceNormalization::Population))
    } else {
        Err(invalid_argument(
            name,
            format!("{name}: normalization flag must be 0 or 1"),
        ))
    }
}

fn parse_integer_variance_normalization(
    name: &'static str,
    value: &IntValue,
) -> BuiltinResult<VarianceNormalization> {
    match value.try_to_i64() {
        Some(0) => Ok(VarianceNormalization::Sample),
        Some(1) => Ok(VarianceNormalization::Population),
        _ => Err(invalid_argument(
            name,
            format!("{name}: normalization flag must be 0 or 1"),
        )),
    }
}

async fn looks_like_dimension(value: &Value) -> BuiltinResult<bool> {
    tensor::dimension_from_value_async(value, "moving", false)
        .await
        .map(|dim| dim.is_some())
        .map_err(|err| invalid_argument("moving", err))
}

async fn parse_window(name: &'static str, value: &Value) -> BuiltinResult<WindowSpec> {
    if let Some(values) = integer_vector(value).await? {
        return match values.as_slice() {
            [raw] => {
                let k = positive_integer_value(name, raw, "window length")?;
                let before = k / 2;
                let after = (k - 1) / 2;
                WindowSpec::Counts { before, after }
                    .checked_count_len(name)?
                    .map(|_| WindowSpec::Counts { before, after })
                    .ok_or_else(|| internal(name, format!("{name}: invalid window")))
            }
            [before, after] => {
                let before = nonnegative_integer_value(name, before, "window backward length")?;
                let after = nonnegative_integer_value(name, after, "window forward length")?;
                WindowSpec::Counts { before, after }
                    .checked_count_len(name)?
                    .map(|_| WindowSpec::Counts { before, after })
                    .ok_or_else(|| internal(name, format!("{name}: invalid window")))
            }
            _ => Err(invalid_argument(
                name,
                format!("{name}: window must be a scalar or a two-element vector"),
            )),
        };
    }
    let values = numeric_vector(name, value).await?;
    match values.as_slice() {
        [raw] => {
            let k = positive_integer(name, *raw, "window length")?;
            let before = k / 2;
            let after = (k - 1) / 2;
            WindowSpec::Counts { before, after }
                .checked_count_len(name)?
                .map(|_| WindowSpec::Counts { before, after })
                .ok_or_else(|| internal(name, format!("{name}: invalid window")))
        }
        [before, after] => {
            let before = nonnegative_integer(name, *before, "window backward length")?;
            let after = nonnegative_integer(name, *after, "window forward length")?;
            WindowSpec::Counts { before, after }
                .checked_count_len(name)?
                .map(|_| WindowSpec::Counts { before, after })
                .ok_or_else(|| internal(name, format!("{name}: invalid window")))
        }
        _ => Err(invalid_argument(
            name,
            format!("{name}: window must be a scalar or a two-element vector"),
        )),
    }
}

async fn integer_vector(value: &Value) -> BuiltinResult<Option<Vec<IntValue>>> {
    let tensor = match value {
        Value::Int(value) => return Ok(Some(vec![value.clone()])),
        Value::Tensor(tensor) => tensor,
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            return Ok(tensor.integer_storage().map(|storage| {
                (0..storage.len())
                    .filter_map(|index| storage.value_at(index))
                    .collect()
            }));
        }
        _ => return Ok(None),
    };
    Ok(tensor.integer_storage().map(|storage| {
        (0..storage.len())
            .filter_map(|index| storage.value_at(index))
            .collect()
    }))
}

async fn parse_sample_window(name: &'static str, value: &Value) -> BuiltinResult<WindowSpec> {
    let values = numeric_vector(name, value).await?;
    match values.as_slice() {
        [raw] => {
            let width = positive_span(name, *raw, "sample-point window length")?;
            Ok(WindowSpec::Span {
                before: width / 2.0,
                after: width / 2.0,
            })
        }
        [before, after] => Ok(WindowSpec::Span {
            before: nonnegative_span(name, *before, "sample-point backward window")?,
            after: nonnegative_span(name, *after, "sample-point forward window")?,
        }),
        _ => Err(invalid_argument(
            name,
            format!("{name}: sample-point window must be a scalar or two-element vector"),
        )),
    }
}

async fn parse_sample_points(name: &'static str, value: &Value) -> BuiltinResult<Vec<f64>> {
    let points = numeric_vector(name, value).await?;
    if points.is_empty() {
        return Err(invalid_argument(
            name,
            format!("{name}: SamplePoints must not be empty"),
        ));
    }
    let mut previous = None;
    for point in &points {
        if !point.is_finite() {
            return Err(invalid_argument(
                name,
                format!("{name}: SamplePoints must be finite"),
            ));
        }
        if let Some(prev) = previous {
            if *point <= prev {
                return Err(invalid_argument(
                    name,
                    format!("{name}: SamplePoints must be strictly increasing"),
                ));
            }
        }
        previous = Some(*point);
    }
    Ok(points)
}

fn parse_endpoints(name: &'static str, value: &Value) -> BuiltinResult<Endpoints> {
    if let Some(text) = value_as_keyword(value) {
        return match text.as_str() {
            "shrink" => Ok(Endpoints::Shrink),
            "discard" => Ok(Endpoints::Discard),
            "fill" => Ok(Endpoints::FillDefault),
            _ => Err(invalid_argument(
                name,
                format!("{name}: unsupported Endpoints value '{text}'"),
            )),
        };
    }
    let scalar = scalar_f64(value).ok_or_else(|| {
        invalid_argument(
            name,
            format!("{name}: Endpoints must be a keyword or scalar fill value"),
        )
    })?;
    Ok(Endpoints::Fill(scalar))
}

async fn numeric_vector(name: &'static str, value: &Value) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(b) => Ok(vec![if *b { 1.0 } else { 0.0 }]),
        Value::Tensor(t) => Ok(tensor::tensor_values_f64(t)),
        Value::LogicalArray(l) => Ok(l
            .data
            .iter()
            .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
            .collect()),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            Ok(tensor::tensor_values_f64(&tensor))
        }
        _ => Err(invalid_argument(
            name,
            format!("{name}: window must be numeric"),
        )),
    }
}

fn scalar_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Some(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor
        .integer_storage()
        .map_or(tensor.data.len(), |storage| storage.len())
}

fn positive_integer(name: &'static str, raw: f64, what: &str) -> BuiltinResult<usize> {
    let value = nonnegative_integer(name, raw, what)?;
    if value == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be positive"),
        ));
    }
    Ok(value)
}

fn positive_integer_value(
    name: &'static str,
    value: &IntValue,
    what: &str,
) -> BuiltinResult<usize> {
    let value = nonnegative_integer_value(name, value, what)?;
    if value == 0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be positive"),
        ));
    }
    Ok(value)
}

fn nonnegative_integer_value(
    name: &'static str,
    value: &IntValue,
    what: &str,
) -> BuiltinResult<usize> {
    value.try_to_usize().ok_or_else(|| {
        invalid_argument(
            name,
            format!("{name}: {what} must be a nonnegative integer"),
        )
    })
}

fn nonnegative_integer(name: &'static str, raw: f64, what: &str) -> BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be finite"),
        ));
    }
    let rounded = raw.round();
    if (rounded - raw).abs() > 1e-9 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be an integer"),
        ));
    }
    if rounded < 0.0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be nonnegative"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} is too large"),
        ));
    }
    Ok(rounded as usize)
}

fn positive_span(name: &'static str, raw: f64, what: &str) -> BuiltinResult<f64> {
    let value = nonnegative_span(name, raw, what)?;
    if value <= 0.0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be positive"),
        ));
    }
    Ok(value)
}

fn nonnegative_span(name: &'static str, raw: f64, what: &str) -> BuiltinResult<f64> {
    if !raw.is_finite() {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be finite"),
        ));
    }
    if raw < 0.0 {
        return Err(invalid_argument(
            name,
            format!("{name}: {what} must be nonnegative"),
        ));
    }
    Ok(raw)
}

fn value_as_keyword(value: &Value) -> Option<String> {
    keyword_of(value).map(|s| s.to_ascii_lowercase())
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn default_dimension(shape: &[usize]) -> usize {
    shape
        .iter()
        .position(|&dim| dim != 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

fn checked_product(name: &'static str, values: &[usize]) -> BuiltinResult<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| internal(name, format!("{name}: shape product overflow")))
    })
}

fn checked_element_count(name: &'static str, shape: &[usize]) -> BuiltinResult<usize> {
    checked_product(name, shape)
}

fn invalid_argument(name: &'static str, detail: impl Into<String>) -> RuntimeError {
    descriptor_error(name, &ERROR_INVALID_ARGUMENT, detail)
}

fn invalid_input(name: &'static str, detail: impl Into<String>) -> RuntimeError {
    descriptor_error(name, &ERROR_INVALID_INPUT, detail)
}

fn internal(name: &'static str, detail: impl Into<String>) -> RuntimeError {
    descriptor_error(name, &ERROR_INTERNAL, detail)
}

fn descriptor_error(
    name: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> RuntimeError {
    let builder = build_runtime_error(format!("{}: {}", descriptor.message, detail.into()))
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:{}", identifier_suffix(descriptor)));
    builder.build()
}

fn identifier_suffix(descriptor: &'static BuiltinErrorDescriptor) -> &'static str {
    match descriptor.code {
        "RM.MOVING.INVALID_ARGUMENT" => "InvalidArgument",
        "RM.MOVING.INVALID_INPUT" => "InvalidInput",
        _ => "Internal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage};

    fn call(name: &'static str, op: MovingOp, args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(moving_builtin(name, op, args))
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    #[test]
    fn moving_operations_reject_typed_complex_integer_inputs() {
        let input = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX]),
                    IntegerStorage::U64(vec![1]),
                )
                .expect("storage"),
                vec![1, 1],
            )
            .expect("tensor"),
        );
        let err = call("movmean", MovingOp::Mean, vec![input, Value::Num(1.0)])
            .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gpu_count_window_reductions_return_resident_results() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, f64::NAN, 4.0],
                    shape: &[1, 4],
                })
                .expect("upload");
            let result = call(
                "movsum",
                MovingOp::Sum,
                vec![
                    Value::GpuTensor(input.clone()),
                    Value::Num(3.0),
                    Value::from("omitnan"),
                ],
            )
            .expect("movsum gpu");
            let Value::GpuTensor(sum_handle) = result else {
                panic!("expected resident gpu result");
            };
            assert!(runmat_accelerate_api::provider_for_handle(&sum_handle).is_some());
            let sum =
                crate::builtins::common::test_support::gather(Value::GpuTensor(sum_handle.clone()))
                    .expect("gather sum");
            assert_eq!(sum.shape, vec![1, 4]);
            assert_eq!(sum.data, vec![3.0, 3.0, 6.0, 4.0]);

            let var_input = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .expect("upload var");
            let result = call(
                "movvar",
                MovingOp::Var,
                vec![
                    Value::GpuTensor(var_input.clone()),
                    Value::Num(3.0),
                    Value::from("Endpoints"),
                    Value::Num(0.0),
                ],
            )
            .expect("movvar gpu");
            let Value::GpuTensor(var_handle) = result else {
                panic!("expected resident gpu var result");
            };
            assert!(runmat_accelerate_api::provider_for_handle(&var_handle).is_some());
            let var =
                crate::builtins::common::test_support::gather(Value::GpuTensor(var_handle.clone()))
                    .expect("gather var");
            assert_eq!(var.shape, vec![1, 2]);
            assert_eq!(var.data, vec![1.0, 1.0]);

            let median_input = provider
                .upload(&HostTensorView {
                    data: &[1.0, f64::NAN, 3.0, 9.0],
                    shape: &[1, 4],
                })
                .expect("upload median");
            let result = call(
                "movmedian",
                MovingOp::Median,
                vec![
                    Value::GpuTensor(median_input.clone()),
                    Value::Num(3.0),
                    Value::from("omitnan"),
                ],
            )
            .expect("movmedian gpu");
            let Value::GpuTensor(median_handle) = result else {
                panic!("expected resident gpu median result");
            };
            assert!(runmat_accelerate_api::provider_for_handle(&median_handle).is_some());
            let median = crate::builtins::common::test_support::gather(Value::GpuTensor(
                median_handle.clone(),
            ))
            .expect("gather median");
            assert_eq!(median.shape, vec![1, 4]);
            assert_eq!(median.data, vec![1.0, 2.0, 6.0, 6.0]);

            provider.free(&input).ok();
            provider.free(&sum_handle).ok();
            provider.free(&var_input).ok();
            provider.free(&var_handle).ok();
            provider.free(&median_input).ok();
            provider.free(&median_handle).ok();
        });
    }

    #[test]
    fn movsum_shrink_vector() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![tensor(vec![1., 2., 3., 4.], vec![1, 4]), Value::Num(3.0)],
        )
        .expect("movsum");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![1, 4]);
        assert_eq!(out.data, vec![3.0, 6.0, 9.0, 7.0]);
    }

    #[test]
    fn movmean_matrix_dimension_two() {
        let input = tensor(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let result = call(
            "movmean",
            MovingOp::Mean,
            vec![input, Value::Num(3.0), Value::Num(2.0)],
        )
        .expect("movmean");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
    }

    #[test]
    fn movprod_omitnan_and_asymmetric_window() {
        let input = tensor(vec![2., f64::NAN, 3., 4.], vec![1, 4]);
        let window = tensor(vec![1., 0.], vec![1, 2]);
        let result = call(
            "movprod",
            MovingOp::Prod,
            vec![input, window, Value::from("omitnan")],
        )
        .expect("movprod");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![2.0, 2.0, 3.0, 12.0]);
    }

    #[test]
    fn moving_window_vector_reads_typed_integer_storage_exactly() {
        let input = tensor(vec![2., f64::NAN, 3., 4.], vec![1, 4]);
        let mut window =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 2]).expect("window");
        window.data.fill(f64::NAN);

        let result = call(
            "movprod",
            MovingOp::Prod,
            vec![input, Value::Tensor(window), Value::from("omitnan")],
        )
        .expect("movprod");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![2.0, 2.0, 3.0, 12.0]);
    }

    #[test]
    fn moving_window_uses_all_integer_storage_classes_without_mirror() {
        let storages = vec![
            IntegerStorage::I8(vec![3]),
            IntegerStorage::I16(vec![3]),
            IntegerStorage::I32(vec![3]),
            IntegerStorage::I64(vec![3]),
            IntegerStorage::U8(vec![3]),
            IntegerStorage::U16(vec![3]),
            IntegerStorage::U32(vec![3]),
            IntegerStorage::U64(vec![3]),
        ];
        for storage in storages {
            let mut window = Tensor::new_integer(storage, vec![1, 1]).expect("window");
            window.data = vec![f64::NAN];
            assert!(matches!(
                block_on(parse_window("movsum", &Value::Tensor(window))),
                Ok(WindowSpec::Counts {
                    before: 1,
                    after: 1
                })
            ));
        }
    }

    #[test]
    fn movmin_fill_endpoint_includes_fill_value() {
        let result = call(
            "movmin",
            MovingOp::Min,
            vec![
                tensor(vec![4., 2., 8.], vec![1, 3]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::Num(0.0),
            ],
        )
        .expect("movmin");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![0.0, 2.0, 0.0]);
    }

    #[test]
    fn movmax_discard_endpoint_shortens_dimension() {
        let result = call(
            "movmax",
            MovingOp::Max,
            vec![
                tensor(vec![1., 9., 3., 4.], vec![1, 4]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::from("discard"),
            ],
        )
        .expect("movmax");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.data, vec![9.0, 9.0]);
    }

    #[test]
    fn movmedian_even_window_uses_average_middle() {
        let result = call(
            "movmedian",
            MovingOp::Median,
            vec![tensor(vec![1., 10., 3., 8.], vec![1, 4]), Value::Num(2.0)],
        )
        .expect("movmedian");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![1.0, 5.5, 6.5, 5.5]);
    }

    #[test]
    fn movstd_defaults_to_sample_normalization_and_accepts_population_weight() {
        let input = tensor(vec![4., 8., 6.], vec![1, 3]);
        let sample = call(
            "movstd",
            MovingOp::Std,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movstd sample");
        let population = call(
            "movstd",
            MovingOp::Std,
            vec![input, Value::Num(3.0), Value::Num(1.0)],
        )
        .expect("movstd population");
        let sample = expect_tensor(sample);
        let population = expect_tensor(population);
        assert!((sample.data[0] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((sample.data[1] - 2.0).abs() < 1e-12);
        assert!((sample.data[2] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((population.data[0] - 2.0).abs() < 1e-12);
        assert!((population.data[1] - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((population.data[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn movstd_normalization_reads_typed_integer_tensor_storage_exactly() {
        let input = tensor(vec![4., 8., 6.], vec![1, 3]);
        let mut normalization =
            Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("integer tensor");
        normalization.data.clear();

        let result = call(
            "movstd",
            MovingOp::Std,
            vec![input, Value::Num(3.0), Value::Tensor(normalization)],
        )
        .expect("movstd population");
        let output = expect_tensor(result);
        assert!((output.data[0] - 2.0).abs() < 1e-12);
        assert!((output.data[1] - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((output.data[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn moving_normalization_uses_all_integer_storage_classes_without_mirror() {
        let storages = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for storage in storages {
            let mut normalization =
                Tensor::new_integer(storage, vec![1, 1]).expect("normalization");
            normalization.data = vec![f64::NAN];
            assert!(matches!(
                block_on(parse_variance_normalization(
                    "movstd",
                    &Value::Tensor(normalization)
                )),
                Ok(Some(VarianceNormalization::Population))
            ));
        }
    }

    #[test]
    fn movstd_uses_weight_then_dimension_grammar() {
        let input = tensor(vec![4., -1., -1., 8., -2., 3., 6., -3., 4.], vec![3, 3]);
        let result = call(
            "movstd",
            MovingOp::Std,
            vec![input, Value::Num(3.0), Value::Num(0.0), Value::Num(2.0)],
        )
        .expect("movstd dim");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![3, 3]);
        assert!((out.data[0] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((out.data[1] - (0.5_f64).sqrt()).abs() < 1e-12);
        assert!((out.data[2] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((out.data[3] - 2.0).abs() < 1e-12);
        assert!((out.data[4] - 1.0).abs() < 1e-12);
        assert!((out.data[5] - (7.0_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn movstd_combines_weight_dimension_and_nanflag() {
        let input = tensor(
            vec![4., -1., -1., f64::NAN, -2., 3., 6., -3., 4.],
            vec![3, 3],
        );
        let result = call(
            "movstd",
            MovingOp::Std,
            vec![
                input,
                Value::Num(3.0),
                Value::Num(0.0),
                Value::Num(2.0),
                Value::from("omitnan"),
            ],
        )
        .expect("movstd dim omitnan");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![3, 3]);
        assert_eq!(out.data[0], 0.0);
        assert!((out.data[1] - 0.5_f64.sqrt()).abs() < 1e-12);
        assert!((out.data[2] - 8.0_f64.sqrt()).abs() < 1e-12);
        assert!((out.data[3] - 2.0_f64.sqrt()).abs() < 1e-12);
        assert_eq!(out.data[6], 0.0);
    }

    #[test]
    fn movstd_rejects_dimension_without_explicit_weight() {
        let err = call(
            "movstd",
            MovingOp::Std,
            vec![
                tensor(vec![1., 2., 3.], vec![1, 3]),
                Value::Num(3.0),
                Value::Num(2.0),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:movstd:InvalidArgument"));
    }

    #[test]
    fn movvar_supports_omitnan_endpoints_and_samplepoints() {
        let omit = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![4., f64::NAN, -1.], vec![1, 3]),
                Value::Num(3.0),
                Value::from("omitnan"),
            ],
        )
        .expect("movvar omitnan");
        assert_eq!(expect_tensor(omit).data, vec![0.0, 12.5, 0.0]);

        let fill = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![1., 2.], vec![1, 2]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::Num(0.0),
            ],
        )
        .expect("movvar fill");
        assert_eq!(expect_tensor(fill).data, vec![1.0, 1.0]);

        let sample_points = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![1., 2., 4.], vec![1, 3]),
                Value::Num(2.1),
                Value::from("SamplePoints"),
                tensor(vec![0., 1., 3.], vec![1, 3]),
            ],
        )
        .expect("movvar samplepoints");
        assert_eq!(expect_tensor(sample_points).data, vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn movvar_large_repeated_fill_uses_stable_accumulation() {
        let result = call(
            "movvar",
            MovingOp::Var,
            vec![
                tensor(vec![1.0e308], vec![1, 1]),
                Value::Num(1_000_001.0),
                Value::from("Endpoints"),
                Value::Num(1.0e308),
            ],
        )
        .expect("movvar stable fill");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![1, 1]);
        assert_eq!(out.data, vec![0.0]);
    }

    #[test]
    fn movstd_discard_endpoint_shortens_dimension() {
        let result = call(
            "movstd",
            MovingOp::Std,
            vec![
                tensor(vec![1., 2., 3., 4.], vec![1, 4]),
                Value::Num(3.0),
                Value::from("Endpoints"),
                Value::from("discard"),
            ],
        )
        .expect("movstd discard");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.data, vec![1.0, 1.0]);
    }

    #[test]
    fn complex_movvar_uses_squared_complex_distance() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1., 0.), (3., 0.), (3., 4.)], vec![1, 3]).unwrap(),
        );
        let result =
            call("movvar", MovingOp::Var, vec![input, Value::Num(3.0)]).expect("complex movvar");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.shape, vec![1, 3]);
        assert!((out.data[0].0 - 2.0).abs() < 1e-12);
        assert_eq!(out.data[0].1, 0.0);
        assert!((out.data[1].0 - 20.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn moving_include_nan_by_default() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![tensor(vec![1., f64::NAN, 3.], vec![1, 3]), Value::Num(3.0)],
        )
        .expect("movsum");
        let out = expect_tensor(result);
        assert!(out.data[0].is_nan());
        assert!(out.data[1].is_nan());
        assert!(out.data[2].is_nan());
    }

    #[test]
    fn complex_movsum_supported() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(1., 1.), (2., -1.), (3., 0.)], vec![1, 3]).unwrap(),
        );
        let result = call("movsum", MovingOp::Sum, vec![input, Value::Num(3.0)]).expect("movsum");
        let Value::ComplexTensor(out) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(out.data, vec![(3.0, 0.0), (6.0, 0.0), (5.0, -1.0)]);
    }

    #[test]
    fn samplepoints_support_irregular_numeric_windows() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![
                tensor(vec![1., 2., 3.], vec![1, 3]),
                Value::Num(2.1),
                Value::from("SamplePoints"),
                tensor(vec![0., 1., 3.], vec![1, 3]),
            ],
        )
        .expect("movsum samplepoints");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn huge_shrink_window_is_clamped_to_input_extent() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![tensor(vec![1., 2.], vec![1, 2]), Value::Num(1.0e12)],
        )
        .expect("movsum huge shrink");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![3.0, 3.0]);
    }

    #[test]
    fn huge_fill_window_accounts_for_repeated_fill_without_expansion() {
        let result = call(
            "movmean",
            MovingOp::Mean,
            vec![
                tensor(vec![1., 2.], vec![1, 2]),
                Value::Num(1.0e12),
                Value::from("Endpoints"),
                Value::Num(0.0),
            ],
        )
        .expect("movmean huge fill");
        let out = expect_tensor(result);
        assert_eq!(out.data, vec![3.0e-12, 3.0e-12]);
    }

    #[test]
    fn complex_movmin_movmax_and_movmedian_use_magnitude_ordering() {
        let input = Value::ComplexTensor(
            ComplexTensor::new(vec![(3., 0.), (1., 1.), (-2., 0.)], vec![1, 3]).unwrap(),
        );
        let min = call(
            "movmin",
            MovingOp::Min,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movmin");
        let max = call(
            "movmax",
            MovingOp::Max,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movmax");
        let median =
            call("movmedian", MovingOp::Median, vec![input, Value::Num(3.0)]).expect("median");

        let Value::ComplexTensor(min) = min else {
            panic!("expected complex movmin");
        };
        let Value::ComplexTensor(max) = max else {
            panic!("expected complex movmax");
        };
        let Value::ComplexTensor(median) = median else {
            panic!("expected complex movmedian");
        };
        assert_eq!(min.data[1], (1.0, 1.0));
        assert_eq!(max.data[1], (3.0, 0.0));
        assert_eq!(median.data[1], (-2.0, 0.0));
    }

    #[test]
    fn rejects_oversized_window_without_overflowing() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = call(
            "movsum",
            MovingOp::Sum,
            vec![
                tensor(vec![1., 2.], vec![1, 2]),
                Value::Tensor(Tensor::new(vec![boundary, 1.0], vec![1, 2]).unwrap()),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:movsum:InvalidArgument"));
    }

    #[test]
    fn dim_beyond_rank_behaves_like_trailing_singleton_dimension() {
        let result = call(
            "movsum",
            MovingOp::Sum,
            vec![
                tensor(vec![2., 3.], vec![1, 2]),
                Value::Num(3.0),
                Value::Num(4.0),
                Value::from("Endpoints"),
                Value::Num(10.0),
            ],
        )
        .expect("movsum trailing dim");
        let out = expect_tensor(result);
        assert_eq!(out.shape, vec![1, 2, 1, 1]);
        assert_eq!(out.data, vec![22.0, 23.0]);
    }

    #[test]
    fn omitnan_empty_windows_use_sum_and_product_identities() {
        let nan = tensor(vec![f64::NAN], vec![1, 1]);
        let sum = call(
            "movsum",
            MovingOp::Sum,
            vec![nan.clone(), Value::Num(1.0), Value::from("omitnan")],
        )
        .expect("movsum omitnan");
        let prod = call(
            "movprod",
            MovingOp::Prod,
            vec![nan, Value::Num(1.0), Value::from("omitnan")],
        )
        .expect("movprod omitnan");
        assert_eq!(expect_tensor(sum).data, vec![0.0]);
        assert_eq!(expect_tensor(prod).data, vec![1.0]);
    }

    #[test]
    fn movmin_omits_nan_by_default_and_accepts_missing_alias() {
        let input = tensor(vec![f64::NAN, 2.0, 3.0], vec![1, 3]);
        let defaulted = call(
            "movmin",
            MovingOp::Min,
            vec![input.clone(), Value::Num(3.0)],
        )
        .expect("movmin");
        let explicit = call(
            "movmin",
            MovingOp::Min,
            vec![input, Value::Num(3.0), Value::from("includemissing")],
        )
        .expect("movmin");
        assert_eq!(expect_tensor(defaulted).data, vec![2.0, 2.0, 2.0]);
        assert!(expect_tensor(explicit).data[0].is_nan());
    }
}
