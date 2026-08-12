//! MATLAB-compatible `fspecial` builtin for generating 2-D image filters.

use std::env;
use std::f64::consts::PI;

use log::warn;
use runmat_accelerate_api::{self, FspecialFilter, FspecialRequest};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::image::filters::type_resolvers::fspecial_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "fspecial";

const FSPECIAL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Generated 2-D correlation kernel.",
}];

const FSPECIAL_INPUTS_KIND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Filter name: 'average'|'disk'|'gaussian'|'laplacian'|'log'|'motion'|'prewitt'|'sobel'|'unsharp'.",
}];

const FSPECIAL_INPUTS_KIND_ARG1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Filter name: 'average'|'disk'|'gaussian'|'laplacian'|'log'|'motion'|'prewitt'|'sobel'|'unsharp'.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description:
            "Filter-specific parameter (size/radius/lengths/alpha/length depending on type).",
    },
];

const FSPECIAL_INPUTS_KIND_ARG2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "type",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Filter name: 'average'|'disk'|'gaussian'|'laplacian'|'log'|'motion'|'prewitt'|'sobel'|'unsharp'.",
    },
    BuiltinParamDescriptor {
        name: "arg1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description:
            "First filter-specific parameter (for example lengths or motion length).",
    },
    BuiltinParamDescriptor {
        name: "arg2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description:
            "Second filter-specific parameter (for example sigma or motion angle).",
    },
];

const FSPECIAL_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "H = fspecial(type)",
        inputs: &FSPECIAL_INPUTS_KIND,
        outputs: &FSPECIAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "H = fspecial(type, arg1)",
        inputs: &FSPECIAL_INPUTS_KIND_ARG1,
        outputs: &FSPECIAL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "H = fspecial(type, arg1, arg2)",
        inputs: &FSPECIAL_INPUTS_KIND_ARG2,
        outputs: &FSPECIAL_OUTPUT,
    },
];

const FSPECIAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSPECIAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:fspecial:InvalidArgument"),
    when: "Filter name or argument count shape is invalid for the selected filter type.",
    message: "fspecial: invalid argument",
};

const FSPECIAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSPECIAL.INVALID_INPUT",
    identifier: Some("RunMat:fspecial:InvalidInput"),
    when: "Filter parameters have invalid values or unsupported input types.",
    message: "fspecial: invalid input",
};

const FSPECIAL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSPECIAL.INTERNAL",
    identifier: Some("RunMat:fspecial:Internal"),
    when: "Kernel generation fails internally.",
    message: "fspecial: internal kernel generation failure",
};

const FSPECIAL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FSPECIAL_ERROR_INVALID_ARGUMENT,
    FSPECIAL_ERROR_INVALID_INPUT,
    FSPECIAL_ERROR_INTERNAL,
];

pub const FSPECIAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FSPECIAL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FSPECIAL_ERRORS,
};

const FSPECIAL_NONDOUBLE_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fspecial-nondouble-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fspecial with a single, integer, or logical size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FspecialNondoubleSizeExtension"),
};

const FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "fspecial-nondouble-parameter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "fspecial with a single, integer, or logical computational parameter is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:FspecialNondoubleParameterExtension"),
    };

const FSPECIAL_UNSHARP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fspecial-unsharp-filter",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fspecial with the unsharp filter type is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:FspecialUnsharpExtension"),
};

const FSPECIAL_RESIDENT_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "fspecial-resident-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "fspecial evaluated under ambient interactive resident-output policy is a RunMat extension; unsupported provider forms fall back to host output",
    error_identifier: Some("RunMat:compatibility:FspecialResidentOutputExtension"),
};

pub const FSPECIAL_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    FSPECIAL_NONDOUBLE_SIZE_EXTENSION,
    FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION,
    FSPECIAL_UNSHARP_EXTENSION,
    FSPECIAL_RESIDENT_OUTPUT_EXTENSION,
];

const FSPECIAL_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "hsize or len",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are decoded from authoritative storage as exact positive structural dimensions.",
    }];

const FSPECIAL_INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "radius, sigma, alpha, or theta",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are admitted only when exactly representable at the binary64 kernel-computation boundary.",
    }];

pub const FSPECIAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "H = fspecial(type, integer_hsize_or_len, ...)",
        inputs: &FSPECIAL_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only exact structural controls; host kernels are double, while the independently gated resident-output extension follows provider F32/F64 precision when supported.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "H = fspecial(type, integer_parameter, ...)",
        inputs: &FSPECIAL_INTEGER_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only computational controls cross one exact binary64 boundary; host kernels are double, while the independently gated resident-output extension follows provider F32/F64 precision when supported.",
    },
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::filters::fspecial")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fspecial",
    op_kind: GpuOpKind::Custom("kernel-generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("fspecial")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Average, gaussian, laplacian, prewitt, sobel, and unsharp execute on the device when supported; disk/log/motion currently gather to host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::filters::fspecial")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fspecial",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Generates constant kernels; fusion is not applicable.",
};

fn fspecial_descriptor_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("fspecial:") {
        detail.to_string()
    } else {
        format!("{}: {}", error.message, detail)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fspecial_argument_error(detail: impl AsRef<str>) -> RuntimeError {
    fspecial_descriptor_error(&FSPECIAL_ERROR_INVALID_ARGUMENT, detail)
}

fn fspecial_error(detail: impl AsRef<str>) -> RuntimeError {
    fspecial_descriptor_error(&FSPECIAL_ERROR_INVALID_INPUT, detail)
}

fn fspecial_internal_error(detail: impl AsRef<str>) -> RuntimeError {
    fspecial_descriptor_error(&FSPECIAL_ERROR_INTERNAL, detail)
}

#[derive(Clone, Copy, Debug)]
enum FilterKind {
    Average,
    Disk,
    Gaussian,
    Laplacian,
    Log,
    Motion,
    Prewitt,
    Sobel,
    Unsharp,
}

/// Shared specification of an `fspecial` kernel used by both the runtime and acceleration providers.
#[derive(Debug, Clone)]
pub enum FspecialFilterSpec {
    Average {
        rows: usize,
        cols: usize,
    },
    Disk {
        radius: f64,
        size: usize,
    },
    Gaussian {
        rows: usize,
        cols: usize,
        sigma: f64,
    },
    Laplacian {
        alpha: f64,
    },
    Log {
        rows: usize,
        cols: usize,
        sigma: f64,
    },
    Motion {
        length: usize,
        kernel_size: usize,
        angle_degrees: f64,
        oversample: usize,
    },
    Prewitt,
    Sobel,
    Unsharp {
        alpha: f64,
    },
}

impl FspecialFilterSpec {
    pub fn generate_tensor(&self) -> BuiltinResult<Tensor> {
        match self {
            FspecialFilterSpec::Average { rows, cols } => generate_average(*rows, *cols),
            FspecialFilterSpec::Disk { radius, size } => generate_disk(*radius, *size),
            FspecialFilterSpec::Gaussian { rows, cols, sigma } => {
                generate_gaussian(*rows, *cols, *sigma)
            }
            FspecialFilterSpec::Laplacian { alpha } => generate_laplacian(*alpha),
            FspecialFilterSpec::Log { rows, cols, sigma } => generate_log(*rows, *cols, *sigma),
            FspecialFilterSpec::Motion {
                length,
                kernel_size,
                angle_degrees,
                oversample,
            } => generate_motion(*length, *kernel_size, *angle_degrees, *oversample),
            FspecialFilterSpec::Prewitt => generate_prewitt(),
            FspecialFilterSpec::Sobel => generate_sobel(),
            FspecialFilterSpec::Unsharp { alpha } => generate_unsharp(*alpha),
        }
    }

    pub fn to_request(&self) -> BuiltinResult<FspecialRequest> {
        use std::convert::TryFrom;
        let filter = match self {
            FspecialFilterSpec::Average { rows, cols } => FspecialFilter::Average {
                rows: u32::try_from(*rows)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                cols: u32::try_from(*cols)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
            },
            FspecialFilterSpec::Disk { radius, size } => FspecialFilter::Disk {
                radius: *radius,
                size: u32::try_from(*size)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
            },
            FspecialFilterSpec::Gaussian { rows, cols, sigma } => FspecialFilter::Gaussian {
                rows: u32::try_from(*rows)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                cols: u32::try_from(*cols)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                sigma: *sigma,
            },
            FspecialFilterSpec::Laplacian { alpha } => FspecialFilter::Laplacian { alpha: *alpha },
            FspecialFilterSpec::Log { rows, cols, sigma } => FspecialFilter::Log {
                rows: u32::try_from(*rows)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                cols: u32::try_from(*cols)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                sigma: *sigma,
            },
            FspecialFilterSpec::Motion {
                length,
                kernel_size,
                angle_degrees,
                oversample,
            } => FspecialFilter::Motion {
                length: u32::try_from(*length)
                    .map_err(|_| fspecial_error("fspecial: LENGTH exceeds GPU limits"))?,
                kernel_size: u32::try_from(*kernel_size)
                    .map_err(|_| fspecial_error("fspecial: kernel dimensions exceed GPU limits"))?,
                angle_degrees: *angle_degrees,
                oversample: u32::try_from(*oversample)
                    .map_err(|_| fspecial_error("fspecial: oversample exceeds GPU limits"))?,
            },
            FspecialFilterSpec::Prewitt => FspecialFilter::Prewitt,
            FspecialFilterSpec::Sobel => FspecialFilter::Sobel,
            FspecialFilterSpec::Unsharp { alpha } => FspecialFilter::Unsharp { alpha: *alpha },
        };
        Ok(FspecialRequest { filter })
    }

    fn is_gpu_supported(&self) -> bool {
        matches!(
            self,
            FspecialFilterSpec::Average { .. }
                | FspecialFilterSpec::Gaussian { .. }
                | FspecialFilterSpec::Laplacian { .. }
                | FspecialFilterSpec::Prewitt
                | FspecialFilterSpec::Sobel
                | FspecialFilterSpec::Unsharp { .. }
        )
    }
}

/// Convert an API request into a runtime specification.
#[allow(dead_code)]
pub fn spec_from_request(filter: &FspecialFilter) -> BuiltinResult<FspecialFilterSpec> {
    Ok(match filter {
        FspecialFilter::Average { rows, cols } => FspecialFilterSpec::Average {
            rows: *rows as usize,
            cols: *cols as usize,
        },
        FspecialFilter::Disk { radius, size } => FspecialFilterSpec::Disk {
            radius: *radius,
            size: *size as usize,
        },
        FspecialFilter::Gaussian { rows, cols, sigma } => FspecialFilterSpec::Gaussian {
            rows: *rows as usize,
            cols: *cols as usize,
            sigma: *sigma,
        },
        FspecialFilter::Laplacian { alpha } => FspecialFilterSpec::Laplacian { alpha: *alpha },
        FspecialFilter::Log { rows, cols, sigma } => FspecialFilterSpec::Log {
            rows: *rows as usize,
            cols: *cols as usize,
            sigma: *sigma,
        },
        FspecialFilter::Motion {
            length,
            kernel_size,
            angle_degrees,
            oversample,
        } => FspecialFilterSpec::Motion {
            length: *length as usize,
            kernel_size: *kernel_size as usize,
            angle_degrees: *angle_degrees,
            oversample: *oversample as usize,
        },
        FspecialFilter::Prewitt => FspecialFilterSpec::Prewitt,
        FspecialFilter::Sobel => FspecialFilterSpec::Sobel,
        FspecialFilter::Unsharp { alpha } => FspecialFilterSpec::Unsharp { alpha: *alpha },
    })
}

#[runtime_builtin(
    name = "fspecial",
    category = "image/filters",
    summary = "Generate standard 2-D filter kernels.",
    keywords = "fspecial,filter,gaussian,sobel,motion,laplacian,disk",
    accel = "array_construct",
    type_resolver(fspecial_type),
    descriptor(crate::builtins::image::filters::fspecial::FSPECIAL_DESCRIPTOR),
    extensions(crate::builtins::image::filters::fspecial::FSPECIAL_EXTENSIONS),
    integer_capabilities(crate::builtins::image::filters::fspecial::FSPECIAL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::image::filters::fspecial"
)]
async fn fspecial_builtin(kind: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_fspecial_extensions_enabled(&kind, &rest)?;
    if should_materialize_on_gpu() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FSPECIAL_RESIDENT_OUTPUT_EXTENSION,
            NAME,
        )?;
    }
    let spec = build_filter_spec(&kind, &rest)?;
    let tensor = spec.generate_tensor()?;
    finalize_output(&spec, tensor)
}

fn ensure_fspecial_extensions_enabled(kind: &Value, rest: &[Value]) -> BuiltinResult<()> {
    let filter_kind = parse_filter_kind(kind)?;
    if matches!(filter_kind, FilterKind::Unsharp) {
        crate::compatibility::ensure_builtin_extension_enabled(&FSPECIAL_UNSHARP_EXTENSION, NAME)?;
    }
    let (size_indices, parameter_indices): (&[usize], &[usize]) = match filter_kind {
        FilterKind::Average => (&[0], &[]),
        FilterKind::Disk | FilterKind::Laplacian | FilterKind::Unsharp => (&[], &[0]),
        FilterKind::Gaussian | FilterKind::Log => (&[0], &[1]),
        FilterKind::Motion => (&[0], &[1]),
        FilterKind::Prewitt | FilterKind::Sobel => (&[], &[]),
    };
    for &index in size_indices {
        if rest.get(index).is_some_and(is_nondouble_numeric_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FSPECIAL_NONDOUBLE_SIZE_EXTENSION,
                NAME,
            )?;
        }
    }
    for &index in parameter_indices {
        if rest.get(index).is_some_and(is_nondouble_numeric_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

fn is_nondouble_numeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() != NumericDType::F64)
}

fn build_filter_spec(kind: &Value, rest: &[Value]) -> BuiltinResult<FspecialFilterSpec> {
    let filter_kind = parse_filter_kind(kind)?;
    match filter_kind {
        FilterKind::Average => {
            ensure_arg_count("average", rest, 0, 1)?;
            let (rows, cols) = parse_average_dims(rest.first())?;
            Ok(FspecialFilterSpec::Average { rows, cols })
        }
        FilterKind::Disk => {
            ensure_arg_count("disk", rest, 0, 1)?;
            let (radius, size) = parse_disk_params(rest.first())?;
            Ok(FspecialFilterSpec::Disk { radius, size })
        }
        FilterKind::Gaussian => {
            ensure_arg_count("gaussian", rest, 0, 2)?;
            let (rows, cols, sigma) = parse_gaussian_params(rest.first(), rest.get(1))?;
            Ok(FspecialFilterSpec::Gaussian { rows, cols, sigma })
        }
        FilterKind::Laplacian => {
            ensure_arg_count("laplacian", rest, 0, 1)?;
            let alpha = parse_laplacian_alpha(rest.first())?;
            Ok(FspecialFilterSpec::Laplacian { alpha })
        }
        FilterKind::Log => {
            ensure_arg_count("log", rest, 0, 2)?;
            let (rows, cols, sigma) = parse_log_params(rest.first(), rest.get(1))?;
            Ok(FspecialFilterSpec::Log { rows, cols, sigma })
        }
        FilterKind::Motion => {
            ensure_arg_count("motion", rest, 0, 2)?;
            let (length, kernel_size, angle, oversample) =
                parse_motion_params(rest.first(), rest.get(1))?;
            Ok(FspecialFilterSpec::Motion {
                length,
                kernel_size,
                angle_degrees: angle,
                oversample,
            })
        }
        FilterKind::Prewitt => {
            ensure_arg_count("prewitt", rest, 0, 0)?;
            Ok(FspecialFilterSpec::Prewitt)
        }
        FilterKind::Sobel => {
            ensure_arg_count("sobel", rest, 0, 0)?;
            Ok(FspecialFilterSpec::Sobel)
        }
        FilterKind::Unsharp => {
            ensure_arg_count("unsharp", rest, 0, 1)?;
            let alpha = parse_unsharp_alpha(rest.first())?;
            Ok(FspecialFilterSpec::Unsharp { alpha })
        }
    }
}

fn finalize_output(spec: &FspecialFilterSpec, tensor: Tensor) -> BuiltinResult<Value> {
    finalize_output_with_gpu_policy(spec, tensor, should_materialize_on_gpu())
}

fn finalize_output_with_gpu_policy(
    spec: &FspecialFilterSpec,
    tensor: Tensor,
    materialize_on_gpu: bool,
) -> BuiltinResult<Value> {
    if !materialize_on_gpu || !spec.is_gpu_supported() {
        return Ok(Value::Tensor(tensor));
    }

    #[cfg(all(test, feature = "wgpu"))]
    {
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if let Some(provider) = runmat_accelerate_api::provider() {
        match spec.to_request() {
            Ok(request) => match provider.fspecial(&request) {
                Ok(handle) => return Ok(Value::GpuTensor(handle)),
                Err(err) => {
                    warn!("fspecial: provider hook unavailable, falling back to host path: {err}")
                }
            },
            Err(error) => {
                warn!(
                    "fspecial: provider hook unavailable, falling back to host path: {}",
                    error.message()
                );
            }
        }
    }

    Ok(Value::Tensor(tensor))
}

fn parse_filter_kind(value: &Value) -> BuiltinResult<FilterKind> {
    let text = value_to_string(value)
        .ok_or_else(|| fspecial_argument_error("first argument must be a string filter name"))?;
    let lower = text.to_ascii_lowercase();
    match lower.as_str() {
        "average" => Ok(FilterKind::Average),
        "disk" => Ok(FilterKind::Disk),
        "gaussian" => Ok(FilterKind::Gaussian),
        "laplacian" => Ok(FilterKind::Laplacian),
        "log" => Ok(FilterKind::Log),
        "motion" => Ok(FilterKind::Motion),
        "prewitt" => Ok(FilterKind::Prewitt),
        "sobel" => Ok(FilterKind::Sobel),
        "unsharp" => Ok(FilterKind::Unsharp),
        other => Err(fspecial_argument_error(format!(
            "fspecial: filter type '{other}' is not supported"
        ))),
    }
}

fn ensure_arg_count(name: &str, args: &[Value], min: usize, max: usize) -> BuiltinResult<()> {
    if args.len() < min || args.len() > max {
        if min == max {
            Err(fspecial_argument_error(format!(
                "fspecial: '{name}' expects exactly {min} argument{}",
                if min == 1 { "" } else { "s" }
            )))
        } else {
            Err(fspecial_argument_error(format!(
                "fspecial: '{name}' expects between {min} and {max} arguments"
            )))
        }
    } else {
        Ok(())
    }
}

fn parse_average_dims(arg: Option<&Value>) -> BuiltinResult<(usize, usize)> {
    match arg {
        None => Ok((3, 3)),
        Some(value) => {
            let dims = parse_lengths_strict(value, "fspecial: LENGTHS must be positive integers")?;
            match dims.len() {
                1 => Ok((dims[0], dims[0])),
                2 => Ok((dims[0], dims[1])),
                _ => Err(fspecial_error(
                    "fspecial: LENGTHS must be a scalar or two-element vector",
                )),
            }
        }
    }
}

fn parse_disk_params(arg: Option<&Value>) -> BuiltinResult<(f64, usize)> {
    let radius = match arg {
        None => 5.0,
        Some(value) => to_positive_scalar(value, "fspecial: RADIUS must be a non-negative scalar")?,
    };
    if radius < 0.0 {
        return Err(fspecial_error("fspecial: RADIUS must be non-negative"));
    }
    let extent_raw = radius.ceil();
    if extent_raw > isize::MAX as f64 {
        return Err(fspecial_error("fspecial: RADIUS is too large"));
    }
    let extent = extent_raw as isize;
    let size = extent
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| fspecial_error("fspecial: RADIUS is too large"))?;
    Ok((radius, size))
}

fn parse_gaussian_params(
    lengths: Option<&Value>,
    sigma_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64)> {
    let dims = match lengths {
        None => vec![3, 3],
        Some(value) => parse_lengths_strict(
            value,
            "fspecial: LENGTHS must be positive integers for gaussian",
        )?,
    };
    let dims = match dims.len() {
        1 => vec![dims[0], dims[0]],
        2 => dims,
        _ => {
            return Err(fspecial_error(
                "fspecial: gaussian lengths must be a scalar or a two-element vector",
            ));
        }
    };
    let sigma = match sigma_value {
        None => 0.5,
        Some(value) => {
            let sigma = to_positive_scalar(value, "fspecial: SIGMA must be a positive scalar")?;
            if sigma <= 0.0 {
                return Err(fspecial_error("fspecial: SIGMA must be positive"));
            }
            sigma
        }
    };
    Ok((dims[0], dims[1], sigma))
}

fn parse_laplacian_alpha(arg: Option<&Value>) -> BuiltinResult<f64> {
    match arg {
        None => Ok(0.2),
        Some(value) => {
            let alpha = to_scalar(value, "fspecial: ALPHA must be a scalar")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(fspecial_error("fspecial: ALPHA must be between 0 and 1"));
            }
            Ok(alpha)
        }
    }
}

fn parse_log_params(
    lengths: Option<&Value>,
    sigma_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64)> {
    let dims = match lengths {
        None => vec![5, 5],
        Some(value) => {
            parse_lengths_strict(value, "fspecial: LENGTHS must be positive integers for log")?
        }
    };
    let dims = match dims.len() {
        1 => vec![dims[0], dims[0]],
        2 => dims,
        _ => {
            return Err(fspecial_error(
                "fspecial: log lengths must be a scalar or two-element vector",
            ));
        }
    };
    let sigma = match sigma_value {
        None => 0.5,
        Some(value) => {
            let sigma = to_positive_scalar(value, "fspecial: SIGMA must be a positive scalar")?;
            if sigma <= 0.0 {
                return Err(fspecial_error("fspecial: SIGMA must be positive"));
            }
            sigma
        }
    };
    Ok((dims[0], dims[1], sigma))
}

fn parse_motion_params(
    length_value: Option<&Value>,
    angle_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64, usize)> {
    let length = parse_motion_length(length_value)?;
    let kernel_size = if length % 2 == 1 {
        length
    } else {
        length
            .checked_add(1)
            .ok_or_else(|| fspecial_error("fspecial: LENGTH is too large"))?
    };
    let angle_deg = match angle_value {
        None => 0.0,
        Some(value) => to_scalar(value, "fspecial: ANGLE must be a scalar")?,
    };
    Ok((length, kernel_size, angle_deg, 8))
}

fn parse_motion_length(value: Option<&Value>) -> BuiltinResult<usize> {
    let Some(value) = value else {
        return Ok(9);
    };
    if let Some(length) = integer_scalar_dimension(value)? {
        if length == 0 {
            return Err(fspecial_error("fspecial: LENGTH must be positive"));
        }
        return Ok(length);
    }

    let len = to_positive_scalar(value, "fspecial: LENGTH must be a positive scalar")?;
    if len <= 0.0 {
        return Err(fspecial_error("fspecial: LENGTH must be positive"));
    }
    let rounded = len.round();
    if !fits_platform_usize(rounded) {
        return Err(fspecial_error("fspecial: LENGTH is too large"));
    }
    let length = rounded as usize;
    if length == 0 {
        return Err(fspecial_error("fspecial: LENGTH must be at least 1"));
    }
    Ok(length)
}

fn parse_unsharp_alpha(arg: Option<&Value>) -> BuiltinResult<f64> {
    match arg {
        None => Ok(0.2),
        Some(value) => {
            let alpha = to_scalar(value, "fspecial: ALPHA must be a scalar")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(fspecial_error("fspecial: ALPHA must be between 0 and 1"));
            }
            Ok(alpha)
        }
    }
}

fn generate_average(rows: usize, cols: usize) -> BuiltinResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| fspecial_error("fspecial: LENGTHS are too large"))?;
    if total == 0 {
        return Err(fspecial_error(
            "fspecial: LENGTHS must be positive integers",
        ));
    }
    let fill = 1.0 / total as f64;
    let data = vec![fill; total];
    Tensor::new(data, vec![rows, cols])
        .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn generate_disk(radius: f64, size: usize) -> BuiltinResult<Tensor> {
    if radius == 0.0 {
        return Tensor::new(vec![1.0], vec![1, 1])
            .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")));
    }

    let total = size
        .checked_mul(size)
        .ok_or_else(|| fspecial_error("fspecial: RADIUS is too large"))?;
    let mut data = vec![0.0f64; total];
    let center = (size as isize / 2) as f64;

    for row in 0..size {
        let y1 = row as f64 - center - 0.5;
        let y2 = y1 + 1.0;
        for col in 0..size {
            let x1 = col as f64 - center - 0.5;
            let x2 = x1 + 1.0;
            let area = circle_rect_area(radius, x1, x2, y1, y2);
            data[col * size + row] = area;
        }
    }

    let normaliser = PI * radius * radius;
    if normaliser <= f64::EPSILON {
        return Err(fspecial_error("fspecial: radius is too small"));
    }
    let mut sum = 0.0;
    for value in &mut data {
        *value /= normaliser;
        sum += *value;
    }
    if sum <= 0.0 {
        return Err(fspecial_error("fspecial: failed to generate disk filter"));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![size, size])
        .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn generate_gaussian(rows: usize, cols: usize, sigma: f64) -> BuiltinResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| fspecial_error("fspecial: LENGTHS are too large"))?;
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let denom = 2.0 * sigma * sigma;
    let mut data = Vec::with_capacity(total);
    let mut sum = 0.0;
    for col in 0..cols {
        let x = col as f64 - col_center;
        for row in 0..rows {
            let y = row as f64 - row_center;
            let value = (-((x * x + y * y) / denom)).exp();
            data.push(value);
            sum += value;
        }
    }
    if sum == 0.0 {
        return Err(fspecial_error(
            "fspecial: gaussian generation failed (degenerate sigma)",
        ));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![rows, cols])
        .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn generate_laplacian(alpha: f64) -> BuiltinResult<Tensor> {
    let scale = 4.0 / (alpha + 1.0);
    let a = alpha / 4.0;
    let b = (1.0 - alpha) / 4.0;
    let mut data = vec![
        a, b, a, //
        b, -1.0, b, //
        a, b, a,
    ];
    for value in &mut data {
        *value *= scale;
    }
    Tensor::new(data, vec![3, 3]).map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn generate_log(rows: usize, cols: usize, sigma: f64) -> BuiltinResult<Tensor> {
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| fspecial_error("fspecial: LENGTHS are too large"))?;
    let row_center = (rows as f64 - 1.0) / 2.0;
    let col_center = (cols as f64 - 1.0) / 2.0;
    let mut gauss = Vec::with_capacity(total);
    let mut gauss_sum = 0.0;
    for col in 0..cols {
        let x = col as f64 - col_center;
        for row in 0..rows {
            let y = row as f64 - row_center;
            let value = (-((x * x + y * y) / (2.0 * sigma * sigma))).exp();
            gauss_sum += value;
            gauss.push((x, y, value));
        }
    }
    if gauss_sum == 0.0 {
        return Err(fspecial_error(
            "fspecial: failed to normalise Laplacian of Gaussian",
        ));
    }
    let mut data = Vec::with_capacity(rows * cols);
    let sigma2 = sigma * sigma;
    let normaliser = 2.0 * PI * sigma6(sigma);
    for (x, y, g) in gauss {
        let radial = x * x + y * y;
        let value = ((radial - 2.0 * sigma2) * g) / normaliser;
        data.push(value / gauss_sum);
    }
    let sum: f64 = data.iter().sum();
    if sum != 0.0 {
        let correction = sum / data.len() as f64;
        for value in &mut data {
            *value -= correction;
        }
    }
    Tensor::new(data, vec![rows, cols])
        .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn sigma6(sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    sigma2 * sigma2 * sigma2
}

fn generate_motion(
    length: usize,
    kernel_size: usize,
    angle_degrees: f64,
    oversample: usize,
) -> BuiltinResult<Tensor> {
    let total = kernel_size
        .checked_mul(kernel_size)
        .ok_or_else(|| fspecial_error("fspecial: LENGTH is too large"))?;
    let total_samples = length
        .checked_mul(oversample)
        .ok_or_else(|| fspecial_error("fspecial: LENGTH is too large"))?;
    let mut data = vec![0.0f64; total];
    let center = (kernel_size as f64 - 1.0) / 2.0;
    let theta = angle_degrees.to_radians();
    let dir_x = theta.cos();
    let dir_y = theta.sin();
    let step = 1.0 / oversample as f64;
    let half = (length as f64 - 1.0) / 2.0;

    for idx in 0..total_samples {
        let t = -half + (idx as f64 + 0.5) * step;
        let x = center + t * dir_x;
        let y = center + t * dir_y;
        deposit_bilinear(&mut data, kernel_size, x, y, 1.0);
    }

    let mut sum = 0.0;
    for value in &data {
        sum += *value;
    }
    if sum == 0.0 {
        return Err(fspecial_error("fspecial: failed to build motion kernel"));
    }
    for value in &mut data {
        *value /= sum;
    }

    Tensor::new(data, vec![kernel_size, kernel_size])
        .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn deposit_bilinear(data: &mut [f64], size: usize, x: f64, y: f64, contribution: f64) {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let xi = xi as isize;
    let yi = yi as isize;
    for dy in 0..=1 {
        for dx in 0..=1 {
            let px = xi + dx;
            let py = yi + dy;
            if px < 0 || py < 0 || px >= size as isize || py >= size as isize {
                continue;
            }
            let wx = if dx == 0 { 1.0 - xf } else { xf };
            let wy = if dy == 0 { 1.0 - yf } else { yf };
            let weight = wx * wy;
            let idx = (px as usize) * size + py as usize;
            data[idx] += contribution * weight;
        }
    }
}

fn generate_prewitt() -> BuiltinResult<Tensor> {
    Tensor::new(
        vec![
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0, //
            1.0, 0.0, -1.0,
        ],
        vec![3, 3],
    )
    .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn generate_sobel() -> BuiltinResult<Tensor> {
    Tensor::new(
        vec![
            1.0, 0.0, -1.0, //
            2.0, 0.0, -2.0, //
            1.0, 0.0, -1.0,
        ],
        vec![3, 3],
    )
    .map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn generate_unsharp(alpha: f64) -> BuiltinResult<Tensor> {
    let denom = alpha + 1.0;
    let mut data = vec![
        -alpha,
        alpha - 1.0,
        -alpha,
        alpha - 1.0,
        alpha + 5.0,
        alpha - 1.0,
        -alpha,
        alpha - 1.0,
        -alpha,
    ];
    for value in &mut data {
        *value /= denom;
    }
    Tensor::new(data, vec![3, 3]).map_err(|e| fspecial_internal_error(format!("fspecial: {e}")))
}

fn should_materialize_on_gpu() -> bool {
    match env::var("RUNMAT_ACCEL_FSPECIAL_DEVICE") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn parse_lengths_strict(value: &Value, err: &str) -> BuiltinResult<Vec<usize>> {
    parse_lengths_inner(value, err, true)
}

fn parse_lengths_inner(
    value: &Value,
    err: &str,
    enforce_positive: bool,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(i) => {
            let len = parse_integer_dimension(i)?;
            if enforce_positive && len == 0 {
                return Err(fspecial_error(err));
            }
            Ok(vec![len])
        }
        Value::Bool(value) => {
            parse_numeric_dimension(if *value { 1.0 } else { 0.0 }).map(|dimension| vec![dimension])
        }
        Value::Num(n) => parse_numeric_dimension(*n).map(|d| vec![d]),
        Value::Tensor(tensor) => {
            let dims = (0..tensor.len())
                .map(|index| {
                    let value = tensor
                        .numeric_value_at(index)
                        .ok_or_else(|| fspecial_error(err))?;
                    if let Some(value) = value.into_int_value() {
                        parse_integer_dimension(&value)
                    } else {
                        parse_numeric_dimension(value.materialize_f64())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            if enforce_positive && dims.contains(&0) {
                return Err(fspecial_error(err));
            }
            Ok(dims)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != logical.shape.iter().product::<usize>() {
                return Err(fspecial_error(err));
            }
            let dims = logical
                .data
                .iter()
                .map(|&v| parse_numeric_dimension(v as f64))
                .collect::<Result<Vec<_>, _>>()?;
            if enforce_positive && dims.contains(&0) {
                return Err(fspecial_error(err));
            }
            Ok(dims)
        }
        _ => Err(fspecial_error(err)),
    }
}

fn parse_integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| fspecial_error("fspecial: dimensions must be non-negative"))
}

fn parse_numeric_dimension(n: f64) -> BuiltinResult<usize> {
    if !n.is_finite() {
        return Err(fspecial_error("fspecial: dimensions must be finite"));
    }
    if n < 0.0 {
        return Err(fspecial_error("fspecial: dimensions must be non-negative"));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(fspecial_error("fspecial: dimensions must be integers"));
    }
    if !fits_platform_usize(rounded) {
        return Err(fspecial_error(
            "fspecial: dimensions are outside the supported platform range",
        ));
    }
    Ok(rounded as usize)
}

fn integer_scalar_dimension(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Int(value) => parse_integer_dimension(value).map(Some),
        Value::Tensor(tensor) => {
            let Some(storage) = tensor.integer_storage() else {
                return Ok(None);
            };
            if storage.len() != 1 {
                return Ok(None);
            }
            let value = storage
                .value_at(0)
                .ok_or_else(|| fspecial_error("fspecial: dimensions must be scalar"))?;
            parse_integer_dimension(&value).map(Some)
        }
        _ => Ok(None),
    }
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

fn to_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(integer) => exact_integer_f64(integer, err),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] == 0 { 0.0 } else { 1.0 })
        }
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let scalar = tensor
                .numeric_value_at(0)
                .ok_or_else(|| fspecial_error(err))?;
            numeric_scalar_f64(scalar, err)
        }
        _ => Err(fspecial_error(err)),
    }
}

fn numeric_scalar_f64(value: NumericScalar, err: &str) -> BuiltinResult<f64> {
    if let Some(integer) = value.into_int_value() {
        exact_integer_f64(&integer, err)
    } else {
        Ok(value.materialize_f64())
    }
}

fn exact_integer_f64(value: &IntValue, err: &str) -> BuiltinResult<f64> {
    if crate::builtins::math::trigonometry::cos::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(fspecial_error(format!(
            "{err}; integer value must be exactly representable as double"
        )))
    }
}

fn to_positive_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    let scalar = to_scalar(value, err)?;
    if scalar.is_nan() || scalar.is_infinite() {
        return Err(fspecial_error(err));
    }
    Ok(scalar)
}

fn circle_rect_area(radius: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let r = radius;
    if (x1 >= r || y1 >= r || x2 <= -r || y2 <= -r) && min_distance_to_circle(x1, y1, x2, y2) >= r {
        return 0.0;
    }

    if x1 < 0.0 && x2 > 0.0 {
        let left = circle_rect_area(r, x1, 0.0, y1, y2);
        let right = circle_rect_area(r, 0.0, x2, y1, y2);
        return left + right;
    }
    if y1 < 0.0 && y2 > 0.0 {
        let bottom = circle_rect_area(r, x1, x2, y1, 0.0);
        let top = circle_rect_area(r, x1, x2, 0.0, y2);
        return bottom + top;
    }
    if x2 <= 0.0 {
        return circle_rect_area(r, -x2, -x1, y1, y2);
    }
    if y2 <= 0.0 {
        return circle_rect_area(r, x1, x2, -y2, -y1);
    }
    circle_rect_area_first_quadrant(r, x1.max(0.0), x2.min(r), y1.max(0.0), y2.min(r))
}

fn min_distance_to_circle(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let cx = if x1 > 0.0 {
        x1
    } else if x2 < 0.0 {
        x2
    } else {
        0.0
    };
    let cy = if y1 > 0.0 {
        y1
    } else if y2 < 0.0 {
        y2
    } else {
        0.0
    };
    (cx * cx + cy * cy).sqrt()
}

fn circle_rect_area_first_quadrant(radius: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let r = radius;
    if x1 >= r || y1 >= r {
        return 0.0;
    }
    let xa = x1.max(0.0);
    let xb = x2.min(r);
    if xb <= xa {
        return 0.0;
    }
    let ya = y1.max(0.0);
    let yb = y2.min(r);
    if yb <= ya {
        return 0.0;
    }
    let rsq = r * r;
    let x_for_y = |y: f64| -> f64 {
        if y >= r {
            0.0
        } else {
            (rsq - y * y).sqrt()
        }
    };
    let x_full = xb.min(x_for_y(yb));
    let mut area = 0.0;
    if x_full > xa {
        area += (yb - ya) * (x_full - xa);
    }
    let x_partial_start = x_full.max(xa);
    let x_partial_end = xb.min(x_for_y(ya));
    if x_partial_end > x_partial_start {
        let arc = arc_integral(r, x_partial_start, x_partial_end);
        area += arc - ya * (x_partial_end - x_partial_start);
    }
    area
}

fn arc_integral(radius: f64, a: f64, b: f64) -> f64 {
    primitive_arc(radius, b) - primitive_arc(radius, a)
}

fn primitive_arc(radius: f64, x: f64) -> f64 {
    let r = radius;
    if x <= -r {
        return -0.5 * PI * r * r;
    }
    if x >= r {
        return 0.5 * PI * r * r;
    }
    let term = (r * r - x * x).max(0.0).sqrt();
    0.5 * (x * term + r * r * clamp_asin(x / r))
}

fn clamp_asin(value: f64) -> f64 {
    value.clamp(-1.0, 1.0).asin()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn assert_close(actual: f64, expected: f64, epsilon: f64) {
        if (actual - expected).abs() > epsilon {
            panic!(
                "values differ: actual={actual:.15e}, expected={expected:.15e}, epsilon={epsilon:.3e}"
            );
        }
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    struct FspecialDeviceEnvRestore {
        previous: Option<OsString>,
    }

    impl Drop for FspecialDeviceEnvRestore {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.as_ref() {
                std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", previous);
            } else {
                std::env::remove_var("RUNMAT_ACCEL_FSPECIAL_DEVICE");
            }
        }
    }

    fn fspecial_env_guard() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn fspecial_device_env_restore() -> FspecialDeviceEnvRestore {
        FspecialDeviceEnvRestore {
            previous: std::env::var_os("RUNMAT_ACCEL_FSPECIAL_DEVICE"),
        }
    }

    fn fspecial_host_tensor(kind: &str, args: Vec<Value>) -> Tensor {
        let _env_guard = fspecial_env_guard();
        let _restore = fspecial_device_env_restore();
        std::env::remove_var("RUNMAT_ACCEL_FSPECIAL_DEVICE");
        match block_on(fspecial_builtin(Value::from(kind), args)).unwrap() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn all_integer_scalar_tensors(value: u8) -> Vec<Tensor> {
        vec![
            Tensor::new_integer(IntegerStorage::I8(vec![value as i8]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I16(vec![value as i16]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I32(vec![value as i32]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::I64(vec![value as i64]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U8(vec![value]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U16(vec![value as u16]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U32(vec![value as u32]), vec![1, 1]).unwrap(),
            Tensor::new_integer(IntegerStorage::U64(vec![value as u64]), vec![1, 1]).unwrap(),
        ]
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_default() {
        let tensor = fspecial_host_tensor("average", Vec::new());
        assert_eq!(tensor.shape, vec![3, 3]);
        for value in tensor.materialize_f64() {
            assert_close(value, 1.0 / 9.0, 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_scalar_size() {
        let args = vec![Value::from(5.0)];
        let tensor = fspecial_host_tensor("average", args);
        assert_eq!(tensor.shape, vec![5, 5]);
        let sum: f64 = tensor.materialize_f64().iter().sum();
        assert_close(sum, 1.0, 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_rectangular_size() {
        let args = vec![Value::from(
            Tensor::new(vec![4.0, 6.0], vec![1, 2]).unwrap(),
        )];
        let tensor = fspecial_host_tensor("average", args);
        assert_eq!(tensor.shape, vec![4, 6]);
        let expected = 1.0 / (4.0 * 6.0);
        for value in tensor.materialize_f64() {
            assert_close(value, expected, 1e-12);
        }
    }

    #[test]
    fn fspecial_lengths_preserve_typed_integer_tensor_bounds() {
        let dims = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![2, 4]), vec![1, 2])
            .expect("dims");
        assert_eq!(
            parse_lengths_strict(
                &Value::Tensor(dims),
                "fspecial: LENGTHS must be positive integers",
            )
            .unwrap(),
            vec![2, 4]
        );

        let negative = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-1]), vec![1, 1])
            .expect("negative");
        assert!(parse_lengths_strict(
            &Value::Tensor(negative),
            "fspecial: LENGTHS must be positive integers",
        )
        .is_err());
    }

    #[test]
    fn fspecial_accepts_all_integer_classes_from_authoritative_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for tensor in all_integer_scalar_tensors(2) {
            let spec = build_filter_spec(&Value::from("average"), &[Value::Tensor(tensor)])
                .expect("integer size");
            assert!(matches!(
                spec,
                FspecialFilterSpec::Average { rows: 2, cols: 2 }
            ));
        }
        for tensor in all_integer_scalar_tensors(1) {
            let spec = build_filter_spec(&Value::from("laplacian"), &[Value::Tensor(tensor)])
                .expect("integer parameter");
            assert!(matches!(spec, FspecialFilterSpec::Laplacian { alpha: 1.0 }));
        }
    }

    #[test]
    fn fspecial_rejects_inexact_wide_integer_computational_parameters() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = (1_u64 << 53) + 1;
        let radius = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap();
        let error = build_filter_spec(&Value::from("disk"), &[Value::Tensor(radius)])
            .expect_err("inexact radius");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn fspecial_nondouble_roles_and_runmat_forms_are_independently_gated() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

        let integer_size = block_on(fspecial_builtin(
            Value::from("average"),
            vec![Value::Int(IntValue::U8(2))],
        ))
        .expect_err("integer size extension");
        assert_eq!(
            integer_size.identifier(),
            FSPECIAL_NONDOUBLE_SIZE_EXTENSION.error_identifier
        );

        let integer_parameter = block_on(fspecial_builtin(
            Value::from("laplacian"),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .expect_err("integer parameter extension");
        assert_eq!(
            integer_parameter.identifier(),
            FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION.error_identifier
        );

        let logical_size = block_on(fspecial_builtin(
            Value::from("average"),
            vec![Value::Bool(true)],
        ))
        .expect_err("logical size extension");
        assert_eq!(
            logical_size.identifier(),
            FSPECIAL_NONDOUBLE_SIZE_EXTENSION.error_identifier
        );

        let single_parameter =
            Tensor::from_numeric_storage(runmat_value::NumericStorage::F32(vec![1.0]), vec![1, 1])
                .unwrap();
        let single_parameter = block_on(fspecial_builtin(
            Value::from("laplacian"),
            vec![Value::Tensor(single_parameter)],
        ))
        .expect_err("single parameter extension");
        assert_eq!(
            single_parameter.identifier(),
            FSPECIAL_NONDOUBLE_PARAMETER_EXTENSION.error_identifier
        );

        let unsharp = block_on(fspecial_builtin(Value::from("unsharp"), Vec::new()))
            .expect_err("unsharp extension");
        assert_eq!(
            unsharp.identifier(),
            FSPECIAL_UNSHARP_EXTENSION.error_identifier
        );
    }

    #[test]
    fn fspecial_resident_output_policy_is_gated_before_provider_dispatch() {
        let _env_guard = fspecial_env_guard();
        let _restore = fspecial_device_env_restore();
        std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(fspecial_builtin(Value::from("average"), Vec::new()))
            .expect_err("resident output extension");
        assert_eq!(
            error.identifier(),
            FSPECIAL_RESIDENT_OUTPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn fspecial_integer_capabilities_distinguish_structural_and_floating_roles() {
        assert_eq!(FSPECIAL_INTEGER_CAPABILITIES.len(), 2);
        assert_eq!(FSPECIAL_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        assert_eq!(
            FSPECIAL_INTEGER_CAPABILITIES[0].computation_domain,
            BuiltinIntegerComputationDomain::Structural
        );
        assert_eq!(
            FSPECIAL_INTEGER_CAPABILITIES[1].computation_domain,
            BuiltinIntegerComputationDomain::FloatingPoint
        );
    }

    #[test]
    fn fspecial_dimension_parsers_reject_unrepresentable_double_bounds() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        assert!(parse_numeric_dimension(boundary).is_err());

        let dims = Tensor::new(vec![boundary], vec![1, 1]).expect("dims");
        assert!(parse_lengths_strict(
            &Value::Tensor(dims),
            "fspecial: LENGTHS must be positive integers",
        )
        .is_err());
    }

    #[test]
    fn fspecial_motion_length_preserves_typed_integer_scalar_bounds() {
        assert_eq!(
            parse_motion_length(Some(&Value::Int(IntValue::U64(17)))).unwrap(),
            17
        );

        let tensor = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![21]), vec![1, 1])
            .expect("typed scalar length");
        assert_eq!(
            parse_motion_length(Some(&Value::Tensor(tensor))).unwrap(),
            21
        );

        let negative = Tensor::new_integer(runmat_value::IntegerStorage::I16(vec![-1]), vec![1, 1])
            .expect("negative");
        assert!(parse_motion_length(Some(&Value::Tensor(negative))).is_err());

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(parse_motion_length(Some(&Value::Num(boundary))).is_err());
    }

    #[test]
    fn fspecial_generators_reject_overflowing_dimensions_before_allocation() {
        assert!(generate_gaussian(usize::MAX, 2, 0.5).is_err());
        assert!(generate_log(usize::MAX, 2, 0.5).is_err());
        assert!(generate_motion(usize::MAX, usize::MAX, 0.0, 8).is_err());
    }

    #[test]
    fn fspecial_disk_rejects_unrepresentable_radius_before_size_cast() {
        let huge_radius = (isize::MAX as f64) / 2.0 + 1.0;
        assert!(parse_disk_params(Some(&Value::Num(huge_radius))).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_average_rejects_zero_size() {
        let args = vec![Value::from(0.0)];
        let err = block_on(fspecial_builtin(Value::from("average"), args))
            .expect_err("fspecial should error");
        assert!(error_message(err).contains("positive"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_gaussian_default_matches_reference() {
        let tensor = fspecial_host_tensor("gaussian", Vec::new());
        assert_eq!(tensor.shape, vec![3, 3]);
        const EXPECTED: [f64; 9] = [
            0.011_343_736_558_495,
            0.083_819_505_802_211,
            0.011_343_736_558_495,
            0.083_819_505_802_211,
            0.619_347_030_557_177,
            0.083_819_505_802_211,
            0.011_343_736_558_495,
            0.083_819_505_802_211,
            0.011_343_736_558_495,
        ];
        for (idx, value) in tensor.materialize_f64().iter().enumerate() {
            assert_close(*value, EXPECTED[idx], 1e-12);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_gaussian_size_sigma() {
        let args = vec![Value::from(7.0), Value::from(2.0)];
        let tensor = fspecial_host_tensor("gaussian", args);
        assert_eq!(tensor.shape, vec![7, 7]);
        let center = tensor.rows / 2;
        let col = center;
        let idx = col * tensor.rows + center;
        assert!(tensor.materialize_f64()[idx] > 0.0);
        let sum: f64 = tensor.materialize_f64().iter().sum();
        assert_close(sum, 1.0, 1e-5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_laplacian_alpha() {
        let args = vec![Value::from(0.2)];
        let t = fspecial_host_tensor("laplacian", args);
        assert_eq!(t.shape, vec![3, 3]);
        let expected = [
            0.16666666666666669,
            0.6666666666666667,
            0.16666666666666669,
            0.6666666666666667,
            -3.3333333333333335,
            0.6666666666666667,
            0.16666666666666669,
            0.6666666666666667,
            0.16666666666666669,
        ];
        for (idx, value) in t.materialize_f64().iter().enumerate() {
            assert_close(*value, expected[idx], 1e-7);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_unsharp_default() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let t = fspecial_host_tensor("unsharp", Vec::new());
        assert_eq!(t.shape, vec![3, 3]);
        let sum: f64 = t.materialize_f64().iter().sum();
        assert_close(sum, 1.0, 1e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_log_basic_properties() {
        let t = fspecial_host_tensor("log", vec![Value::from(5.0), Value::from(0.5)]);
        assert_eq!(t.shape, vec![5, 5]);
        let sum: f64 = t.materialize_f64().iter().sum();
        assert_close(sum, 0.0, 1e-12);
        let center = t.rows / 2;
        let idx = center * t.rows + center;
        assert!(t.materialize_f64()[idx] < 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_disk_sum_is_one() {
        let t = fspecial_host_tensor("disk", vec![Value::from(5.0)]);
        assert_eq!(t.shape, vec![11, 11]);
        let sum: f64 = t.materialize_f64().iter().sum();
        assert_close(sum, 1.0, 1e-10);
        let idx = t.rows * (t.cols / 2) + t.rows / 2;
        assert!(t.materialize_f64()[idx] > 0.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_disk_negative_radius_errors() {
        let err = block_on(fspecial_builtin(
            Value::from("disk"),
            vec![Value::from(-1.0)],
        ))
        .expect_err("fspecial should error");
        let message = err.message().to_string();
        assert!(message.contains("non-negative"));
        assert_eq!(err.identifier(), FSPECIAL_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_motion_sum_is_one() {
        let t = fspecial_host_tensor("motion", vec![Value::from(15.0), Value::from(45.0)]);
        assert_eq!(t.shape, vec![15, 15]);
        let sum: f64 = t.materialize_f64().iter().sum();
        assert_close(sum, 1.0, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fspecial_invalid_filter_name() {
        let err = block_on(fspecial_builtin(Value::from("notafilter"), Vec::new()))
            .expect_err("fspecial should error");
        let message = err.message().to_string();
        assert!(message.contains("not supported"));
        assert_eq!(err.identifier(), FSPECIAL_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn fspecial_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = FSPECIAL_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "H = fspecial(type)",
                "H = fspecial(type, arg1)",
                "H = fspecial(type, arg1, arg2)"
            ]
        );
    }

    #[test]
    fn fspecial_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = FSPECIAL_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec![
                "RM.FSPECIAL.INVALID_ARGUMENT",
                "RM.FSPECIAL.INVALID_INPUT",
                "RM.FSPECIAL.INTERNAL",
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn fspecial_gaussian_gpu_matches_cpu() {
        let _env_guard = fspecial_env_guard();
        let _restore = fspecial_device_env_restore();
        std::env::set_var("RUNMAT_ACCEL_FSPECIAL_DEVICE", "1");
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let spec = build_filter_spec(&Value::from("gaussian"), &[]).unwrap();
        let tensor = spec.generate_tensor().unwrap();
        let gpu_tensor = match finalize_output_with_gpu_policy(&spec, tensor, true).unwrap() {
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather gpu result")
            }
            Value::Tensor(t) => t,
            other => panic!("unexpected result {other:?}"),
        };
        let host_tensor =
            match block_on(fspecial_builtin(Value::from("gaussian"), Vec::new())).unwrap() {
                Value::Tensor(t) => t,
                Value::GpuTensor(handle) => {
                    test_support::gather(Value::GpuTensor(handle)).expect("gather fallback")
                }
                other => panic!("unexpected result {other:?}"),
            };
        assert_eq!(gpu_tensor.shape, host_tensor.shape);
        for (a, b) in gpu_tensor
            .materialize_f64()
            .iter()
            .zip(host_tensor.materialize_f64().iter())
        {
            assert_close(*a, *b, 1e-6);
        }
    }
}
