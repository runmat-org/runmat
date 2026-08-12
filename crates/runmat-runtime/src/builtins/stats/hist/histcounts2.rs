//! MATLAB-compatible `histcounts2` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::stats::type_resolvers::histcounts2_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "histcounts2";
const DEFAULT_BIN_COUNT: usize = 10;
const RANGE_EPS: f64 = 1.0e-12;
const HISTCOUNTS2_OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D histogram bin counts.",
}];

const HISTCOUNTS2_OUTPUT_NX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "2-D histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "Xedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin edges.",
    },
];

const HISTCOUNTS2_OUTPUT_NXY: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "2-D histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "Xedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin edges.",
    },
    BuiltinParamDescriptor {
        name: "Yedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin edges.",
    },
];

const HISTCOUNTS2_OUTPUT_ALL: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "2-D histogram bin counts.",
    },
    BuiltinParamDescriptor {
        name: "Xedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis bin edges.",
    },
    BuiltinParamDescriptor {
        name: "Yedges",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis bin edges.",
    },
    BuiltinParamDescriptor {
        name: "binX",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based X-bin indices, or zero for unbinned observations.",
    },
    BuiltinParamDescriptor {
        name: "binY",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based Y-bin indices, or zero for unbinned observations.",
    },
];

const HISTCOUNTS2_INPUTS_XY: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
];

const HISTCOUNTS2_INPUTS_XY_BINS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "bins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "NumBins scalar/vector or positional edge specification.",
    },
];

const HISTCOUNTS2_INPUTS_XY_BINS_BINSY: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "binsX",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis NumBins scalar or edge vector.",
    },
    BuiltinParamDescriptor {
        name: "binsY",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis NumBins scalar or edge vector.",
    },
];

const HISTCOUNTS2_INPUTS_XY_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "X-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Y-axis input data.",
    },
    BuiltinParamDescriptor {
        name: "name_value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/value pairs controlling binning and normalization.",
    },
];

const HISTCOUNTS2_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y, bins)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y, binsX, binsY)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS_BINSY,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = histcounts2(X, Y, Name, Value, ...)",
        inputs: &HISTCOUNTS2_INPUTS_XY_NAMEVALUE,
        outputs: &HISTCOUNTS2_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges] = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_NX,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges] = histcounts2(X, Y, bins)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS,
        outputs: &HISTCOUNTS2_OUTPUT_NX,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges] = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_NXY,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges] = histcounts2(X, Y, binsX, binsY)",
        inputs: &HISTCOUNTS2_INPUTS_XY_BINS_BINSY,
        outputs: &HISTCOUNTS2_OUTPUT_NXY,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges] = histcounts2(X, Y, Name, Value, ...)",
        inputs: &HISTCOUNTS2_INPUTS_XY_NAMEVALUE,
        outputs: &HISTCOUNTS2_OUTPUT_NXY,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges, binX, binY] = histcounts2(X, Y)",
        inputs: &HISTCOUNTS2_INPUTS_XY,
        outputs: &HISTCOUNTS2_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[N, Xedges, Yedges, binX, binY] = histcounts2(X, Y, Name, Value, ...)",
        inputs: &HISTCOUNTS2_INPUTS_XY_NAMEVALUE,
        outputs: &HISTCOUNTS2_OUTPUT_ALL,
    },
];

const HISTCOUNTS2_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.INVALID_ARGUMENT",
    identifier: Some("RunMat:histcounts2:InvalidArgument"),
    when: "Arguments are malformed, inconsistent, or unsupported.",
    message: "histcounts2: invalid argument",
};

const HISTCOUNTS2_ERROR_NUMBINS_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.NUMBINS_INVALID",
    identifier: Some("RunMat:histcounts2:NumBinsInvalid"),
    when: "NumBins is zero, negative, non-finite, or non-integer.",
    message: "histcounts2: NumBins must be a positive finite scalar",
};

const HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.BINMETHOD_CONFLICT",
    identifier: Some("RunMat:histcounts2:BinMethodConflict"),
    when: "BinMethod is combined with incompatible bin controls.",
    message: "histcounts2: BinMethod cannot be combined with BinEdges, NumBins, or BinWidth",
};

const HISTCOUNTS2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTCOUNTS2.INTERNAL",
    identifier: Some("RunMat:histcounts2:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "histcounts2: internal operation failed",
};

const HISTCOUNTS2_ERRORS: [BuiltinErrorDescriptor; 4] = [
    HISTCOUNTS2_ERROR_INVALID_ARGUMENT,
    HISTCOUNTS2_ERROR_NUMBINS_INVALID,
    HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT,
    HISTCOUNTS2_ERROR_INTERNAL,
];

pub const HISTCOUNTS2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HISTCOUNTS2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HISTCOUNTS2_ERRORS,
};

const HISTCOUNTS2_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "histcounts2-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "histcounts2 with gpuArray inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Histcounts2GpuInputExtension"),
};
const HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "histcounts2-extra-bin-syntax",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "histcounts2 axis-specific or scalar-width convenience syntax is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Histcounts2ExtraBinSyntaxExtension"),
};
pub const HISTCOUNTS2_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    HISTCOUNTS2_GPU_EXTENSION,
    HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
];
const HISTCOUNTS2_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X_and_Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented; X and Y may use different classes but must have identical sizes.",
    },
    BuiltinIntegerInputCapability {
        name: "bin_controls",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed NumBins and explicit integer edges follow their scalar/vector contracts; same-class wide explicit edges are classified exactly.",
    },
];
pub const HISTCOUNTS2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[N, Xedges, Yedges, binX, binY] = histcounts2(integer_X, integer_Y, ...)",
        inputs: &HISTCOUNTS2_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Counts, returned edges, and bin indices are double; source-domain exact comparisons are retained for same-class explicit integer edges before that output boundary.",
    }];

fn builtin_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn descriptor_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with(error, error.message)
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::hist::histcounts2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "histcounts2",
    op_kind: GpuOpKind::Custom("histcounts2"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Omit,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Current builds gather documented host inputs and materialise host-resident outputs; no provider-native hook is registered.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::hist::histcounts2")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "histcounts2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Histogram binning terminates fusion chains and materialises host-resident outputs.",
};

#[runtime_builtin(
    name = "histcounts2",
    category = "stats/hist",
    summary = "Count paired observations in 2-D histogram bins.",
    keywords = "histcounts2,2d histogram,joint distribution,binning,probability,gpu",
    accel = "reduction",
    sink = true,
    type_resolver(histcounts2_type),
    descriptor(crate::builtins::stats::hist::histcounts2::HISTCOUNTS2_DESCRIPTOR),
    extensions(crate::builtins::stats::hist::histcounts2::HISTCOUNTS2_EXTENSIONS),
    integer_capabilities(
        crate::builtins::stats::hist::histcounts2::HISTCOUNTS2_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::stats::hist::histcounts2"
)]
async fn histcounts2_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    parse_options(&rest)?;
    ensure_histcounts2_public_syntax(&rest)?;
    let eval = evaluate(x, y, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_counts_value()]));
        }
        if out_count == 2 {
            let (counts, x_edges) = eval.into_pair();
            return Ok(Value::OutputList(vec![counts, x_edges]));
        }
        if out_count > 5 {
            return Err(builtin_error("histcounts2: too many output arguments"));
        }
        let (counts, x_edges, y_edges, x_bins, y_bins) = eval.into_all();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![counts, x_edges, y_edges, x_bins, y_bins],
        ));
    }
    Ok(eval.into_counts_value())
}

pub(crate) fn ensure_histcounts2_public_syntax(args: &[Value]) -> BuiltinResult<()> {
    let mut extension = false;
    if args.len() >= 2 && !is_option_key(&args[0]) && !is_option_key(&args[1]) {
        let first = numeric_vector(&args[0], NAME, "positional X argument")?;
        let second = numeric_vector(&args[1], NAME, "positional Y argument")?;
        extension |= first.len() == 1 && second.len() == 1;
    }
    let mut index = if args.first().is_some_and(|value| !is_option_key(value)) {
        if args.get(1).is_some_and(|value| !is_option_key(value)) {
            2
        } else {
            1
        }
    } else {
        0
    };
    while index + 1 < args.len() {
        let key = tensor::value_to_string(&args[index])
            .unwrap_or_default()
            .to_ascii_lowercase();
        let value = &args[index + 1];
        extension |= matches!(
            key.as_str(),
            "xbinwidth" | "ybinwidth" | "binlimits" | "xbinmethod" | "ybinmethod"
        );
        if key == "binwidth" {
            extension |= numeric_vector(value, NAME, "BinWidth")?.len() == 1;
        }
        if key == "binmethod" {
            extension |= tensor::value_to_string(value).is_some_and(|method| {
                matches!(
                    method.trim().to_ascii_lowercase().as_str(),
                    "sturges" | "sqrt"
                )
            });
        }
        index += 2;
    }
    if extension {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

/// Evaluate `histcounts2` once and surface all outputs.
pub async fn evaluate(x: Value, y: Value, rest: &[Value]) -> BuiltinResult<Histcounts2Evaluation> {
    let options = parse_options(rest)?;
    if matches!(&x, Value::GpuTensor(_)) || matches!(&y, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&HISTCOUNTS2_GPU_EXTENSION, NAME)?;
    }
    let x_tensor = match x {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        other => tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?,
    };
    let y_tensor = match y {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        other => tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?,
    };
    histcounts2_from_tensors(x_tensor, y_tensor, &options)
}

fn histcounts2_from_tensors(
    x_tensor: Tensor,
    y_tensor: Tensor,
    options: &Histcounts2Options,
) -> BuiltinResult<Histcounts2Evaluation> {
    if x_tensor.len() != y_tensor.len() {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must contain the same number of elements"
        )));
    }
    if x_tensor.shape != y_tensor.shape {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must be the same size"
        )));
    }

    // Histogram statistics and returned edges are defined in the floating
    // computation domain even when observations arrive in integer storage.
    let x_shape = x_tensor.shape.clone();
    let y_shape = y_tensor.shape.clone();
    let x_exact = x_tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let y_exact = y_tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let x_values = tensor::tensor_into_values_f64(x_tensor);
    let y_values = tensor::tensor_into_values_f64(y_tensor);
    let x_axis = collect_axis_data(&x_values);
    let y_axis = collect_axis_data(&y_values);

    let x_edges = compute_edges_for_axis(&x_axis, &options.x, "X")?;
    let y_edges = compute_edges_for_axis(&y_axis, &options.y, "Y")?;

    let x_bin_indices = compute_axis_bin_indices(
        &x_values,
        x_exact.as_deref(),
        &x_edges,
        options.x.exact_edges.as_deref(),
    );
    let y_bin_indices = compute_axis_bin_indices(
        &y_values,
        y_exact.as_deref(),
        &y_edges,
        options.y.exact_edges.as_deref(),
    );
    let counts = compute_histogram_counts(
        &x_bin_indices,
        &y_bin_indices,
        x_edges.len() - 1,
        y_edges.len() - 1,
    );
    let normalised = apply_normalization_2d(
        &counts,
        &x_edges,
        &y_edges,
        options.normalization,
        x_values.len(),
    );

    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let counts_tensor = Tensor::new(normalised, vec![x_bins, y_bins])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let x_edges_tensor = Tensor::new(x_edges.clone(), vec![1, x_edges.len()])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let y_edges_tensor = Tensor::new(y_edges.clone(), vec![1, y_edges.len()])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let x_bins_tensor =
        Tensor::new(x_bin_indices, x_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let y_bins_tensor =
        Tensor::new(y_bin_indices, y_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;

    Ok(Histcounts2Evaluation::new(
        counts_tensor,
        x_edges_tensor,
        y_edges_tensor,
        x_bins_tensor,
        y_bins_tensor,
    ))
}

fn compute_histogram_counts(
    x_indices: &[f64],
    y_indices: &[f64],
    x_bins: usize,
    y_bins: usize,
) -> Vec<f64> {
    let mut counts = vec![0.0f64; x_bins * y_bins];
    for (&x, &y) in x_indices.iter().zip(y_indices) {
        if x > 0.0 && y > 0.0 {
            let ix = x as usize - 1;
            let iy = y as usize - 1;
            let idx = ix + iy * x_bins;
            if idx < counts.len() {
                counts[idx] += 1.0;
            }
        }
    }
    counts
}

fn compute_axis_bin_indices(
    values: &[f64],
    exact_values: Option<&[IntValue]>,
    edges: &[f64],
    exact_edges: Option<&[IntValue]>,
) -> Vec<f64> {
    if let (Some(values), Some(edges_exact)) = (exact_values, exact_edges) {
        if values
            .first()
            .zip(edges_exact.first())
            .is_some_and(|(value, edge)| {
                std::mem::discriminant(value) == std::mem::discriminant(edge)
            })
        {
            return values
                .iter()
                .map(|value| {
                    exact_integer_bin_index(value, edges_exact).map_or(0.0, |i| (i + 1) as f64)
                })
                .collect();
        }
    }
    values
        .iter()
        .map(|&value| find_bin_index(value, edges).map_or(0.0, |i| (i + 1) as f64))
        .collect()
}

fn exact_integer_bin_index(value: &IntValue, edges: &[IntValue]) -> Option<usize> {
    if edges.len() < 2
        || exact_int_cmp(value, &edges[0])? == Ordering::Less
        || exact_int_cmp(value, edges.last()?)? == Ordering::Greater
    {
        return None;
    }
    if value == edges.last()? {
        return Some(edges.len() - 2);
    }
    (0..edges.len() - 1).find(|&index| {
        exact_int_cmp(value, &edges[index]).is_some_and(|order| order != Ordering::Less)
            && exact_int_cmp(value, &edges[index + 1]) == Some(Ordering::Less)
    })
}

fn exact_int_cmp(left: &IntValue, right: &IntValue) -> Option<Ordering> {
    macro_rules! same_variant_cmp {
        ($($variant:ident),+ $(,)?) => {
            match (left, right) {
                $((IntValue::$variant(a), IntValue::$variant(b)) => Some(a.cmp(b)),)+
                _ => None,
            }
        };
    }
    same_variant_cmp!(I8, I16, I32, I64, U8, U16, U32, U64)
}

#[derive(Clone)]
struct AxisData {
    values: Vec<f64>,
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
}

fn collect_axis_data(values: &[f64]) -> AxisData {
    let mut filtered = Vec::new();
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;

    for &sample in values {
        if sample.is_nan() {
            continue;
        }
        filtered.push(sample);
        if sample.is_finite() {
            min_val = Some(match min_val {
                Some(current) if sample >= current => current,
                Some(_) => sample,
                None => sample,
            });
            max_val = Some(match max_val {
                Some(current) if sample <= current => current,
                Some(_) => sample,
                None => sample,
            });
        }
    }

    let original_range_zero = match (min_val, max_val) {
        (Some(minimum), Some(maximum)) => approx_equal(minimum, maximum),
        _ => false,
    };

    AxisData {
        values: filtered,
        min_val,
        max_val,
        original_range_zero,
    }
}

fn compute_edges_for_axis(
    data: &AxisData,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    if let Some(edges) = &options.explicit_edges {
        if options.exact_edges.is_none() {
            validate_edges(edges, axis)?;
        }
        return Ok(edges.clone());
    }

    if let Some(method) = options.bin_method {
        return compute_edges_with_method(
            &data.values,
            data.min_val,
            data.max_val,
            data.original_range_zero,
            method,
            options,
            axis,
        );
    }

    compute_edges_standard(
        data.min_val,
        data.max_val,
        data.original_range_zero,
        options,
        axis,
    )
}

fn compute_edges_standard(
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: data range for {axis} must be finite; specify {axis}BinLimits or {axis}BinEdges"
        )));
    }

    if upper < lower {
        return Err(builtin_error(format!(
            "{NAME}: {axis} bin limits must be increasing"
        )));
    }

    if options.bin_limits.is_some() && approx_equal(lower, upper) {
        return Err(builtin_error(format!(
            "{NAME}: {axis}BinLimits must specify a non-zero width"
        )));
    }

    if let Some(width) = options.bin_width {
        if !width.is_finite() || width <= 0.0 {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinWidth must be a positive finite scalar"
            )));
        }

        if original_range_zero && options.bin_limits.is_none() {
            let centre = min_val.unwrap_or(0.0);
            lower = centre - width / 2.0;
            upper = centre + width / 2.0;
        } else if options.bin_limits.is_none() {
            if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
                lower = minimum;
                upper = maximum;
            }
        }

        if approx_equal(lower, upper) {
            upper = lower + width;
        }

        let bins = ((upper - lower) / width).ceil().max(1.0) as usize;
        let mut edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            edges.push(lower + width * i as f64);
        }
        if let Some(last) = edges.last_mut() {
            *last = upper;
        }
        validate_edges(&edges, axis)?;
        return Ok(edges);
    }

    let mut num_bins = options.num_bins.unwrap_or(DEFAULT_BIN_COUNT);
    if num_bins == 0 {
        return Err(builtin_error(format!(
            "{NAME}: NumBins must be a positive integer"
        )));
    }

    if original_range_zero {
        let centre = min_val.unwrap_or(0.0);
        if options.num_bins.is_none() && options.bin_limits.is_none() && options.bin_width.is_none()
        {
            num_bins = 1;
        }
        if options.bin_limits.is_none() && options.bin_width.is_none() {
            lower = centre - 0.5;
            upper = centre + 0.5;
        }
    } else if options.bin_limits.is_none() {
        if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
            lower = lower.min(minimum);
            upper = upper.max(maximum);
        }
    }

    if approx_equal(lower, upper) {
        upper = lower + 1.0;
    }

    let edges = linspace(lower, upper, num_bins + 1);
    validate_edges(&edges, axis)?;
    Ok(edges)
}

fn compute_edges_with_method(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    method: BinMethod,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    if values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    if matches!(method, BinMethod::Integers) {
        let edges = compute_integer_edges(min_val, max_val, options, axis)?;
        validate_edges(&edges, axis)?;
        return Ok(edges);
    }

    let (lower, upper) = derive_initial_limits(min_val, max_val, options.bin_limits);
    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: {axis} data range must be finite for BinMethod"
        )));
    }

    if approx_equal(lower, upper) {
        if options.bin_limits.is_some() {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinLimits must specify a non-zero width"
            )));
        }
        return compute_edges_standard(min_val, max_val, true, options, axis);
    }

    let finite_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    let range = upper - lower;
    if range <= 0.0 {
        return compute_edges_standard(min_val, max_val, true, options, axis);
    }

    let mut sorted = finite_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let width = match method {
        BinMethod::Auto => {
            let fd = freedman_diaconis_width(&sorted);
            if fd > 0.0 {
                fd
            } else {
                let scott = scott_width(&finite_values);
                if scott > 0.0 {
                    scott
                } else {
                    range / sturges_bin_count(sorted.len()) as f64
                }
            }
        }
        BinMethod::Scott => scott_width(&finite_values),
        BinMethod::Fd => freedman_diaconis_width(&sorted),
        BinMethod::Sturges => range / sturges_bin_count(sorted.len()) as f64,
        BinMethod::Sqrt => range / sqrt_bin_count(sorted.len()) as f64,
        BinMethod::Integers => unreachable!(),
    };

    if !width.is_finite() || width <= 0.0 {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    let bins = ((range / width).ceil().max(1.0)) as usize;
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    validate_edges(&edges, axis)?;
    Ok(edges)
}

fn compute_integer_edges(
    min_val: Option<f64>,
    max_val: Option<f64>,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: {axis}BinLimits must be finite for 'integers' BinMethod"
        )));
    }

    if approx_equal(lower, upper) {
        let centre = min_val.or(max_val).unwrap_or(lower);
        lower = centre - 0.5;
        upper = centre + 0.5;
    }

    if let Some((lo, hi)) = options.bin_limits {
        lower = lo;
        upper = hi;
    } else {
        lower = lower.floor();
        upper = upper.ceil();
    }

    if upper <= lower {
        upper = lower + 1.0;
    }

    let mut edges = Vec::new();
    edges.push(lower);

    if let Some((lo, hi)) = options.bin_limits {
        let mut k = lo.ceil() as i64;
        while (k as f64) < hi {
            let candidate = k as f64;
            if candidate > lo + RANGE_EPS {
                edges.push(candidate);
            }
            k += 1;
        }
    } else {
        let mut current = lower.floor() as i64 + 1;
        let end = upper.ceil() as i64;
        while current < end {
            edges.push(current as f64);
            current += 1;
        }
    }

    if !approx_equal(*edges.last().unwrap(), upper) {
        edges.push(upper);
    }

    if edges.len() < 2 {
        edges.push(upper + 1.0);
    }

    Ok(edges)
}

fn derive_initial_limits(
    min_val: Option<f64>,
    max_val: Option<f64>,
    bin_limits: Option<(f64, f64)>,
) -> (f64, f64) {
    if let Some((lo, hi)) = bin_limits {
        (lo, hi)
    } else if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
        (minimum, maximum)
    } else {
        (0.0, 1.0)
    }
}

fn find_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    match edges.binary_search_by(|edge| edge.partial_cmp(&value).unwrap_or(Ordering::Less)) {
        Ok(index) => {
            if index == 0 {
                Some(0)
            } else if index < edges.len() - 1 {
                Some(index)
            } else {
                Some(edges.len() - 2)
            }
        }
        Err(index) => {
            if index == 0 || index > edges.len() - 1 {
                None
            } else {
                Some(index - 1)
            }
        }
    }
}

fn apply_normalization_2d(
    counts: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    mode: HistogramNormalization,
    observation_count: usize,
) -> Vec<f64> {
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let total = observation_count as f64;
    let x_widths: Vec<f64> = x_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let y_widths: Vec<f64> = y_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();

    match mode {
        HistogramNormalization::Count => counts.to_vec(),
        HistogramNormalization::Probability => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::Percentage => {
            if total > 0.0 {
                counts.iter().map(|&c| 100.0 * c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CountDensity => {
            let mut out = vec![0.0; counts.len()];
            for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                    let idx = ix + iy * x_bins;
                    let area = x_width * y_width;
                    out[idx] = if area > 0.0 { counts[idx] / area } else { 0.0 };
                }
            }
            out
        }
        HistogramNormalization::Pdf => {
            if total > 0.0 {
                let mut out = vec![0.0; counts.len()];
                for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                    for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                        let idx = ix + iy * x_bins;
                        let area = x_width * y_width;
                        out[idx] = if area > 0.0 {
                            counts[idx] / (total * area)
                        } else {
                            0.0
                        };
                    }
                }
                out
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CumCount => cumulative_2d(counts, x_bins, y_bins, None),
        HistogramNormalization::Cdf => {
            if total > 0.0 {
                cumulative_2d(counts, x_bins, y_bins, Some(total))
            } else {
                vec![0.0; counts.len()]
            }
        }
    }
}

fn cumulative_2d(counts: &[f64], x_bins: usize, y_bins: usize, divisor: Option<f64>) -> Vec<f64> {
    let mut out = vec![0.0; counts.len()];
    for y in 0..y_bins {
        for x in 0..x_bins {
            let idx = x + y * x_bins;
            let left = (x > 0).then(|| out[idx - 1]).unwrap_or(0.0);
            let below = (y > 0).then(|| out[idx - x_bins]).unwrap_or(0.0);
            let diagonal = (x > 0 && y > 0)
                .then(|| out[idx - x_bins - 1])
                .unwrap_or(0.0);
            out[idx] = counts[idx] + left + below - diagonal;
        }
    }
    if let Some(divisor) = divisor {
        for value in &mut out {
            *value /= divisor;
        }
    }
    out
}

fn validate_edges(edges: &[f64], axis: &str) -> BuiltinResult<()> {
    if edges.len() < 2 {
        return Err(builtin_error(format!(
            "{NAME}: {axis}BinEdges must contain at least two elements"
        )));
    }
    for pair in edges.windows(2) {
        if pair[0].is_nan() || pair[1].is_nan() {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must contain finite numbers"
            )));
        }
        if pair[1] <= pair[0] {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must be strictly increasing"
            )));
        }
    }
    Ok(())
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(start + step * i as f64);
    }
    if let Some(last) = out.last_mut() {
        *last = stop;
    }
    out
}

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= RANGE_EPS * (a.abs().max(b.abs()).max(1.0))
}

fn scott_width(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let sigma = standard_deviation(values);
    if sigma <= 0.0 {
        return 0.0;
    }
    3.5 * sigma / (n as f64).powf(1.0 / 3.0)
}

fn freedman_diaconis_width(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 2 {
        return 0.0;
    }
    let q1 = percentile(sorted, 0.25);
    let q3 = percentile(sorted, 0.75);
    let iqr = q3 - q1;
    if iqr <= 0.0 {
        return 0.0;
    }
    2.0 * iqr / (n as f64).powf(1.0 / 3.0)
}

fn sturges_bin_count(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        ((n as f64).log2() + 1.0).ceil().max(1.0) as usize
    }
}

fn sqrt_bin_count(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).sqrt().ceil().max(1.0) as usize
    }
}

fn standard_deviation(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut acc = 0.0;
    for &value in values {
        let diff = value - mean;
        acc += diff * diff;
    }
    (acc / (n as f64 - 1.0).max(1.0)).sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let clamped = p.clamp(0.0, 1.0);
    let rank = clamped * (n as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

#[derive(Clone, Default)]
struct Histcounts2Options {
    x: AxisOptions,
    y: AxisOptions,
    normalization: HistogramNormalization,
}

impl Histcounts2Options {
    fn validate(&self) -> BuiltinResult<()> {
        self.x.validate("X")?;
        self.y.validate("Y")?;
        Ok(())
    }
}

#[derive(Clone, Default)]
struct AxisOptions {
    explicit_edges: Option<Vec<f64>>,
    exact_edges: Option<Vec<IntValue>>,
    num_bins: Option<usize>,
    bin_width: Option<f64>,
    bin_limits: Option<(f64, f64)>,
    bin_method: Option<BinMethod>,
}

impl AxisOptions {
    fn validate(&self, axis: &str) -> BuiltinResult<()> {
        if self.explicit_edges.is_some()
            && (self.num_bins.is_some() || self.bin_width.is_some() || self.bin_limits.is_some())
        {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges cannot be combined with NumBins, {axis}BinWidth, or {axis}BinLimits"
            )));
        }
        if self.bin_method.is_some()
            && (self.explicit_edges.is_some()
                || self.bin_width.is_some()
                || self.num_bins.is_some())
        {
            return Err(descriptor_error(&HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT));
        }
        if self.num_bins.is_some() && self.bin_width.is_some() {
            return Err(builtin_error(format!(
                "{NAME}: specify only one of NumBins or {axis}BinWidth"
            )));
        }
        if let Some((lo, hi)) = self.bin_limits {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(builtin_error(format!(
                    "{NAME}: {axis}BinLimits must be finite"
                )));
            }
            if hi < lo {
                return Err(builtin_error(format!(
                    "{NAME}: {axis}BinLimits must be increasing"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinMethod {
    Auto,
    Scott,
    Fd,
    Sturges,
    Sqrt,
    Integers,
}

#[derive(Clone, Copy, Debug, Default)]
enum HistogramNormalization {
    #[default]
    Count,
    Probability,
    Percentage,
    CountDensity,
    Pdf,
    CumCount,
    Cdf,
}

#[derive(Clone)]
pub struct Histcounts2Evaluation {
    counts: Tensor,
    x_edges: Tensor,
    y_edges: Tensor,
    x_bins: Tensor,
    y_bins: Tensor,
}

impl Histcounts2Evaluation {
    fn new(
        counts: Tensor,
        x_edges: Tensor,
        y_edges: Tensor,
        x_bins: Tensor,
        y_bins: Tensor,
    ) -> Self {
        Self {
            counts,
            x_edges,
            y_edges,
            x_bins,
            y_bins,
        }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        (Value::Tensor(counts), Value::Tensor(x_edges))
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        let y_edges = self.y_edges;
        (
            Value::Tensor(counts),
            Value::Tensor(x_edges),
            Value::Tensor(y_edges),
        )
    }

    pub fn into_all(self) -> (Value, Value, Value, Value, Value) {
        (
            Value::Tensor(self.counts),
            Value::Tensor(self.x_edges),
            Value::Tensor(self.y_edges),
            Value::Tensor(self.x_bins),
            Value::Tensor(self.y_bins),
        )
    }
}

fn parse_options(args: &[Value]) -> BuiltinResult<Histcounts2Options> {
    let mut options = Histcounts2Options::default();
    let mut index = 0;

    if index < args.len() && !is_option_key(&args[index]) {
        if index + 1 < args.len() && !is_option_key(&args[index + 1]) {
            let x_vec = numeric_vector(&args[index], NAME, "positional X argument")?;
            let y_vec = numeric_vector(&args[index + 1], NAME, "positional Y argument")?;
            if x_vec.len() >= 2 && y_vec.len() >= 2 {
                validate_edge_value(&args[index], &x_vec, "X")?;
                validate_edge_value(&args[index + 1], &y_vec, "Y")?;
                options.x.explicit_edges = Some(x_vec);
                options.y.explicit_edges = Some(y_vec);
                options.x.exact_edges = exact_integer_vector(&args[index]);
                options.y.exact_edges = exact_integer_vector(&args[index + 1]);
            } else if x_vec.len() == 1 && y_vec.len() == 1 {
                let nx = positive_usize(&args[index], NAME, "NumBins")?;
                let ny = positive_usize(&args[index + 1], NAME, "NumBins")?;
                options.x.num_bins = Some(nx);
                options.y.num_bins = Some(ny);
            } else {
                return Err(builtin_error(format!(
                    "{NAME}: positional bin arguments must either specify two edge vectors or two scalar bin counts"
                )));
            }
            index += 2;
        } else {
            let bins = num_bins_vector(&args[index])?;
            match bins.as_slice() {
                [n] => {
                    options.x.num_bins = Some(*n);
                    options.y.num_bins = Some(*n);
                }
                [nx, ny] => {
                    options.x.num_bins = Some(*nx);
                    options.y.num_bins = Some(*ny);
                }
                _ => {
                    return Err(builtin_error(format!(
                        "{NAME}: NumBins must be a scalar or two-element vector"
                    )))
                }
            }
            index += 1;
        }
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| builtin_error(format!("{NAME}: expected name/value pair arguments")))?;
        index += 1;
        if index >= args.len() {
            return Err(builtin_error(format!(
                "{NAME}: missing value for option '{key}'"
            )));
        }
        let value = &args[index];
        index += 1;
        let lowered = key.trim().to_ascii_lowercase();

        match lowered.as_str() {
            "numbins" => {
                let bins = num_bins_vector(value)?;
                match bins.as_slice() {
                    [n] => {
                        options.x.num_bins = Some(*n);
                        options.y.num_bins = Some(*n);
                    }
                    [nx, ny] => {
                        options.x.num_bins = Some(*nx);
                        options.y.num_bins = Some(*ny);
                    }
                    _ => {
                        return Err(builtin_error(format!(
                            "{NAME}: NumBins must be a scalar or two-element vector"
                        )))
                    }
                }
            }
            "xbinwidth" => {
                let width = positive_scalar(value, NAME, "XBinWidth")?;
                options.x.bin_width = Some(width);
            }
            "ybinwidth" => {
                let width = positive_scalar(value, NAME, "YBinWidth")?;
                options.y.bin_width = Some(width);
            }
            "binwidth" => {
                let widths = numeric_vector(value, NAME, "BinWidth")?;
                match widths.len() {
                    1 => {
                        let width = positive_scalar_from_f64(widths[0], "BinWidth")?;
                        options.x.bin_width = Some(width);
                        options.y.bin_width = Some(width);
                    }
                    2 => {
                        let wx = positive_scalar_from_f64(widths[0], "BinWidth")?;
                        let wy = positive_scalar_from_f64(widths[1], "BinWidth")?;
                        options.x.bin_width = Some(wx);
                        options.y.bin_width = Some(wy);
                    }
                    _ => {
                        return Err(builtin_error(format!(
                            "{NAME}: BinWidth must be a scalar or two-element vector"
                        )))
                    }
                }
            }
            "xbinlimits" => {
                let limits = numeric_vector(value, NAME, "XBinLimits")?;
                if limits.len() != 2 {
                    return Err(builtin_error(format!(
                        "{NAME}: XBinLimits must contain exactly two elements"
                    )));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(builtin_error(format!(
                        "{NAME}: XBinLimits must be increasing"
                    )));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(builtin_error(format!("{NAME}: XBinLimits must be finite")));
                }
                options.x.bin_limits = Some((lo, hi));
            }
            "ybinlimits" => {
                let limits = numeric_vector(value, NAME, "YBinLimits")?;
                if limits.len() != 2 {
                    return Err(builtin_error(format!(
                        "{NAME}: YBinLimits must contain exactly two elements"
                    )));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(builtin_error(format!(
                        "{NAME}: YBinLimits must be increasing"
                    )));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(builtin_error(format!("{NAME}: YBinLimits must be finite")));
                }
                options.y.bin_limits = Some((lo, hi));
            }
            "binlimits" => {
                let limits = numeric_vector(value, NAME, "BinLimits")?;
                match limits.len() {
                    2 => {
                        let lo = limits[0];
                        let hi = limits[1];
                        if hi < lo {
                            return Err(builtin_error(format!(
                                "{NAME}: BinLimits must be increasing"
                            )));
                        }
                        if !lo.is_finite() || !hi.is_finite() {
                            return Err(builtin_error(format!("{NAME}: BinLimits must be finite")));
                        }
                        options.x.bin_limits = Some((lo, hi));
                        options.y.bin_limits = Some((lo, hi));
                    }
                    4 => {
                        let x_lo = limits[0];
                        let x_hi = limits[1];
                        let y_lo = limits[2];
                        let y_hi = limits[3];
                        if x_hi < x_lo || y_hi < y_lo {
                            return Err(builtin_error(format!(
                                "{NAME}: BinLimits must be increasing"
                            )));
                        }
                        if !x_lo.is_finite()
                            || !x_hi.is_finite()
                            || !y_lo.is_finite()
                            || !y_hi.is_finite()
                        {
                            return Err(builtin_error(format!("{NAME}: BinLimits must be finite")));
                        }
                        options.x.bin_limits = Some((x_lo, x_hi));
                        options.y.bin_limits = Some((y_lo, y_hi));
                    }
                    _ => {
                        return Err(builtin_error(format!(
                            "{NAME}: BinLimits must contain two or four elements"
                        )))
                    }
                }
            }
            "xbinedges" => {
                let edges = numeric_vector(value, NAME, "XBinEdges")?;
                validate_edge_value(value, &edges, "X")?;
                options.x.explicit_edges = Some(edges);
                options.x.exact_edges = exact_integer_vector(value);
            }
            "ybinedges" => {
                let edges = numeric_vector(value, NAME, "YBinEdges")?;
                validate_edge_value(value, &edges, "Y")?;
                options.y.explicit_edges = Some(edges);
                options.y.exact_edges = exact_integer_vector(value);
            }
            "binmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: BinMethod must be a string")))?;
                let method = parse_bin_method(&text)?;
                options.x.bin_method = Some(method);
                options.y.bin_method = Some(method);
            }
            "xbinmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: XBinMethod must be a string")))?;
                options.x.bin_method = Some(parse_bin_method(&text)?);
            }
            "ybinmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: YBinMethod must be a string")))?;
                options.y.bin_method = Some(parse_bin_method(&text)?);
            }
            "normalization" => {
                let text = tensor::value_to_string(value).ok_or_else(|| {
                    builtin_error(format!("{NAME}: Normalization must be a string"))
                })?;
                options.normalization = parse_normalization(&text)?;
            }
            other => {
                return Err(builtin_error(format!(
                    "{NAME}: unrecognised option '{other}'"
                )));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

fn is_option_key(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn numeric_vector(value: &Value, name: &str, option: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{name}: {option} must be numeric")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn exact_integer_vector(value: &Value) -> Option<Vec<IntValue>> {
    match value {
        Value::Int(value) => Some(vec![value.clone()]),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.exact_values()),
        _ => None,
    }
}

fn validate_edge_value(value: &Value, floating: &[f64], axis: &str) -> BuiltinResult<()> {
    if let Some(exact) = exact_integer_vector(value) {
        if exact.len() < 2 {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must contain at least two elements"
            )));
        }
        if exact
            .windows(2)
            .any(|pair| exact_int_cmp(&pair[0], &pair[1]) != Some(Ordering::Less))
        {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must be strictly increasing"
            )));
        }
        Ok(())
    } else {
        validate_edges(floating, axis)
    }
}

fn num_bins_vector(value: &Value) -> BuiltinResult<Vec<usize>> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(vec![integer
            .try_to_usize()
            .filter(valid_num_bins)
            .ok_or_else(|| {
                descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID)
            })?]);
    }
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{NAME}: NumBins must be numeric")))?;
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .exact_values()
            .into_iter()
            .map(|value| {
                value
                    .try_to_usize()
                    .filter(valid_num_bins)
                    .ok_or_else(|| descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID))
            })
            .collect();
    }
    tensor
        .materialize_f64()
        .iter()
        .map(|value| positive_usize_from_f64(*value, "NumBins"))
        .collect()
}

fn positive_usize(value: &Value, name: &str, option: &str) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value.try_to_usize().filter(valid_num_bins).ok_or_else(|| {
            if name == NAME && option == "NumBins" {
                descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID)
            } else {
                builtin_error(format!("{name}: {option} must be a positive finite scalar"))
            }
        });
    }
    let scalar = scalar_value(value, name, option)?;
    positive_usize_from_f64(scalar, option)
}

fn valid_num_bins(value: &usize) -> bool {
    *value > 0 && *value < usize::MAX
}

fn positive_scalar(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
    let scalar = scalar_value(value, name, option)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        return Err(builtin_error(format!(
            "{name}: {option} must be a positive finite scalar"
        )));
    }
    Ok(scalar)
}

fn scalar_value(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(builtin_error(format!("{name}: {option} must be a scalar")));
            }
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(builtin_error(format!("{name}: {option} must be a scalar")));
            }
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(builtin_error(format!(
            "{name}: {option} must be numeric, got {:?}",
            other
        ))),
    }
}

fn positive_usize_from_f64(value: f64, option: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value <= 0.0 {
        if option == "NumBins" {
            return Err(descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID));
        }
        return Err(builtin_error(format!(
            "{NAME}: {option} must be a positive finite scalar"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > f64::EPSILON {
        return Err(builtin_error(format!(
            "{NAME}: {option} must be an integer"
        )));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(builtin_error(format!(
            "{NAME}: {option} is outside the supported range"
        )));
    }
    Ok(rounded as usize)
}

fn positive_scalar_from_f64(value: f64, option: &str) -> BuiltinResult<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(builtin_error(format!(
            "{NAME}: {option} must be a positive finite scalar"
        )));
    }
    Ok(value)
}

fn parse_bin_method(text: &str) -> BuiltinResult<BinMethod> {
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BinMethod::Auto),
        "scott" => Ok(BinMethod::Scott),
        "fd" | "freedmandiaconis" => Ok(BinMethod::Fd),
        "sturges" => Ok(BinMethod::Sturges),
        "sqrt" => Ok(BinMethod::Sqrt),
        "integers" => Ok(BinMethod::Integers),
        other => Err(builtin_error(format!(
            "{NAME}: unrecognised BinMethod value '{other}'"
        ))),
    }
}

fn parse_normalization(text: &str) -> BuiltinResult<HistogramNormalization> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" => Ok(HistogramNormalization::Count),
        "probability" => Ok(HistogramNormalization::Probability),
        "percentage" => Ok(HistogramNormalization::Percentage),
        "countdensity" => Ok(HistogramNormalization::CountDensity),
        "pdf" | "probabilitydensity" => Ok(HistogramNormalization::Pdf),
        "cumcount" => Ok(HistogramNormalization::CumCount),
        "cdf" => Ok(HistogramNormalization::Cdf),
        other => Err(builtin_error(format!(
            "{NAME}: unrecognised Normalization value '{other}'"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntValue, IntegerStorage, ResolveContext, Type};

    fn tensor_from_value(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("expected tensor value, got {:?}", other),
        }
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    fn double_values(tensor: &Tensor) -> &[f64] {
        tensor.as_f64_slice().expect("expected double tensor")
    }

    #[test]
    fn histcounts2_type_defaults_to_unknown_matrix() {
        let ctx = ResolveContext::new(Vec::new());
        let out = histcounts2_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(5), Some(1)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(5), Some(1)]),
                },
            ],
            &ctx,
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn histcounts2_type_uses_edge_vectors() {
        let ctx = ResolveContext::new(Vec::new());
        let out = histcounts2_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(5), Some(1)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(5), Some(1)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(4)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(6)]),
                },
            ],
            &ctx,
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(5)])
            }
        );
    }

    #[test]
    fn histcounts2_numbins_preserves_typed_integers_and_rejects_lossy_f64() {
        assert_eq!(
            positive_usize(&Value::Int(IntValue::U16(3)), NAME, "NumBins").unwrap(),
            3
        );
        assert_eq!(
            positive_usize(
                &integer_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
                NAME,
                "NumBins",
            )
            .unwrap(),
            4
        );
        assert_eq!(
            positive_usize(
                &integer_tensor(IntegerStorage::U16(vec![9]), vec![1, 1]),
                NAME,
                "NumBins",
            )
            .unwrap(),
            9
        );
        assert_eq!(
            num_bins_vector(&integer_tensor(
                IntegerStorage::U16(vec![7, 11]),
                vec![1, 2],
            ))
            .unwrap(),
            vec![7, 11]
        );
        for value in [
            Value::Int(IntValue::I8(-1)),
            integer_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
            integer_tensor(IntegerStorage::I16(vec![2, -1]), vec![1, 2]),
            integer_tensor(IntegerStorage::U64(vec![usize::MAX as u64]), vec![1, 1]),
            integer_tensor(IntegerStorage::U64(vec![2, usize::MAX as u64]), vec![1, 2]),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(positive_usize(&value, NAME, "NumBins").is_err());
        }
        assert!(num_bins_vector(&integer_tensor(
            IntegerStorage::I16(vec![2, -1]),
            vec![1, 2],
        ))
        .is_err());
    }

    #[test]
    fn histcounts2_numeric_vectors_read_typed_integer_storage_exactly() {
        assert_eq!(
            numeric_vector(
                &integer_tensor(IntegerStorage::I16(vec![1, 3, 5]), vec![1, 3]),
                NAME,
                "XBinEdges",
            )
            .unwrap(),
            vec![1.0, 3.0, 5.0]
        );
        assert_eq!(
            scalar_value(
                &integer_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                NAME,
                "BinWidth",
            )
            .unwrap(),
            2.0
        );
    }

    #[test]
    fn histcounts2_accepts_every_integer_class_and_native_single_at_floating_boundary() {
        let integer_cases = [
            IntegerStorage::I8(vec![0, 1, 2]),
            IntegerStorage::I16(vec![0, 1, 2]),
            IntegerStorage::I32(vec![0, 1, 2]),
            IntegerStorage::I64(vec![0, 1, 2]),
            IntegerStorage::U8(vec![0, 1, 2]),
            IntegerStorage::U16(vec![0, 1, 2]),
            IntegerStorage::U32(vec![0, 1, 2]),
            IntegerStorage::U64(vec![0, 1, 2]),
        ];
        for storage in integer_cases {
            let x = Tensor::new_integer(storage, vec![3, 1]).expect("integer observations");
            let y =
                Tensor::from_f32(vec![0.25, 1.25, 2.25], vec![3, 1]).expect("single observations");
            let eval = block_on(evaluate(
                Value::Tensor(x),
                Value::Tensor(y),
                &[
                    Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                    Value::Tensor(Tensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                ],
            ))
            .expect("typed histcounts2");
            let counts = tensor_from_value(eval.into_counts_value());
            assert_eq!(
                double_values(&counts),
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            );
        }
    }

    #[test]
    fn histcounts2_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = HISTCOUNTS2_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"N = histcounts2(X, Y)"));
        assert!(labels.contains(&"N = histcounts2(X, Y, binsX, binsY)"));
        assert!(labels.contains(&"[N, Xedges, Yedges] = histcounts2(X, Y, Name, Value, ...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_basic_counts() {
        let x = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![0.2, 0.9, 1.4, 2.8], vec![4, 1]).unwrap();
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            ],
        ))
        .expect("histcounts2");
        let (counts, xedges, yedges) = eval.into_triple();
        let counts = tensor_from_value(counts);
        assert_eq!(counts.shape, vec![4, 3]);
        let xedges = tensor_from_value(xedges);
        let yedges = tensor_from_value(yedges);
        assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(double_values(&yedges), &[0.0, 1.0, 2.0, 3.0]);
        let rows = counts.shape[0];
        let count = |ix: usize, iy: usize| double_values(&counts)[ix + iy * rows];
        assert_eq!(count(0, 0), 1.0);
        assert_eq!(count(1, 0), 1.0);
        assert_eq!(count(2, 1), 1.0);
        assert_eq!(count(3, 2), 1.0);
        for iy in 0..counts.shape[1] {
            for ix in 0..rows {
                let keep = matches!((ix, iy), (0, 0) | (1, 0) | (2, 1) | (3, 2));
                if !keep {
                    assert_eq!(count(ix, iy), 0.0);
                }
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_probability_normalization() {
        let x = Tensor::new(vec![0.2, 0.4, 1.1, 1.5], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![0.1, 0.8, 1.2, 1.9], vec![4, 1]).unwrap();
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::from("Normalization"),
                Value::from("probability"),
            ],
        ))
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(counts.shape, vec![2, 2]);
        assert_eq!(double_values(&counts), &[0.5, 0.0, 0.0, 0.5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_nan_pairs_excluded() {
        let x = Tensor::new(vec![1.0, 2.0, f64::NAN, 3.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![2.0, 2.0, 2.0, f64::NAN], vec![4, 1]).unwrap();
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
            ],
        ))
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(double_values(&counts).iter().sum::<f64>(), 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_num_bins_vector() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::from("NumBins"),
                Value::Tensor(Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap()),
            ],
        ))
        .expect("histcounts2");
        let counts = tensor_from_value(eval.into_counts_value());
        assert_eq!(counts.shape, vec![2, 4]);
        assert_eq!(double_values(&counts).iter().sum::<f64>(), 4.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_bin_width_and_limits() {
        let x_tensor = Tensor::new(vec![0.2, 1.2, 2.4, 2.6], vec![4, 1]).unwrap();
        let y_tensor = Tensor::new(vec![0.1, 0.6, 1.4, 2.2], vec![4, 1]).unwrap();
        let bin_limits = Tensor::new(vec![0.0, 3.0, 0.0, 2.5], vec![4, 1]).unwrap();

        let eval = block_on(evaluate(
            Value::Tensor(x_tensor.clone()),
            Value::Tensor(y_tensor.clone()),
            &[
                Value::from("XBinWidth"),
                Value::Num(1.0),
                Value::from("YBinWidth"),
                Value::Num(0.5),
                Value::from("BinLimits"),
                Value::Tensor(bin_limits.clone()),
            ],
        ))
        .expect("histcounts2");
        let (counts_v, xedges_v, yedges_v) = eval.into_triple();
        let counts = tensor_from_value(counts_v);
        let xedges = tensor_from_value(xedges_v);
        let yedges = tensor_from_value(yedges_v);

        assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(double_values(&yedges), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(counts.shape, vec![3, 5]);
        assert_eq!(double_values(&counts).iter().sum::<f64>(), 4.0);

        let density_eval = block_on(evaluate(
            Value::Tensor(x_tensor),
            Value::Tensor(y_tensor),
            &[
                Value::from("XBinWidth"),
                Value::Num(1.0),
                Value::from("YBinWidth"),
                Value::Num(0.5),
                Value::from("BinLimits"),
                Value::Tensor(bin_limits),
                Value::from("Normalization"),
                Value::from("countdensity"),
            ],
        ))
        .expect("histcounts2 countdensity");
        let density = tensor_from_value(density_eval.into_counts_value());
        let positives: Vec<f64> = double_values(&density)
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .collect();
        assert!(!positives.is_empty());
        for value in positives {
            assert!((value - 2.0).abs() < 1e-12);
        }
        assert_eq!(double_values(&density)[0], 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_cdf_normalization() {
        let x = Tensor::new(vec![0.1, 0.9, 1.2, 1.8], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![0.2, 0.7, 1.4, 1.6], vec![4, 1]).unwrap();
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap()),
                Value::from("Normalization"),
                Value::from("cdf"),
            ],
        ))
        .expect("histcounts2");
        let cdf = tensor_from_value(eval.into_counts_value());
        assert_eq!(cdf.shape, vec![2, 2]);
        assert_eq!(double_values(&cdf).last().copied().unwrap(), 1.0);
        assert!(double_values(&cdf).windows(2).all(|w| w[0] <= w[1] + 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_integer_bin_method() {
        let x = Tensor::new(vec![0.2, 1.7, 2.1], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.2, 1.8, 2.9], vec![3, 1]).unwrap();
        let eval = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::from("XBinMethod"),
                Value::from("integers"),
                Value::from("YBinMethod"),
                Value::from("integers"),
                Value::from("BinLimits"),
                Value::Tensor(Tensor::new(vec![0.0, 3.0, 1.0, 3.0], vec![4, 1]).unwrap()),
            ],
        ))
        .expect("histcounts2");
        let (counts_v, xedges_v, yedges_v) = eval.into_triple();
        let counts = tensor_from_value(counts_v);
        let xedges = tensor_from_value(xedges_v);
        let yedges = tensor_from_value(yedges_v);
        assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(double_values(&yedges), &[1.0, 2.0, 3.0]);
        assert_eq!(double_values(&counts).iter().sum::<f64>(), 3.0);
        assert!(double_values(&xedges)
            .windows(2)
            .all(|w| (w[1] - w[0] - 1.0).abs() < 1e-12));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_num_bins_zero_errors() {
        let x = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[Value::from("NumBins"), Value::Num(0.0)],
        ))
        .err()
        .expect("NumBins=0 should fail");
        assert_eq!(
            err.identifier(),
            HISTCOUNTS2_ERROR_NUMBINS_INVALID.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_binmethod_conflict_errors() {
        let x = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let err = block_on(evaluate(
            Value::Tensor(x),
            Value::Tensor(y),
            &[
                Value::from("XBinEdges"),
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::from("XBinMethod"),
                Value::from("auto"),
            ],
        ))
        .err()
        .expect("XBinEdges + XBinMethod should fail");
        assert_eq!(
            err.identifier(),
            HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT.identifier
        );
    }

    #[test]
    fn histcounts2_returns_bin_indices_with_source_shapes() {
        let x = integer_tensor(IntegerStorage::I16(vec![0, 2, 9]), vec![1, 3]);
        let y = integer_tensor(IntegerStorage::I16(vec![0, 2, 1]), vec![1, 3]);
        let edges = integer_tensor(IntegerStorage::I16(vec![0, 1, 3]), vec![1, 3]);
        let eval = block_on(evaluate(x, y, &[edges.clone(), edges])).expect("histcounts2");
        let (_, _, _, bin_x, bin_y) = eval.into_all();
        let bin_x = tensor_from_value(bin_x);
        let bin_y = tensor_from_value(bin_y);
        assert_eq!(bin_x.shape, vec![1, 3]);
        assert_eq!(bin_y.shape, vec![1, 3]);
        assert_eq!(double_values(&bin_x), &[1.0, 2.0, 0.0]);
        assert_eq!(double_values(&bin_y), &[1.0, 2.0, 2.0]);
    }

    #[test]
    fn histcounts2_wide_explicit_edges_classify_exactly() {
        let base = 9_007_199_254_740_992_u64;
        let x = integer_tensor(IntegerStorage::U64(vec![base, base + 1]), vec![1, 2]);
        let y = integer_tensor(IntegerStorage::U64(vec![base, base + 1]), vec![1, 2]);
        let edges = integer_tensor(
            IntegerStorage::U64(vec![base, base + 1, base + 2]),
            vec![1, 3],
        );
        let eval = block_on(evaluate(x, y, &[edges.clone(), edges])).expect("histcounts2");
        let (counts, _, _, bin_x, bin_y) = eval.into_all();
        assert_eq!(
            double_values(&tensor_from_value(counts)),
            &[1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(double_values(&tensor_from_value(bin_x)), &[1.0, 2.0]);
        assert_eq!(double_values(&tensor_from_value(bin_y)), &[1.0, 2.0]);
    }

    #[test]
    fn histcounts2_percentage_and_cumulative_are_two_dimensional() {
        let counts = vec![1.0, 2.0, 3.0, 4.0];
        let edges = vec![0.0, 1.0, 2.0];
        assert_eq!(
            apply_normalization_2d(
                &counts,
                &edges,
                &edges,
                HistogramNormalization::Percentage,
                10
            ),
            vec![10.0, 20.0, 30.0, 40.0]
        );
        assert_eq!(
            apply_normalization_2d(
                &counts,
                &edges,
                &edges,
                HistogramNormalization::CumCount,
                10
            ),
            vec![1.0, 3.0, 4.0, 10.0]
        );
        assert_eq!(
            apply_normalization_2d(&counts, &edges, &edges, HistogramNormalization::Cdf, 10),
            vec![0.1, 0.3, 0.4, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn histcounts2_gpu_roundtrip() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let x = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
            let y = Tensor::new(vec![1.0, 1.1, 2.9], vec![3, 1]).unwrap();
            let x_view = runmat_accelerate_api::HostTensorView {
                data: double_values(&x),
                shape: &x.shape,
            };
            let y_view = runmat_accelerate_api::HostTensorView {
                data: double_values(&y),
                shape: &y.shape,
            };
            let x_handle = provider.upload(&x_view).expect("upload X");
            let y_handle = provider.upload(&y_view).expect("upload Y");
            let eval = block_on(evaluate(
                Value::GpuTensor(x_handle),
                Value::GpuTensor(y_handle),
                &[
                    Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                    Value::Tensor(Tensor::new(vec![0.0, 2.0, 3.0], vec![3, 1]).unwrap()),
                ],
            ))
            .expect("histcounts2");
            let counts = tensor_from_value(eval.into_counts_value());
            assert_eq!(counts.shape, vec![3, 2]);
            let rows = counts.shape[0];
            let count = |ix: usize, iy: usize| double_values(&counts)[ix + iy * rows];
            assert_eq!(count(0, 0), 1.0);
            assert_eq!(count(1, 0), 1.0);
            assert_eq!(count(2, 1), 1.0);
            assert_eq!(count(2, 0), 0.0);
            assert_eq!(count(0, 1), 0.0);
            assert_eq!(count(1, 1), 0.0);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn histcounts2_wgpu_roundtrip() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let x = Tensor::new(vec![0.5, 1.5, 2.5], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![1.0, 1.1, 2.9], vec![3, 1]).unwrap();

        let x_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: double_values(&x),
                shape: &x.shape,
            })
            .expect("upload x");
        let y_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: double_values(&y),
                shape: &y.shape,
            })
            .expect("upload y");

        let eval = block_on(evaluate(
            Value::GpuTensor(x_handle),
            Value::GpuTensor(y_handle),
            &[
                Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![0.0, 2.0, 3.0], vec![3, 1]).unwrap()),
            ],
        ))
        .expect("histcounts2");

        let (counts_v, xedges_v, yedges_v) = eval.into_triple();
        let counts = tensor_from_value(counts_v);
        let xedges = tensor_from_value(xedges_v);
        let yedges = tensor_from_value(yedges_v);

        assert_eq!(double_values(&xedges), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(double_values(&yedges), &[0.0, 2.0, 3.0]);

        assert_eq!(counts.shape, vec![3, 2]);
        let rows = counts.shape[0];
        let count = |ix: usize, iy: usize| double_values(&counts)[ix + iy * rows];
        assert_eq!(count(0, 0), 1.0);
        assert_eq!(count(1, 0), 1.0);
        assert_eq!(count(2, 1), 1.0);
        assert_eq!(count(2, 0), 0.0);
        assert_eq!(count(0, 1), 0.0);
        assert_eq!(count(1, 1), 0.0);
    }
}
