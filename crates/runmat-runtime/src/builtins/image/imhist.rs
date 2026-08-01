//! MATLAB-compatible `imhist` grayscale and indexed-image histograms.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, LogicalArray, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::image::color::common;
use crate::builtins::image::type_resolvers::imhist_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[cfg(feature = "plot-core")]
use runmat_plot::plots::BarChart;

const NAME: &str = "imhist";
const DEFAULT_GRAYSCALE_BINS: usize = 256;
const LOGICAL_BINS: usize = 2;
const MAX_BINS: usize = 1_000_000;
#[cfg(feature = "plot-core")]
const MAX_PLOT_BINS: usize = 4096;
const INTEGER_TOL: f64 = 1.0e-9;

const IMHIST_OUTPUT_COUNTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "counts",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Histogram bin counts as a column vector.",
}];

const IMHIST_OUTPUT_COUNTS_BINS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "counts",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Histogram bin counts as a column vector.",
    },
    BuiltinParamDescriptor {
        name: "binLocations",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Intensity or colormap-index bin locations as a column vector.",
    },
];

const IMHIST_INPUTS_IMAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grayscale intensity image.",
}];

const IMHIST_INPUTS_IMAGE_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grayscale intensity image.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("256"),
        description: "Number of bins.",
    },
];

const IMHIST_INPUTS_INDEXED: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexed image matrix.",
    },
    BuiltinParamDescriptor {
        name: "map",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Colormap with one RGB row per indexed-image bin.",
    },
];

const IMHIST_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "counts = imhist(I)",
        inputs: &IMHIST_INPUTS_IMAGE,
        outputs: &IMHIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "counts = imhist(I, n)",
        inputs: &IMHIST_INPUTS_IMAGE_N,
        outputs: &IMHIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, binLocations] = imhist(I)",
        inputs: &IMHIST_INPUTS_IMAGE,
        outputs: &IMHIST_OUTPUT_COUNTS_BINS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, binLocations] = imhist(I, n)",
        inputs: &IMHIST_INPUTS_IMAGE_N,
        outputs: &IMHIST_OUTPUT_COUNTS_BINS,
    },
    BuiltinSignatureDescriptor {
        label: "counts = imhist(X, map)",
        inputs: &IMHIST_INPUTS_INDEXED,
        outputs: &IMHIST_OUTPUT_COUNTS,
    },
    BuiltinSignatureDescriptor {
        label: "[counts, binLocations] = imhist(X, map)",
        inputs: &IMHIST_INPUTS_INDEXED,
        outputs: &IMHIST_OUTPUT_COUNTS_BINS,
    },
];

const IMHIST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:imhist:InvalidArgument"),
    when: "Image input, bin count, or colormap arguments are malformed or unsupported.",
    message: "imhist: invalid argument",
};

const IMHIST_ERROR_UNSUPPORTED_IMAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.UNSUPPORTED_IMAGE",
    identifier: Some("RunMat:imhist:UnsupportedImage"),
    when: "Input cannot be interpreted as a grayscale or indexed image.",
    message: "imhist: unsupported image input",
};

const IMHIST_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.PLOT_FAILED",
    identifier: Some("RunMat:imhist:PlotFailed"),
    when: "Statement-form histogram rendering fails.",
    message: "imhist: plotting failed",
};

const IMHIST_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:imhist:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "imhist: too many output arguments",
};

const IMHIST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMHIST.INTERNAL",
    identifier: Some("RunMat:imhist:Internal"),
    when: "Internal histogram assembly fails.",
    message: "imhist: internal error",
};

const IMHIST_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IMHIST_ERROR_INVALID_ARGUMENT,
    IMHIST_ERROR_UNSUPPORTED_IMAGE,
    IMHIST_ERROR_PLOT_FAILED,
    IMHIST_ERROR_TOO_MANY_OUTPUTS,
    IMHIST_ERROR_INTERNAL,
];

pub const IMHIST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IMHIST_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IMHIST_ERRORS,
};

fn imhist_error_with_message(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn imhist_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let raw = detail.as_ref().trim();
    let normalized = raw.strip_prefix("imhist:").map(str::trim).unwrap_or(raw);
    let message = if normalized.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {}", error.message, normalized)
    };
    imhist_error_with_message(error, message)
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_INVALID_ARGUMENT, detail)
}

fn unsupported(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_UNSUPPORTED_IMAGE, detail)
}

fn internal(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_INTERNAL, detail)
}

fn too_many_outputs() -> RuntimeError {
    imhist_error_with_message(
        &IMHIST_ERROR_TOO_MANY_OUTPUTS,
        IMHIST_ERROR_TOO_MANY_OUTPUTS.message,
    )
}

#[cfg(feature = "plot-core")]
fn plot_failed(detail: impl AsRef<str>) -> RuntimeError {
    imhist_error_with_detail(&IMHIST_ERROR_PLOT_FAILED, detail)
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        invalid(err.message())
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::image::imhist")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("image-histogram"),
    supported_precisions: &[crate::builtins::common::spec::ScalarType::F32, crate::builtins::common::spec::ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("imhist")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "imhist gathers gpuArray inputs today so image-class binning and output shapes remain MATLAB-compatible.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::image::imhist")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "imhist materializes histogram counts and terminates fusion chains.",
};

#[runtime_builtin(
    name = "imhist",
    category = "image",
    summary = "Compute or display grayscale and indexed-image histograms.",
    keywords = "imhist,image,histogram,intensity,grayscale,indexed,colormap",
    sink = true,
    suppress_auto_output = true,
    type_resolver(imhist_type),
    descriptor(crate::builtins::image::imhist::IMHIST_DESCRIPTOR),
    builtin_path = "crate::builtins::image::imhist"
)]
async fn imhist_builtin(image: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let eval = evaluate(image, &rest).await?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        eval.render_plot()?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            eval.render_plot()?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.counts_value()]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(eval.outputs()));
        }
        return Err(too_many_outputs());
    }

    Ok(eval.counts_value())
}

pub async fn evaluate(image: Value, rest: &[Value]) -> BuiltinResult<ImhistEvaluation> {
    let image = common::gather_value(NAME, &image).await.map_err(map_flow)?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for arg in rest {
        gathered_rest.push(common::gather_value(NAME, arg).await.map_err(map_flow)?);
    }
    let call = parse_call(image, &gathered_rest)?;
    Ok(match call.mode {
        ImhistMode::Grayscale { bins } => {
            let input = GrayscaleInput::from_value(call.image)?;
            input.evaluate(bins)?
        }
        ImhistMode::Indexed { bins } => {
            let input = IndexedInput::from_value(call.image, bins)?;
            input.evaluate()?
        }
    })
}

struct ParsedCall {
    image: Value,
    mode: ImhistMode,
}

enum ImhistMode {
    Grayscale { bins: Option<usize> },
    Indexed { bins: usize },
}

fn parse_call(image: Value, rest: &[Value]) -> BuiltinResult<ParsedCall> {
    match rest {
        [] => Ok(ParsedCall {
            image,
            mode: ImhistMode::Grayscale { bins: None },
        }),
        [second] => {
            if let Some(bins) = parse_optional_bin_count(second)? {
                Ok(ParsedCall {
                    image,
                    mode: ImhistMode::Grayscale { bins: Some(bins) },
                })
            } else {
                let bins = parse_colormap_bins(second)?;
                Ok(ParsedCall {
                    image,
                    mode: ImhistMode::Indexed { bins },
                })
            }
        }
        _ => Err(invalid(
            "expected imhist(I), imhist(I, n), or imhist(X, map)",
        )),
    }
}

fn parse_optional_bin_count(value: &Value) -> BuiltinResult<Option<usize>> {
    if let Some(bins) = integer_scalar_bin_count(value)? {
        validate_bin_count(bins)?;
        return Ok(Some(bins));
    }
    let Some(raw) = scalar_number(value) else {
        return Ok(None);
    };
    if !raw.is_finite() || raw < 1.0 || (raw.round() - raw).abs() > INTEGER_TOL {
        return Err(invalid("bin count must be a positive integer scalar"));
    }
    let rounded = raw.round();
    if !fits_platform_usize(rounded) {
        return Err(invalid("bin count is outside the supported platform range"));
    }
    if rounded > MAX_BINS as f64 {
        return Err(invalid(format!(
            "bin count {:.0} exceeds maximum supported bin count {MAX_BINS}",
            rounded
        )));
    }
    let bins = rounded as usize;
    validate_bin_count(bins)?;
    Ok(Some(bins))
}

fn parse_colormap_bins(value: &Value) -> BuiltinResult<usize> {
    let tensor = Tensor::try_from(value)
        .map_err(|err| invalid(format!("colormap must be an Nx3 numeric array: {err}")))?;
    if tensor.shape.len() != 2 || tensor.cols != 3 || tensor.rows == 0 {
        return Err(invalid("colormap must be a non-empty Nx3 numeric array"));
    }
    let values = tensor_utils::tensor_values_f64_cow(&tensor);
    if !values.iter().all(|value| value.is_finite()) {
        return Err(invalid("colormap values must be finite"));
    }
    if !values.iter().all(|value| (0.0..=1.0).contains(value)) {
        return Err(invalid("colormap values must be in the range [0, 1]"));
    }
    validate_bin_count(tensor.rows)?;
    Ok(tensor.rows)
}

fn validate_bin_count(bins: usize) -> BuiltinResult<()> {
    if bins == 0 {
        return Err(invalid("bin count must be positive"));
    }
    if bins > MAX_BINS {
        return Err(invalid(format!(
            "bin count {bins} exceeds maximum supported bin count {MAX_BINS}"
        )));
    }
    Ok(())
}

#[derive(Clone)]
struct GrayscaleInput {
    storage: NumericStorage,
    class_min: f64,
    class_max: f64,
    default_bins: usize,
}

impl GrayscaleInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => Self::from_tensor(tensor),
            Value::LogicalArray(logical) => Self::from_logical(logical),
            Value::Num(value) => {
                Self::from_float_storage(NumericStorage::F64(vec![value]), &[1, 1])
            }
            Value::Int(value) => Ok(Self::from_integer_scalar(value)),
            Value::Bool(value) => Ok(Self {
                storage: NumericStorage::F64(vec![if value { 1.0 } else { 0.0 }]),
                class_min: 0.0,
                class_max: 1.0,
                default_bins: LOGICAL_BINS,
            }),
            other => Err(unsupported(format!(
                "expected grayscale numeric or logical image, got {other:?}"
            ))),
        }
    }

    fn from_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        if !is_grayscale_shape(&tensor.shape) {
            return Err(unsupported(
                "expected an MxN grayscale image; truecolor RGB images are not accepted",
            ));
        }
        let shape = tensor.shape.clone();
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| internal(format!("grayscale tensor storage: {err}")))?;
        match storage {
            storage @ NumericStorage::I8(_) => Ok(Self::from_integer_storage(
                storage,
                i8::MIN as f64,
                i8::MAX as f64,
            )),
            storage @ NumericStorage::I16(_) => Ok(Self::from_integer_storage(
                storage,
                i16::MIN as f64,
                i16::MAX as f64,
            )),
            storage @ NumericStorage::I32(_) => Ok(Self::from_integer_storage(
                storage,
                i32::MIN as f64,
                i32::MAX as f64,
            )),
            storage @ NumericStorage::I64(_) => Ok(Self::from_integer_storage(
                storage,
                i64::MIN as f64,
                i64::MAX as f64,
            )),
            storage @ NumericStorage::U8(_) => Ok(Self::from_integer_storage(storage, 0.0, 255.0)),
            storage @ NumericStorage::U16(_) => {
                Ok(Self::from_integer_storage(storage, 0.0, 65535.0))
            }
            storage @ NumericStorage::U32(_) => {
                Ok(Self::from_integer_storage(storage, 0.0, u32::MAX as f64))
            }
            storage @ NumericStorage::U64(_) => {
                Ok(Self::from_integer_storage(storage, 0.0, u64::MAX as f64))
            }
            storage @ (NumericStorage::F32(_) | NumericStorage::F64(_)) => {
                Self::from_float_storage(storage, &shape)
            }
        }
    }

    fn from_integer_scalar(value: IntValue) -> Self {
        match value {
            IntValue::I8(value) => Self::from_integer_storage(
                NumericStorage::I8(vec![value]),
                i8::MIN as f64,
                i8::MAX as f64,
            ),
            IntValue::I16(value) => Self::from_integer_storage(
                NumericStorage::I16(vec![value]),
                i16::MIN as f64,
                i16::MAX as f64,
            ),
            IntValue::I32(value) => Self::from_integer_storage(
                NumericStorage::I32(vec![value]),
                i32::MIN as f64,
                i32::MAX as f64,
            ),
            IntValue::I64(value) => Self::from_integer_storage(
                NumericStorage::I64(vec![value]),
                i64::MIN as f64,
                i64::MAX as f64,
            ),
            IntValue::U8(value) => {
                Self::from_integer_storage(NumericStorage::U8(vec![value]), 0.0, 255.0)
            }
            IntValue::U16(value) => {
                Self::from_integer_storage(NumericStorage::U16(vec![value]), 0.0, 65535.0)
            }
            IntValue::U32(value) => {
                Self::from_integer_storage(NumericStorage::U32(vec![value]), 0.0, u32::MAX as f64)
            }
            IntValue::U64(value) => {
                Self::from_integer_storage(NumericStorage::U64(vec![value]), 0.0, u64::MAX as f64)
            }
        }
    }

    fn from_logical(logical: LogicalArray) -> BuiltinResult<Self> {
        if !is_grayscale_shape(&logical.shape) {
            return Err(unsupported("expected an MxN logical image"));
        }
        Ok(Self {
            storage: NumericStorage::F64(
                logical
                    .data
                    .into_iter()
                    .map(|value| if value == 0 { 0.0 } else { 1.0 })
                    .collect(),
            ),
            class_min: 0.0,
            class_max: 1.0,
            default_bins: LOGICAL_BINS,
        })
    }

    fn from_integer_storage(storage: NumericStorage, class_min: f64, class_max: f64) -> Self {
        Self {
            storage,
            class_min,
            class_max,
            default_bins: DEFAULT_GRAYSCALE_BINS,
        }
    }

    fn from_float_storage(storage: NumericStorage, shape: &[usize]) -> BuiltinResult<Self> {
        if !is_grayscale_shape(shape) {
            return Err(unsupported(
                "expected an MxN grayscale image; truecolor RGB images are not accepted",
            ));
        }
        let valid = match &storage {
            NumericStorage::F64(values) => values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            NumericStorage::F32(values) => values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            _ => false,
        };
        if !valid {
            return Err(invalid(
                "floating-point grayscale image values must be finite and normalized to [0, 1]",
            ));
        }
        Ok(Self {
            storage,
            class_min: 0.0,
            class_max: 1.0,
            default_bins: DEFAULT_GRAYSCALE_BINS,
        })
    }

    fn evaluate(&self, requested_bins: Option<usize>) -> BuiltinResult<ImhistEvaluation> {
        let bins = requested_bins.unwrap_or(self.default_bins);
        validate_bin_count(bins)?;
        let locations = linspace(self.class_min, self.class_max, bins);
        let counts = histogram_counts(&self.storage, self.class_min, self.class_max, bins)?;
        ImhistEvaluation::from_counts_locations(counts, locations)
    }
}

struct IndexedInput {
    storage: NumericStorage,
    zero_based: bool,
    bins: usize,
}

impl IndexedInput {
    fn from_value(value: Value, bins: usize) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => {
                if !is_grayscale_shape(&tensor.shape) {
                    return Err(unsupported("indexed image must be an MxN matrix"));
                }
                let zero_based = matches!(
                    tensor.numeric_dtype(),
                    runmat_builtins::NumericDType::U8 | runmat_builtins::NumericDType::U16
                );
                let storage = tensor
                    .into_numeric_storage()
                    .map_err(|err| internal(format!("indexed tensor storage: {err}")))?;
                Ok(Self {
                    storage,
                    zero_based,
                    bins,
                })
            }
            Value::LogicalArray(logical) => {
                if !is_grayscale_shape(&logical.shape) {
                    return Err(unsupported("indexed image must be an MxN matrix"));
                }
                Ok(Self {
                    storage: NumericStorage::F64(
                        logical
                            .data
                            .into_iter()
                            .map(|value| if value == 0 { 0.0 } else { 1.0 })
                            .collect(),
                    ),
                    zero_based: true,
                    bins,
                })
            }
            Value::Num(value) => Ok(Self {
                storage: NumericStorage::F64(vec![value]),
                zero_based: false,
                bins,
            }),
            Value::Int(value) => {
                let zero_based = matches!(value, IntValue::U8(_) | IntValue::U16(_));
                Ok(Self {
                    storage: numeric_storage_from_int(value),
                    zero_based,
                    bins,
                })
            }
            Value::Bool(value) => Ok(Self {
                storage: NumericStorage::F64(vec![if value { 1.0 } else { 0.0 }]),
                zero_based: true,
                bins,
            }),
            other => Err(unsupported(format!(
                "expected indexed numeric or logical image, got {other:?}"
            ))),
        }
    }

    fn evaluate(&self) -> BuiltinResult<ImhistEvaluation> {
        let mut counts = vec![0.0; self.bins];
        count_indexed_values(&self.storage, self.zero_based, &mut counts)?;
        let locations: Vec<f64> = (1..=self.bins).map(|value| value as f64).collect();
        ImhistEvaluation::from_counts_locations(counts, locations)
    }
}

fn numeric_storage_from_int(value: IntValue) -> NumericStorage {
    match value {
        IntValue::I8(value) => NumericStorage::I8(vec![value]),
        IntValue::I16(value) => NumericStorage::I16(vec![value]),
        IntValue::I32(value) => NumericStorage::I32(vec![value]),
        IntValue::I64(value) => NumericStorage::I64(vec![value]),
        IntValue::U8(value) => NumericStorage::U8(vec![value]),
        IntValue::U16(value) => NumericStorage::U16(vec![value]),
        IntValue::U32(value) => NumericStorage::U32(vec![value]),
        IntValue::U64(value) => NumericStorage::U64(vec![value]),
    }
}

pub struct ImhistEvaluation {
    counts: Tensor,
    locations: Tensor,
}

impl ImhistEvaluation {
    fn from_counts_locations(counts: Vec<f64>, locations: Vec<f64>) -> BuiltinResult<Self> {
        if counts.len() != locations.len() {
            return Err(internal("counts and bin locations length mismatch"));
        }
        let rows = counts.len();
        let counts = Tensor::new(counts, vec![rows, 1])
            .map_err(|err| internal(format!("counts tensor: {err}")))?;
        let locations = Tensor::new(locations, vec![rows, 1])
            .map_err(|err| internal(format!("bin location tensor: {err}")))?;
        Ok(Self { counts, locations })
    }

    fn counts_value(&self) -> Value {
        Value::Tensor(self.counts.clone())
    }

    fn locations_value(&self) -> Value {
        Value::Tensor(self.locations.clone())
    }

    fn outputs(&self) -> Vec<Value> {
        vec![self.counts_value(), self.locations_value()]
    }

    fn render_plot(&self) -> BuiltinResult<()> {
        render_imhist_plot(&self.counts, &self.locations)
    }
}

#[cfg(feature = "plot-core")]
fn render_imhist_plot(counts: &Tensor, locations: &Tensor) -> BuiltinResult<()> {
    let plot_data = plot_display_bins(counts, locations)?;
    let mut chart = BarChart::new(plot_data.labels, plot_data.counts)
        .map_err(|err| plot_failed(format!("chart construction failed: {err}")))?;
    chart.set_bar_width(0.95);
    chart.set_color(glam::Vec4::new(0.1, 0.1, 0.1, 0.95));
    let mut chart = Some(chart);
    let render_result = crate::builtins::plotting::state::render_active_plot(
        NAME,
        crate::builtins::plotting::state::PlotRenderOptions {
            title: "Image Histogram",
            x_label: "Intensity",
            y_label: "Count",
            ..Default::default()
        },
        move |figure, axes| {
            figure.add_bar_chart_on_axes(
                chart.take().expect("imhist chart consumed exactly once"),
                axes,
            );
            Ok(())
        },
    );
    if let Err(err) = render_result {
        let lower = err.message().to_ascii_lowercase();
        if lower.contains("plotting is unavailable") || lower.contains("non-main thread") {
            return Ok(());
        }
        return Err(plot_failed(err.message()));
    }
    Ok(())
}

#[cfg(not(feature = "plot-core"))]
fn render_imhist_plot(_counts: &Tensor, _locations: &Tensor) -> BuiltinResult<()> {
    Ok(())
}

fn is_grayscale_shape(shape: &[usize]) -> bool {
    matches!(shape.len(), 0..=2)
}

fn scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn integer_scalar_bin_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let integer = match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    };
    let Some(integer) = integer else {
        return Ok(None);
    };
    integer
        .try_to_usize()
        .ok_or_else(|| invalid("bin count is outside the supported platform range"))
        .map(Some)
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    (0..count).map(|idx| start + step * idx as f64).collect()
}

fn histogram_counts(
    storage: &NumericStorage,
    class_min: f64,
    class_max: f64,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    match storage {
        NumericStorage::F64(values) => histogram_float_counts(values, class_min, class_max, bins),
        NumericStorage::F32(values) => {
            let values: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
            histogram_float_counts(&values, class_min, class_max, bins)
        }
        NumericStorage::I8(values) => histogram_signed_counts(values, i8::MIN, bins),
        NumericStorage::I16(values) => histogram_signed_counts(values, i16::MIN, bins),
        NumericStorage::I32(values) => histogram_signed_counts(values, i32::MIN, bins),
        NumericStorage::I64(values) => histogram_signed_counts(values, i64::MIN, bins),
        NumericStorage::U8(values) => histogram_unsigned_counts(values, u8::MAX, bins),
        NumericStorage::U16(values) => histogram_unsigned_counts(values, u16::MAX, bins),
        NumericStorage::U32(values) => histogram_unsigned_counts(values, u32::MAX, bins),
        NumericStorage::U64(values) => histogram_unsigned_counts(values, u64::MAX, bins),
    }
}

fn histogram_float_counts(
    values: &[f64],
    class_min: f64,
    class_max: f64,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let mut counts = vec![0.0; bins];
    if bins == 0 {
        return Ok(counts);
    }
    if bins == 1 || (class_max - class_min).abs() <= f64::EPSILON {
        for &value in values {
            if !value.is_finite() || value < class_min || value > class_max {
                return Err(invalid(
                    "grayscale image values are outside the image class range",
                ));
            }
            counts[0] += 1.0;
        }
        return Ok(counts);
    }
    let scale = (bins - 1) as f64 / (class_max - class_min);
    for &value in values {
        if !value.is_finite() || value < class_min || value > class_max {
            return Err(invalid(
                "grayscale image values are outside the image class range",
            ));
        }
        let relative = ((value - class_min) * scale).round();
        let index = if relative <= 0.0 {
            0
        } else if relative >= (bins - 1) as f64 {
            bins - 1
        } else {
            relative as usize
        };
        counts[index] += 1.0;
    }
    Ok(counts)
}

fn nearest_integer_bin(offset: u128, range: u128, bins: usize) -> usize {
    if bins <= 1 || range == 0 {
        return 0;
    }
    let numerator = offset * (bins - 1) as u128;
    ((numerator * 2 + range) / (range * 2)) as usize
}

trait HistogramSigned: Copy {
    fn to_i128(self) -> i128;
}

macro_rules! impl_histogram_signed {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl HistogramSigned for $ty {
                fn to_i128(self) -> i128 {
                    self as i128
                }
            }
        )+
    };
}

impl_histogram_signed!(i8, i16, i32, i64);

fn histogram_signed_counts<T: HistogramSigned>(
    values: &[T],
    minimum: T,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let minimum = minimum.to_i128();
    let range = (-minimum * 2 - 1) as u128;
    let mut counts = vec![0.0; bins];
    for &value in values {
        let offset = (value.to_i128() - minimum) as u128;
        counts[nearest_integer_bin(offset, range, bins)] += 1.0;
    }
    Ok(counts)
}

trait HistogramUnsigned: Copy {
    fn to_u128(self) -> u128;
}

macro_rules! impl_histogram_unsigned {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl HistogramUnsigned for $ty {
                fn to_u128(self) -> u128 {
                    self as u128
                }
            }
        )+
    };
}

impl_histogram_unsigned!(u8, u16, u32, u64);

fn histogram_unsigned_counts<T: HistogramUnsigned>(
    values: &[T],
    maximum: T,
    bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let range = maximum.to_u128();
    let mut counts = vec![0.0; bins];
    for &value in values {
        counts[nearest_integer_bin(value.to_u128(), range, bins)] += 1.0;
    }
    Ok(counts)
}

fn count_indexed_values(
    storage: &NumericStorage,
    zero_based: bool,
    counts: &mut [f64],
) -> BuiltinResult<()> {
    match storage {
        NumericStorage::F64(values) => count_float_indices(values, zero_based, counts),
        NumericStorage::F32(values) => {
            let values: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
            count_float_indices(&values, zero_based, counts)
        }
        NumericStorage::I8(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::I16(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::I32(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::I64(values) => count_signed_indices(values, zero_based, counts),
        NumericStorage::U8(values) => count_unsigned_indices(values, zero_based, counts),
        NumericStorage::U16(values) => count_unsigned_indices(values, zero_based, counts),
        NumericStorage::U32(values) => count_unsigned_indices(values, zero_based, counts),
        NumericStorage::U64(values) => count_unsigned_indices(values, zero_based, counts),
    }
}

fn count_float_indices(values: &[f64], zero_based: bool, counts: &mut [f64]) -> BuiltinResult<()> {
    for &value in values {
        if !value.is_finite() || (value.round() - value).abs() > INTEGER_TOL {
            return Err(invalid(
                "indexed image values must be finite integer indices",
            ));
        }
        let rounded = value.round();
        let adjusted = if zero_based { rounded } else { rounded - 1.0 };
        if adjusted < 0.0 || adjusted >= counts.len() as f64 {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        }
        counts[adjusted as usize] += 1.0;
    }
    Ok(())
}

fn count_signed_indices<T: HistogramSigned + std::fmt::Display>(
    values: &[T],
    zero_based: bool,
    counts: &mut [f64],
) -> BuiltinResult<()> {
    for &value in values {
        let raw = value.to_i128();
        let adjusted = if zero_based { raw } else { raw - 1 };
        if adjusted < 0 || adjusted as u128 >= counts.len() as u128 {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        }
        counts[adjusted as usize] += 1.0;
    }
    Ok(())
}

fn count_unsigned_indices<T: HistogramUnsigned + std::fmt::Display>(
    values: &[T],
    zero_based: bool,
    counts: &mut [f64],
) -> BuiltinResult<()> {
    for &value in values {
        let raw = value.to_u128();
        let Some(adjusted) = raw.checked_sub(u128::from(!zero_based)) else {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        };
        if adjusted >= counts.len() as u128 {
            return Err(invalid(format!(
                "indexed image value {value} is outside the colormap range"
            )));
        }
        counts[adjusted as usize] += 1.0;
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
fn format_bin_label(value: f64) -> String {
    if (value.round() - value).abs() <= INTEGER_TOL {
        format!("{:.0}", value)
    } else {
        format!("{:.3}", value)
    }
}

#[cfg(feature = "plot-core")]
struct PlotDisplayBins {
    labels: Vec<String>,
    counts: Vec<f64>,
}

#[cfg(feature = "plot-core")]
fn plot_display_bins(counts: &Tensor, locations: &Tensor) -> BuiltinResult<PlotDisplayBins> {
    let counts = tensor_utils::tensor_values_f64_cow(counts);
    let locations = tensor_utils::tensor_values_f64_cow(locations);
    if counts.len() != locations.len() {
        return Err(internal("counts and bin locations length mismatch"));
    }
    if counts.is_empty() {
        return Err(internal("histogram has no bins to plot"));
    }
    if counts.len() <= MAX_PLOT_BINS {
        return Ok(PlotDisplayBins {
            labels: locations
                .iter()
                .map(|value| format_bin_label(*value))
                .collect(),
            counts: counts.to_vec(),
        });
    }

    let stride = counts.len().div_ceil(MAX_PLOT_BINS);
    let mut labels = Vec::with_capacity(counts.len().div_ceil(stride));
    let mut display_counts = Vec::with_capacity(labels.capacity());
    for start in (0..counts.len()).step_by(stride) {
        let end = (start + stride).min(counts.len());
        let total = counts[start..end].iter().sum::<f64>();
        let location = 0.5 * (locations[start] + locations[end - 1]);
        labels.push(format_bin_label(location));
        display_counts.push(total);
    }

    Ok(PlotDisplayBins {
        labels,
        counts: display_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, NumericDType};

    fn call(image: Value, rest: Vec<Value>, outputs: Option<usize>) -> Value {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(imhist_builtin(image, rest)).expect("imhist")
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Tensor {
        Tensor::new_with_dtype(data, shape, dtype).unwrap()
    }

    #[test]
    fn uint8_grayscale_default_bins_count_exact_intensities() {
        let image = tensor(vec![0.0, 1.0, 1.0, 255.0], vec![2, 2], NumericDType::U8);
        let Value::Tensor(counts) = call(Value::Tensor(image), vec![], None) else {
            panic!("expected counts tensor");
        };
        assert_eq!(counts.shape, vec![256, 1]);
        assert_eq!(counts.data[0], 1.0);
        assert_eq!(counts.data[1], 2.0);
        assert_eq!(counts.data[255], 1.0);
    }

    #[test]
    fn exact_integer_grayscale_images_use_their_storage_class_range() {
        let u8 = Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 1, u8::MAX]), vec![2, 2])
            .expect("uint8 tensor");
        let u16 = Tensor::new_integer(
            IntegerStorage::U16(vec![0, 1, u16::MAX, u16::MAX]),
            vec![2, 2],
        )
        .expect("uint16 tensor");

        let u8_image = GrayscaleInput::from_tensor(u8).expect("uint8 image");
        assert_eq!(u8_image.class_max, 255.0);
        assert_eq!(u8_image.default_bins, 256);
        let u8_eval = u8_image.evaluate(None).expect("uint8 histogram");
        assert_eq!(u8_eval.counts.data[0], 1.0);
        assert_eq!(u8_eval.counts.data[1], 2.0);
        assert_eq!(u8_eval.counts.data[255], 1.0);

        let u16_image = GrayscaleInput::from_tensor(u16).expect("uint16 image");
        assert_eq!(u16_image.class_max, 65535.0);
        assert_eq!(u16_image.default_bins, 256);
    }

    #[test]
    fn two_outputs_return_counts_and_bin_locations_as_columns() {
        let image = tensor(vec![0.0, 0.5, 1.0, 1.0], vec![2, 2], NumericDType::F64);
        let Value::OutputList(outputs) = call(Value::Tensor(image), vec![Value::Num(3.0)], Some(2))
        else {
            panic!("expected output list");
        };
        let counts = Tensor::try_from(&outputs[0]).unwrap();
        let locations = Tensor::try_from(&outputs[1]).unwrap();
        assert_eq!(counts.shape, vec![3, 1]);
        assert_eq!(locations.shape, vec![3, 1]);
        assert_eq!(counts.data, vec![1.0, 1.0, 2.0]);
        assert_eq!(locations.data, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn logical_image_uses_two_bins() {
        let logical = LogicalArray::new(vec![0, 1, 1, 0, 1, 0], vec![2, 3]).unwrap();
        let Value::Tensor(counts) = call(Value::LogicalArray(logical), vec![], None) else {
            panic!("expected counts");
        };
        assert_eq!(counts.shape, vec![2, 1]);
        assert_eq!(counts.data, vec![3.0, 3.0]);
    }

    #[test]
    fn indexed_image_counts_colormap_indices() {
        let image = tensor(vec![1.0, 2.0, 3.0, 2.0], vec![2, 2], NumericDType::F64);
        let map = tensor(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![3, 3],
            NumericDType::F64,
        );
        let Value::OutputList(outputs) =
            call(Value::Tensor(image), vec![Value::Tensor(map)], Some(2))
        else {
            panic!("expected output list");
        };
        let counts = Tensor::try_from(&outputs[0]).unwrap();
        let locations = Tensor::try_from(&outputs[1]).unwrap();
        assert_eq!(counts.data, vec![1.0, 2.0, 1.0]);
        assert_eq!(locations.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn uint8_indexed_image_uses_zero_based_colormap_indices() {
        let image = tensor(vec![0.0, 1.0, 1.0, 2.0], vec![2, 2], NumericDType::U8);
        let map = tensor(vec![0.0; 9], vec![3, 3], NumericDType::F64);
        let Value::Tensor(counts) = call(Value::Tensor(image), vec![Value::Tensor(map)], None)
        else {
            panic!("expected counts");
        };
        assert_eq!(counts.data, vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn exact_integer_indexed_images_use_zero_based_indices() {
        let image = Tensor::new_integer(IntegerStorage::U16(vec![0, 1, 1, 2]), vec![2, 2])
            .expect("uint16 tensor");
        let indexed = IndexedInput::from_value(Value::Tensor(image), 3).expect("indexed input");

        assert!(indexed.zero_based);
        assert_eq!(indexed.storage, NumericStorage::U16(vec![0, 1, 1, 2]));
    }

    #[test]
    fn wide_integer_grayscale_storage_stays_exact_through_binning() {
        let image = Tensor::new_integer(
            IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
            vec![1, 3],
        )
        .expect("uint64 tensor");
        let grayscale = GrayscaleInput::from_tensor(image).expect("uint64 image");

        assert_eq!(
            grayscale.storage,
            NumericStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX])
        );
        let evaluation = grayscale.evaluate(Some(256)).expect("uint64 histogram");
        let counts = tensor_utils::tensor_values_f64_cow(&evaluation.counts);
        assert_eq!(counts.iter().sum::<f64>(), 3.0);
        assert_eq!(counts[0], 2.0);
        assert_eq!(counts[255], 1.0);
    }

    #[test]
    fn wide_integer_index_errors_report_the_exact_native_value() {
        let exact = (1_u64 << 53) + 1;
        let image = Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1])
            .expect("uint64 tensor");
        let indexed =
            IndexedInput::from_value(Value::Tensor(image), 3).expect("indexed uint64 image");

        let Err(err) = indexed.evaluate() else {
            panic!("wide index must be rejected");
        };
        assert!(err.message().contains(&exact.to_string()));
    }

    #[test]
    fn scalar_number_reads_typed_integer_storage_exactly() {
        let mut bins = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
            .expect("typed bin count");
        bins.data.clear();

        assert_eq!(scalar_number(&Value::Tensor(bins)), Some(2026.0));
    }

    #[test]
    fn bin_count_parser_preserves_typed_integer_scalar_bounds() {
        assert_eq!(
            parse_optional_bin_count(&Value::Int(IntValue::U32(1024))).unwrap(),
            Some(1024)
        );

        let bins = Tensor::new_integer(IntegerStorage::U64(vec![2048]), vec![1, 1])
            .expect("typed bin count");
        assert_eq!(
            parse_optional_bin_count(&Value::Tensor(bins)).unwrap(),
            Some(2048)
        );

        let negative = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
            .expect("negative bin count");
        assert!(parse_optional_bin_count(&Value::Tensor(negative)).is_err());
        assert!(parse_optional_bin_count(&Value::Int(IntValue::U64(MAX_BINS as u64 + 1))).is_err());

        for storage in [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ] {
            let mut bins = Tensor::new_integer(storage, vec![1, 1]).expect("typed bin count");
            bins.data.fill(f64::NAN);
            assert_eq!(
                parse_optional_bin_count(&Value::Tensor(bins)).unwrap(),
                Some(2)
            );
        }
    }

    #[test]
    fn bin_count_parser_rejects_unrepresentable_double_before_casting() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(parse_optional_bin_count(&Value::Num(boundary)).is_err());
    }

    #[test]
    fn rejects_out_of_range_floating_grayscale_values() {
        let image = tensor(vec![0.0, 35.0, 220.0, 255.0], vec![2, 2], NumericDType::F64);
        let err = block_on(imhist_builtin(Value::Tensor(image), vec![])).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:imhist:InvalidArgument"));
        assert!(err.message().contains("normalized to [0, 1]"));
    }

    #[test]
    fn rejects_nan_indexed_image_values() {
        let image = tensor(vec![1.0, f64::NAN], vec![1, 2], NumericDType::F64);
        let map = tensor(vec![0.0; 6], vec![2, 3], NumericDType::F64);
        let err = block_on(imhist_builtin(
            Value::Tensor(image),
            vec![Value::Tensor(map)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:imhist:InvalidArgument"));
        assert!(err.message().contains("finite integer indices"));
    }

    #[test]
    fn rejects_colormap_values_outside_unit_range() {
        let image = tensor(vec![1.0], vec![1, 1], NumericDType::F64);
        let map = tensor(vec![1.5, 0.0, 0.0], vec![1, 3], NumericDType::F64);
        let err = block_on(imhist_builtin(
            Value::Tensor(image),
            vec![Value::Tensor(map)],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:imhist:InvalidArgument"));
        assert!(err.message().contains("range [0, 1]"));
    }

    #[test]
    fn rejects_more_than_two_outputs() {
        let image = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
        let err = {
            let _guard = crate::output_count::push_output_count(Some(3));
            block_on(imhist_builtin(Value::Tensor(image), vec![])).unwrap_err()
        };
        assert_eq!(err.identifier(), Some("RunMat:imhist:TooManyOutputs"));
    }

    #[test]
    fn rejects_rgb_shaped_input() {
        let image = tensor(vec![0.0; 12], vec![2, 2, 3], NumericDType::F64);
        let err = block_on(imhist_builtin(Value::Tensor(image), vec![])).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:imhist:UnsupportedImage"));
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn plotting_downsamples_large_histograms_without_changing_outputs() {
        let image = tensor(vec![0.0, 65535.0], vec![1, 2], NumericDType::U16);
        let eval = block_on(evaluate(
            Value::Tensor(image),
            &[Value::Num((MAX_PLOT_BINS + 100) as f64)],
        ))
        .unwrap();
        assert_eq!(eval.counts.data.len(), MAX_PLOT_BINS + 100);
        let plot = plot_display_bins(&eval.counts, &eval.locations).unwrap();
        assert!(plot.counts.len() <= MAX_PLOT_BINS);
        assert_eq!(plot.counts.iter().sum::<f64>(), 2.0);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn statement_form_renders_bar_chart_without_value() {
        use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
        use crate::builtins::plotting::{
            clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
        };
        use runmat_plot::plots::PlotElement;

        let _guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        let _requested = crate::output_context::push_output_count(0);
        let image = tensor(vec![0.0, 1.0, 1.0, 2.0], vec![2, 2], NumericDType::U8);
        let out = block_on(imhist_builtin(Value::Tensor(image), vec![])).unwrap();
        assert_eq!(out, Value::OutputList(Vec::new()));
        if let Some(fig) = clone_figure(current_figure_handle()) {
            if let Some(plot) = fig.plots().next() {
                assert!(matches!(plot, PlotElement::Bar(_)));
            }
        }
    }

    #[cfg(not(feature = "plot-core"))]
    #[test]
    fn statement_form_noops_without_plot_core() {
        let _requested = crate::output_context::push_output_count(0);
        let image = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
        let out = block_on(imhist_builtin(Value::Tensor(image), vec![])).unwrap();
        assert_eq!(out, Value::OutputList(Vec::new()));
    }
}
