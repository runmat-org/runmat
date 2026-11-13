use super::super::integer_bins;
use super::*;
use runmat_builtins::{HISTCOUNTS_ERROR_BINMETHOD_CONFLICT, HISTCOUNTS_ERROR_BINWIDTH_INVALID};
#[derive(Clone, Default)]
pub(super) struct HistcountsOptions {
    pub(super) explicit_edges: Option<Vec<f64>>,
    pub(super) exact_edges: Option<Vec<IntValue>>,
    pub(super) num_bins: Option<usize>,
    pub(super) bin_width: Option<f64>,
    pub(super) bin_limits: Option<(f64, f64)>,
    pub(super) bin_method: Option<BinMethod>,
    pub(super) normalization: HistogramNormalization,
}

impl HistcountsOptions {
    fn validate(&self) -> BuiltinResult<()> {
        if self.explicit_edges.is_some()
            && (self.num_bins.is_some() || self.bin_width.is_some() || self.bin_limits.is_some())
        {
            return Err(builtin_error(
                "histcounts: BinEdges cannot be combined with NumBins, BinWidth, or BinLimits",
            ));
        }
        if self.bin_method.is_some()
            && (self.explicit_edges.is_some()
                || self.bin_width.is_some()
                || self.num_bins.is_some())
        {
            return Err(descriptor_error(&HISTCOUNTS_ERROR_BINMETHOD_CONFLICT));
        }
        if self.num_bins.is_some() && self.bin_width.is_some() {
            return Err(builtin_error(
                "histcounts: specify only one of NumBins or BinWidth",
            ));
        }
        if let Some((lo, hi)) = self.bin_limits {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(builtin_error("histcounts: BinLimits must be finite"));
            }
            if hi < lo {
                return Err(builtin_error("histcounts: BinLimits must be increasing"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BinMethod {
    Auto,
    Scott,
    Fd,
    Sturges,
    Sqrt,
    Integers,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) enum HistogramNormalization {
    #[default]
    Count,
    Probability,
    CountDensity,
    Pdf,
    CumCount,
    Cdf,
}

pub(super) fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<HistcountsOptions> {
    let mut options = HistcountsOptions::default();
    let mut index = 0;

    if index < args.len() && !is_option_key(&args[index]) {
        match classify_bin_argument(&args[index])? {
            BinArgument::NumBins(n) => {
                options.num_bins = Some(n);
            }
            BinArgument::Edges(edges, exact) => {
                validate_bin_edges(&edges, exact.as_deref())?;
                options.explicit_edges = Some(edges);
                options.exact_edges = exact;
            }
        }
        index += 1;
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| builtin_error("histcounts: expected name/value pair arguments"))?;
        index += 1;
        if index >= args.len() {
            return Err(builtin_error(format!(
                "histcounts: missing value for option '{key}'"
            )));
        }
        let value = &args[index];
        index += 1;

        match runmat_builtins::HistcountsOption::parse(&key) {
            Some(runmat_builtins::HistcountsOption::BinEdges) => {
                let edges = numeric_vector(value, "histcounts", "BinEdges")?;
                let exact = integer_bins::exact_integer_vector(value);
                validate_bin_edges(&edges, exact.as_deref())?;
                options.explicit_edges = Some(edges);
                options.exact_edges = exact;
            }
            Some(runmat_builtins::HistcountsOption::NumBins) => {
                let scalar = positive_usize(value, "histcounts", "NumBins")?;
                options.num_bins = Some(scalar);
            }
            Some(runmat_builtins::HistcountsOption::BinWidth) => {
                let width = positive_scalar(value, "histcounts", "BinWidth")?;
                options.bin_width = Some(width);
            }
            Some(runmat_builtins::HistcountsOption::BinLimits) => {
                let limits = numeric_vector(value, "histcounts", "BinLimits")?;
                if limits.len() != 2 {
                    return Err(builtin_error(
                        "histcounts: BinLimits must contain exactly two elements",
                    ));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(builtin_error("histcounts: BinLimits must be increasing"));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(builtin_error("histcounts: BinLimits must be finite"));
                }
                options.bin_limits = Some((lo, hi));
            }
            Some(runmat_builtins::HistcountsOption::Normalization) => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error("histcounts: Normalization must be a string"))?;
                options.normalization = parse_normalization(&text)?;
            }
            Some(runmat_builtins::HistcountsOption::BinMethod) => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error("histcounts: BinMethod must be a string"))?;
                options.bin_method = Some(parse_bin_method(&text)?);
            }
            None => {
                return Err(builtin_error(format!(
                    "histcounts: unrecognised option '{}'",
                    key.trim().to_ascii_lowercase()
                )));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

#[derive(Debug)]
enum BinArgument {
    NumBins(usize),
    Edges(Vec<f64>, Option<Vec<IntValue>>),
}

fn classify_bin_argument(value: &Value) -> BuiltinResult<BinArgument> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let n = positive_usize(value, "histcounts", "NumBins")?;
            Ok(BinArgument::NumBins(n))
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(BinArgument::NumBins(positive_usize(
                    value,
                    "histcounts",
                    "NumBins",
                )?))
            } else {
                let edges = tensor::tensor_values_f64(tensor);
                Ok(BinArgument::Edges(
                    edges,
                    integer_bins::exact_integer_vector(value),
                ))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(BinArgument::NumBins(positive_usize_from_f64(
                    f64::from(logical.data[0]),
                    "histcounts",
                    "NumBins",
                )?))
            } else {
                Ok(BinArgument::Edges(
                    logical.data.iter().copied().map(f64::from).collect(),
                    None,
                ))
            }
        }
        Value::GpuTensor(_) => Err(build_runtime_error(
            "histcounts: bin specification cannot be a gpuArray",
        )
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
        .build()),
        other => Err(builtin_error(format!(
            "histcounts: unsupported bin specification {:?}",
            other
        ))),
    }
}

fn is_option_key(value: &Value) -> bool {
    if let Some(text) = tensor::value_to_string(value) {
        runmat_builtins::HistcountsOption::parse(&text).is_some()
    } else {
        false
    }
}

pub(super) fn numeric_vector(value: &Value, name: &str, option: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{name}: {option} must be numeric")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

pub(super) fn positive_usize(value: &Value, name: &str, option: &str) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value
            .try_to_usize()
            .filter(|value| *value > 0 && *value < usize::MAX)
            .ok_or_else(|| {
                builtin_error(format!("{name}: {option} must be a positive finite scalar"))
            });
    }
    let scalar = scalar_value(value, name, option)?;
    positive_usize_from_f64(scalar, name, option)
}

fn positive_usize_from_f64(scalar: f64, name: &str, option: &str) -> BuiltinResult<usize> {
    if scalar <= 0.0 || !scalar.is_finite() {
        return Err(builtin_error(format!(
            "{name}: {option} must be a positive finite scalar"
        )));
    }
    let rounded = scalar.round();
    if (scalar - rounded).abs() > f64::EPSILON {
        return Err(builtin_error(format!(
            "{name}: {option} must be an integer"
        )));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(builtin_error(format!(
            "{name}: {option} is outside the supported range"
        )));
    }
    Ok(rounded as usize)
}

fn positive_scalar(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
    let scalar = scalar_value(value, name, option)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        if name == BUILTIN_NAME && option == "BinWidth" {
            return Err(descriptor_error(&HISTCOUNTS_ERROR_BINWIDTH_INVALID));
        }
        return Err(builtin_error(format!(
            "{name}: {option} must be a positive finite scalar"
        )));
    }
    Ok(scalar)
}

pub(super) fn scalar_value(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
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

fn parse_bin_method(text: &str) -> BuiltinResult<BinMethod> {
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BinMethod::Auto),
        "scott" => Ok(BinMethod::Scott),
        "fd" | "freedmandiaconis" => Ok(BinMethod::Fd),
        "sturges" => Ok(BinMethod::Sturges),
        "sqrt" => Ok(BinMethod::Sqrt),
        "integers" => Ok(BinMethod::Integers),
        other => Err(builtin_error(format!(
            "histcounts: unrecognised BinMethod value '{other}'"
        ))),
    }
}

fn parse_normalization(text: &str) -> BuiltinResult<HistogramNormalization> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" => Ok(HistogramNormalization::Count),
        "probability" => Ok(HistogramNormalization::Probability),
        "countdensity" => Ok(HistogramNormalization::CountDensity),
        "pdf" | "probabilitydensity" => Ok(HistogramNormalization::Pdf),
        "cumcount" => Ok(HistogramNormalization::CumCount),
        "cdf" => Ok(HistogramNormalization::Cdf),
        other => Err(builtin_error(format!(
            "histcounts: unrecognised Normalization value '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests;
