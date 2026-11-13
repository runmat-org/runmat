use super::super::integer_bins::{exact_integer_vector, validate_exact_edges};
use super::edges::validate_edges;
use super::*;
use runmat_builtins::{HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT, HISTCOUNTS2_ERROR_NUMBINS_INVALID};
#[derive(Clone, Default)]
pub(super) struct Histcounts2Options {
    pub(super) x: AxisOptions,
    pub(super) y: AxisOptions,
    pub(super) normalization: HistogramNormalization,
}

impl Histcounts2Options {
    fn validate(&self) -> BuiltinResult<()> {
        self.x.validate("X")?;
        self.y.validate("Y")?;
        Ok(())
    }
}

#[derive(Clone, Default)]
pub(super) struct AxisOptions {
    pub(super) explicit_edges: Option<Vec<f64>>,
    pub(super) exact_edges: Option<Vec<IntValue>>,
    pub(super) num_bins: Option<usize>,
    pub(super) bin_width: Option<f64>,
    pub(super) bin_limits: Option<(f64, f64)>,
    pub(super) bin_method: Option<BinMethod>,
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
    Percentage,
    CountDensity,
    Pdf,
    CumCount,
    Cdf,
}

pub(super) fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Histcounts2Options> {
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
        match Histcounts2Option::parse(&key) {
            Some(Histcounts2Option::NumBins) => {
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
            Some(Histcounts2Option::XBinWidth) => {
                let width = positive_scalar(value, NAME, "XBinWidth")?;
                options.x.bin_width = Some(width);
            }
            Some(Histcounts2Option::YBinWidth) => {
                let width = positive_scalar(value, NAME, "YBinWidth")?;
                options.y.bin_width = Some(width);
            }
            Some(Histcounts2Option::BinWidth) => {
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
            Some(Histcounts2Option::XBinLimits) => {
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
            Some(Histcounts2Option::YBinLimits) => {
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
            Some(Histcounts2Option::BinLimits) => {
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
            Some(Histcounts2Option::XBinEdges) => {
                let edges = numeric_vector(value, NAME, "XBinEdges")?;
                validate_edge_value(value, &edges, "X")?;
                options.x.explicit_edges = Some(edges);
                options.x.exact_edges = exact_integer_vector(value);
            }
            Some(Histcounts2Option::YBinEdges) => {
                let edges = numeric_vector(value, NAME, "YBinEdges")?;
                validate_edge_value(value, &edges, "Y")?;
                options.y.explicit_edges = Some(edges);
                options.y.exact_edges = exact_integer_vector(value);
            }
            Some(Histcounts2Option::BinMethod) => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: BinMethod must be a string")))?;
                let method = parse_bin_method(&text)?;
                options.x.bin_method = Some(method);
                options.y.bin_method = Some(method);
            }
            Some(Histcounts2Option::XBinMethod) => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: XBinMethod must be a string")))?;
                options.x.bin_method = Some(parse_bin_method(&text)?);
            }
            Some(Histcounts2Option::YBinMethod) => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: YBinMethod must be a string")))?;
                options.y.bin_method = Some(parse_bin_method(&text)?);
            }
            Some(Histcounts2Option::Normalization) => {
                let text = tensor::value_to_string(value).ok_or_else(|| {
                    builtin_error(format!("{NAME}: Normalization must be a string"))
                })?;
                options.normalization = parse_normalization(&text)?;
            }
            None => {
                return Err(builtin_error(format!(
                    "{NAME}: unrecognised option '{}'",
                    key.trim().to_ascii_lowercase()
                )));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

pub(super) fn is_option_key(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

pub(super) fn numeric_vector(value: &Value, name: &str, option: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{name}: {option} must be numeric")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn validate_edge_value(value: &Value, floating: &[f64], axis: &str) -> BuiltinResult<()> {
    if let Some(exact) = exact_integer_vector(value) {
        if exact.len() < 2 {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must contain at least two elements"
            )));
        }
        if !validate_exact_edges(&exact) {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must be strictly increasing"
            )));
        }
        Ok(())
    } else {
        validate_edges(floating, axis)
    }
}

pub(super) fn num_bins_vector(value: &Value) -> BuiltinResult<Vec<usize>> {
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

pub(super) fn positive_usize(value: &Value, name: &str, option: &str) -> BuiltinResult<usize> {
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
