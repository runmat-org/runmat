//! Public spelling is decoded once here; planning uses only these identities.
use super::super::admission::boxplot_matrix_dimensions;
use super::*;
use crate::builtins::common::grouping_keys::KeyAtom;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::stats::summary::boxplot) enum FactorDirection {
    Data,
    List,
    Auto,
}

impl FactorDirection {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "data" => Some(Self::Data),
            "list" => Some(Self::List),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    pub(super) fn resolved(self, levels: &[KeyAtom]) -> Self {
        match self {
            Self::Auto
                if levels
                    .iter()
                    .all(|key| matches!(key, KeyAtom::Number(_) | KeyAtom::Integer(_))) =>
            {
                Self::Data
            }
            Self::Auto => Self::List,
            explicit => explicit,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::builtins::stats::summary::boxplot) enum FactorGap {
    None,
    Auto,
    Percentages(Vec<f64>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::stats::summary::boxplot) enum FactorSeparator {
    None,
    Auto,
    Factors(Vec<usize>),
}

pub(in crate::builtins::stats::summary::boxplot) fn validate_direction_count(
    requested: &[FactorDirection],
    factor_count: usize,
) -> BuiltinResult<()> {
    if requested.len() != 1 && requested.len() != factor_count {
        return Err(invalid(
            "boxplot: FactorDirection length must match grouping factor count",
        ));
    }
    Ok(())
}

pub(in crate::builtins::stats::summary::boxplot) fn validate_gap_count(
    gap: &FactorGap,
    factor_count: usize,
) -> BuiltinResult<()> {
    if let FactorGap::Percentages(gaps) = gap {
        if gaps.len() > factor_count {
            return Err(invalid(
                "boxplot: FactorGap length exceeds grouping factor count",
            ));
        }
    }
    Ok(())
}

pub(in crate::builtins::stats::summary::boxplot) fn validate_separator_count(
    separator: &FactorSeparator,
    factor_count: usize,
) -> BuiltinResult<()> {
    if let FactorSeparator::Factors(indices) = separator {
        if indices.len() > factor_count || indices.iter().any(|&index| index > factor_count) {
            return Err(invalid(
                "boxplot: FactorSeparator index exceeds grouping factor count",
            ));
        }
    }
    Ok(())
}

pub(in crate::builtins::stats::summary::boxplot) fn parse_directions(
    value: &Value,
) -> BuiltinResult<Vec<FactorDirection>> {
    let values = super::super::options::text_vector(value, "FactorDirection")?;
    if values.is_empty() {
        return Err(invalid("boxplot: FactorDirection must not be empty"));
    }
    values
        .into_iter()
        .map(|value| {
            FactorDirection::parse(&value.to_ascii_lowercase()).ok_or_else(|| {
                invalid("boxplot: FactorDirection must be 'data', 'list', or 'auto'")
            })
        })
        .collect()
}

pub(in crate::builtins::stats::summary::boxplot) fn parse_full_factors(
    value: &Value,
) -> BuiltinResult<bool> {
    match super::super::options::text_scalar(value, "FullFactors")?.as_str() {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(invalid("boxplot: FullFactors must be 'on' or 'off'")),
    }
}

pub(in crate::builtins::stats::summary::boxplot) fn parse_gap(
    value: &Value,
) -> BuiltinResult<FactorGap> {
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        let text = super::super::options::text_scalar(value, "FactorGap")?;
        return if text == "auto" {
            Ok(FactorGap::Auto)
        } else {
            Err(invalid("boxplot: FactorGap must be numeric or 'auto'"))
        };
    }
    let tensor = crate::builtins::common::tensor::value_into_tensor_for(NAME, value.clone())
        .map_err(|err| invalid(format!("boxplot: FactorGap must be numeric ({err})")))?;
    ensure_vector(&tensor, "FactorGap")?;
    let values = crate::builtins::common::tensor::tensor_into_values_f64(tensor);
    if values.is_empty() {
        return Ok(FactorGap::None);
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "boxplot: FactorGap must contain nonnegative finite values",
        ));
    }
    Ok(FactorGap::Percentages(values))
}

pub(in crate::builtins::stats::summary::boxplot) fn parse_separator(
    value: &Value,
) -> BuiltinResult<FactorSeparator> {
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        let text = super::super::options::text_scalar(value, "FactorSeparator")?;
        return if text == "auto" {
            Ok(FactorSeparator::Auto)
        } else {
            Err(invalid(
                "boxplot: FactorSeparator must be numeric or 'auto'",
            ))
        };
    }
    let tensor = crate::builtins::common::tensor::value_into_tensor_for(NAME, value.clone())
        .map_err(|err| invalid(format!("boxplot: FactorSeparator must be numeric ({err})")))?;
    ensure_vector(&tensor, "FactorSeparator")?;
    let count = crate::builtins::common::tensor::tensor_element_len(&tensor);
    if count == 0 {
        return Ok(FactorSeparator::None);
    }
    // Structural indices never pass through binary64 when the input is an integer.
    let mut factors = Vec::with_capacity(count);
    for index in 0..count {
        let scalar = tensor
            .numeric_value_at(index)
            .ok_or_else(|| invalid("boxplot: malformed FactorSeparator storage"))?;
        let factor = if let Some(integer) = scalar.into_int_value() {
            integer.try_to_usize().filter(|value| *value > 0)
        } else {
            let floating = match scalar {
                NumericScalar::F64(value) => value,
                NumericScalar::F32(value) => f64::from(value),
                _ => unreachable!("integer scalar handled above"),
            };
            runmat_types::decode_shape_dimension(floating)
                .ok()
                .filter(|value| *value > 0)
        }
        .ok_or_else(|| invalid("boxplot: FactorSeparator must contain positive integer indices"))?;
        factors.push(factor);
    }
    Ok(FactorSeparator::Factors(factors))
}

fn ensure_vector(tensor: &Tensor, option: &str) -> BuiltinResult<()> {
    let (rows, cols) = boxplot_matrix_dimensions(tensor);
    if tensor.shape().len() > 2 || (rows > 1 && cols > 1) {
        Err(invalid(format!(
            "boxplot: {option} must be a scalar or vector"
        )))
    } else {
        Ok(())
    }
}
