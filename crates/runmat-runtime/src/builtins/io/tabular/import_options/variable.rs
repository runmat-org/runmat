use super::invalid;
use crate::BuiltinResult;
use runmat_value::{NumericScalar, Value};
use std::borrow::Cow;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EmptyFieldRule {
    Missing,
    Auto,
    Error,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WhitespaceRule {
    Trim,
    Preserve,
}

#[derive(Clone, Debug)]
pub(crate) struct VariableImportOptions {
    pub name: String,
    pub type_name: String,
    pub treat_as_missing: BTreeSet<String>,
    pub decimal_separator: Option<char>,
    pub thousands_separator: Option<Option<char>>,
    pub empty_field_rule: EmptyFieldRule,
    pub whitespace_rule: WhitespaceRule,
    pub fill_value: Option<Value>,
    pub input_format: Option<String>,
}

impl VariableImportOptions {
    pub(crate) fn has_value_controls(&self) -> bool {
        !self.treat_as_missing.is_empty()
            || self.decimal_separator.is_some()
            || self.thousands_separator.is_some()
            || self.empty_field_rule != EmptyFieldRule::Missing
            || self.whitespace_rule != WhitespaceRule::Trim
            || self.fill_value.is_some()
            || self.input_format.is_some()
    }

    pub(crate) fn new(name: String, type_name: String) -> Self {
        Self {
            name,
            type_name,
            treat_as_missing: BTreeSet::new(),
            decimal_separator: None,
            thousands_separator: None,
            empty_field_rule: EmptyFieldRule::Missing,
            whitespace_rule: WhitespaceRule::Trim,
            fill_value: None,
            input_format: None,
        }
    }

    pub(crate) fn is_missing(&self, token: &str) -> bool {
        self.treat_as_missing
            .contains(&token.trim().to_ascii_lowercase())
    }

    pub(crate) fn apply(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        if name.eq_ignore_ascii_case("TreatAsMissing") {
            self.treat_as_missing = super::selectors::text_list(value)?
                .into_iter()
                .map(|text| text.trim().to_ascii_lowercase())
                .collect();
        } else if name.eq_ignore_ascii_case("DecimalSeparator") {
            self.decimal_separator = Some(separator(value, name)?);
        } else if name.eq_ignore_ascii_case("ThousandsSeparator") {
            let text = super::selectors::scalar_text(value)?;
            self.thousands_separator = Some(if text.is_empty() {
                None
            } else {
                Some(separator(value, name)?)
            });
        } else if name.eq_ignore_ascii_case("EmptyFieldRule") {
            let rule = match super::selectors::scalar_text(value)?
                .to_ascii_lowercase()
                .as_str()
            {
                "missing" => EmptyFieldRule::Missing,
                "auto" => EmptyFieldRule::Auto,
                "error" => EmptyFieldRule::Error,
                _ => {
                    return Err(invalid(
                        "setvaropts: EmptyFieldRule must be missing, auto, or error",
                    ))
                }
            };
            if rule == EmptyFieldRule::Auto {
                return Err(invalid("setvaropts: EmptyFieldRule 'auto' is not supported by the current tabular reader"));
            }
            self.empty_field_rule = rule;
        } else if name.eq_ignore_ascii_case("WhitespaceRule") {
            let rule = match super::selectors::scalar_text(value)?
                .to_ascii_lowercase()
                .as_str()
            {
                "trim" => WhitespaceRule::Trim,
                "preserve" => WhitespaceRule::Preserve,
                _ => {
                    return Err(invalid(
                        "setvaropts: WhitespaceRule must be trim or preserve",
                    ))
                }
            };
            if rule == WhitespaceRule::Preserve {
                return Err(invalid("setvaropts: WhitespaceRule 'preserve' is not supported by the current tabular reader"));
            }
            self.whitespace_rule = rule;
        } else if name.eq_ignore_ascii_case("FillValue") {
            self.fill_value = Some(value.clone());
        } else if name.eq_ignore_ascii_case("InputFormat") {
            let format = super::selectors::scalar_text(value)?;
            if !crate::builtins::common::validation::supported_datetime_input_format(&format) {
                return Err(invalid("setvaropts: unsupported datetime InputFormat"));
            }
            self.input_format = Some(format);
        } else {
            return Err(invalid(format!(
                "setvaropts: unsupported variable option '{name}'"
            )));
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> BuiltinResult<()> {
        if self.thousands_separator.flatten() == self.decimal_separator
            && self.decimal_separator.is_some()
        {
            return Err(invalid(
                "setvaropts: DecimalSeparator and ThousandsSeparator must differ",
            ));
        }
        Ok(())
    }

    pub(crate) fn numeric(&self, token: &str) -> Option<f64> {
        let normalized = normalize_numeric_token(
            token,
            self.decimal_separator.unwrap_or('.'),
            self.thousands_separator.flatten(),
        );
        match normalized.to_ascii_lowercase().as_str() {
            "nan" => Some(f64::NAN),
            "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
            "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
            _ => normalized.parse::<f64>().ok(),
        }
    }

    pub(crate) fn numeric_fill(&self) -> BuiltinResult<Option<NumericScalar>> {
        let Some(value) = &self.fill_value else {
            return Ok(None);
        };
        match value {
            Value::Num(number) => Ok(Some(NumericScalar::F64(*number))),
            Value::Int(integer) => Ok(Some(NumericScalar::from(integer.clone()))),
            Value::Bool(flag) => Ok(Some(NumericScalar::F64(f64::from(*flag)))),
            Value::Tensor(tensor) if tensor.len() == 1 => Ok(tensor.numeric_value_at(0)),
            _ => Err(invalid(
                "setvaropts: numeric FillValue must be a numeric scalar",
            )),
        }
    }
}

fn separator(value: &Value, name: &str) -> BuiltinResult<char> {
    let text = super::selectors::scalar_text(value)?;
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) if ch != '\n' && ch != '\r' => Ok(ch),
        _ => Err(invalid(format!("setvaropts: {name} must be one character"))),
    }
}

pub(crate) fn normalize_numeric_token<'a>(
    token: &'a str,
    decimal_separator: char,
    thousands_separator: Option<char>,
) -> Cow<'a, str> {
    let mut normalized = Cow::Borrowed(token);
    if let Some(thousands) = thousands_separator {
        if thousands != decimal_separator && normalized.contains(thousands) {
            normalized = Cow::Owned(normalized.chars().filter(|ch| *ch != thousands).collect());
        }
    }
    if decimal_separator != '.' && normalized.contains(decimal_separator) {
        normalized = Cow::Owned(normalized.replace(decimal_separator, "."));
    }
    normalized
}
