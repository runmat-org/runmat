use super::super::{invalid, numeric_vector, positive_integer, positive_scalar, scalar_text};
use super::{OptionName, Options};
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::stats::ml::mnrfit) fn parse_options(
    values: Vec<(OptionName, Value)>,
) -> BuiltinResult<Options> {
    let mut options = Options::default();
    for (name, value) in &values {
        match name {
            OptionName::Model => {
                let model = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Model option must be text"))?;
                match model.to_ascii_lowercase().as_str() {
                    "nominal" => {}
                    other => {
                        return Err(invalid(format!(
                            "mnrfit: Model '{other}' is not supported yet"
                        )));
                    }
                }
            }
            OptionName::Link => {
                let link = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Link option must be text"))?;
                match link.to_ascii_lowercase().as_str() {
                    "logit" => {}
                    other => {
                        return Err(invalid(format!(
                            "mnrfit: Link '{other}' is not supported yet"
                        )));
                    }
                }
            }
            OptionName::Constant => {
                options.constant = match text_or_logical(value, "Constant")? {
                    BoolOrText::Bool(flag) => flag,
                    BoolOrText::Text(text) => match text.to_ascii_lowercase().as_str() {
                        "on" => true,
                        "off" => false,
                        other => {
                            return Err(invalid(format!(
                                "mnrfit: Constant must be 'on' or 'off', got '{other}'"
                            )));
                        }
                    },
                };
            }
            OptionName::Interactions => {
                let text = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Interactions option must be text"))?;
                if !text.eq_ignore_ascii_case("off") {
                    return Err(invalid(
                        "mnrfit: Interactions values other than 'off' are not supported yet",
                    ));
                }
            }
            OptionName::Weights => {
                options.weights = Some(numeric_vector(value.clone(), "Weights")?)
            }
            OptionName::MaxIter | OptionName::IterationLimit => {
                options.max_iter = positive_integer(value, "IterationLimit")?
            }
            OptionName::TolX | OptionName::Tolerance => {
                options.tol_x = positive_scalar(value, "Tolerance")?
            }
            OptionName::Options => apply_statset_options(value, &mut options)?,
            OptionName::Display => {
                let _ = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: Display option must be text"))?;
            }
            OptionName::EstDisp => {
                let text = scalar_text(value)
                    .ok_or_else(|| invalid("mnrfit: EstDisp option must be text"))?;
                if !text.eq_ignore_ascii_case("off") {
                    return Err(invalid(
                        "mnrfit: overdispersion estimation is not supported yet",
                    ));
                }
            }
        }
    }
    if options.max_iter == 0 {
        return Err(invalid("mnrfit: IterationLimit must be positive"));
    }
    if !options.tol_x.is_finite() || options.tol_x <= 0.0 {
        return Err(invalid("mnrfit: Tolerance must be positive and finite"));
    }
    Ok(options)
}

fn apply_statset_options(value: &Value, options: &mut Options) -> BuiltinResult<()> {
    let Value::Struct(st) = value else {
        return Err(invalid("mnrfit: Options must be a struct"));
    };
    if let Some(value) = st.fields.get("MaxIter") {
        options.max_iter = positive_integer(value, "Options.MaxIter")?;
    }
    if let Some(value) = st.fields.get("IterationLimit") {
        options.max_iter = positive_integer(value, "Options.IterationLimit")?;
    }
    if let Some(value) = st.fields.get("TolX") {
        options.tol_x = positive_scalar(value, "Options.TolX")?;
    }
    if let Some(value) = st.fields.get("Tolerance") {
        options.tol_x = positive_scalar(value, "Options.Tolerance")?;
    }
    Ok(())
}

enum BoolOrText {
    Bool(bool),
    Text(String),
}

fn text_or_logical(value: &Value, label: &str) -> BuiltinResult<BoolOrText> {
    if let Some(text) = scalar_text(value) {
        return Ok(BoolOrText::Text(text));
    }
    match value {
        Value::Bool(flag) => Ok(BoolOrText::Bool(*flag)),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(BoolOrText::Bool(logical.data[0] != 0))
        }
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(BoolOrText::Bool(*value != 0.0)),
        _ => Err(invalid(format!("mnrfit: {label} must be text or logical"))),
    }
}
