//! Form, option, numeric and residency admission for step metrics.

use super::*;
use crate::builtins::common::tensor;
use crate::builtins::common::validation::{
    native_integer_value_is_exact_f64, value_has_native_integer_class,
};
use crate::builtins::control::tf_model::{scalar_f64, scalar_text};

pub(super) enum Form {
    Model(Value),
    Samples(Box<Samples>),
}

pub(super) struct Samples {
    pub(super) y: Value,
    pub(super) time: Option<Value>,
    pub(super) y_final: Option<Value>,
}

pub(super) struct Request {
    pub(super) form: Form,
    pub(super) threshold: Value,
}

pub(super) fn preflight(
    input: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Request> {
    let is_model = matches!(input, Value::Object(_));
    let mut index = 0;
    let form = if is_model {
        preflight_response_model_operand(&input, ResponseKind::StepInfo)?;
        Form::Model(input)
    } else {
        preflight_numeric_role(&input, &STEPINFO_EXTENSIONS[0], "response")?;
        let time = rest.get(0).filter(|value| !is_option_name(value)).cloned();
        if let Some(value) = &time {
            preflight_numeric_role(value, &STEPINFO_EXTENSIONS[0], "time")?;
            index = 1;
        }
        let y_final = rest
            .get(index)
            .filter(|value| !is_option_name(value))
            .cloned();
        if let Some(value) = &y_final {
            if index == 0 {
                return Err(error("a steady-state value requires a time input"));
            }
            preflight_numeric_role(value, &STEPINFO_EXTENSIONS[0], "steady-state")?;
            index += 1;
        }
        // Validate every host-known role before downloading any other role.
        let host_y = host_vector(&input, "y")?;
        let host_t = time
            .as_ref()
            .map(|value| host_vector(value, "t"))
            .transpose()?
            .flatten();
        if let Some(value) = y_final
            .as_ref()
            .filter(|v| !matches!(v, Value::GpuTensor(_)))
        {
            let _ = final_scalar(value)?;
        }
        if let (Some(y), Some(t)) = (host_y, host_t) {
            metrics::validate_inputs(&y, &t)?;
        }
        Form::Samples(Box::new(Samples {
            y: input,
            time,
            y_final,
        }))
    };
    if is_model && rest.first().is_some_and(|value| !is_option_name(value)) {
        return Err(error("stepinfo(sys) accepts only name-value options"));
    }
    let threshold = parse_options(rest.view(), index)?;
    Ok(Request { form, threshold })
}

fn host_vector(value: &Value, role: &str) -> BuiltinResult<Option<Vec<f64>>> {
    if matches!(value, Value::GpuTensor(_)) {
        Ok(None)
    } else {
        numeric_vector(value.clone(), role).map(Some)
    }
}

fn parse_options(
    rest: crate::call::FunctionArgumentView<'_>,
    start: usize,
) -> BuiltinResult<Value> {
    if !(rest.len() - start).is_multiple_of(2) {
        return Err(error("name-value options must come in pairs"));
    }
    let mut threshold = Value::Num(DEFAULT_SETTLING_THRESHOLD);
    for pair in rest
        .range(start..)
        .expect("stepinfo option suffix")
        .chunks_exact(2)
    {
        let name = scalar_text(&pair[0], "option name", BUILTIN_NAME)?;
        if !name.trim().eq_ignore_ascii_case("SettlingTimeThreshold") {
            return Err(error(format!("unsupported option '{name}'")));
        }
        preflight_numeric_role(&pair[1], &STEPINFO_EXTENSIONS[1], "threshold")?;
        if !matches!(pair[1], Value::GpuTensor(_)) {
            let _ = threshold_scalar(&pair[1])?;
        }
        threshold = pair[1].clone();
    }
    Ok(threshold)
}

pub(super) async fn gather_threshold(request: &Request) -> BuiltinResult<f64> {
    let gathered = crate::dispatcher::gather_if_needed_async(&request.threshold).await?;
    exact_integer(&gathered, "threshold")?;
    threshold_scalar(&gathered)
}

pub(super) async fn gather_samples(
    y: Value,
    time: Option<Value>,
    y_final: Option<Value>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, f64)> {
    let y = gathered_vector(y, "y").await?;
    let t = match time {
        Some(value) => gathered_vector(value, "t").await?,
        None => (0..y.len()).map(|index| index as f64).collect(),
    };
    let y_final = match y_final {
        Some(value) => {
            let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
            exact_integer(&gathered, "steady-state")?;
            final_scalar(&gathered)?
        }
        None => *y
            .last()
            .ok_or_else(|| error("response vector cannot be empty"))?,
    };
    metrics::validate_inputs(&y, &t)?;
    Ok((y, t, y_final))
}

async fn gathered_vector(value: Value, role: &str) -> BuiltinResult<Vec<f64>> {
    let gathered = crate::dispatcher::gather_if_needed_async(&value).await?;
    exact_integer(&gathered, role)?;
    numeric_vector(gathered, role)
}

fn preflight_numeric_role(
    value: &Value,
    extension: &runmat_builtins::BuiltinExtensionDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    if !matches!(value, Value::GpuTensor(_)) {
        exact_integer(value, role)?;
    }
    Ok(())
}

fn exact_integer(value: &Value, role: &str) -> BuiltinResult<()> {
    if native_integer_value_is_exact_f64(value) {
        Ok(())
    } else {
        Err(error(format!(
            "{role} integer values must be exactly representable as double"
        )))
    }
}

fn is_option_name(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn numeric_vector(value: Value, role: &str) -> BuiltinResult<Vec<f64>> {
    let values = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(value) => vec![if value { 1.0 } else { 0.0 }],
        Value::Tensor(value) => {
            vector_shape(value.shape(), role)?;
            tensor::tensor_values_f64(&value)
        }
        Value::LogicalArray(value) => {
            vector_shape(&value.shape, role)?;
            value
                .data
                .iter()
                .map(|bit| if *bit == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        _ => return Err(error(format!("{role} must be a real numeric vector"))),
    };
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(error(format!("{role} must contain finite nonempty data")));
    }
    Ok(values)
}

fn vector_shape(shape: &[usize], role: &str) -> BuiltinResult<()> {
    if shape.iter().filter(|dim| **dim > 1).count() > 1 {
        Err(error(format!("{role} must be a vector")))
    } else {
        Ok(())
    }
}

fn final_scalar(value: &Value) -> BuiltinResult<f64> {
    let result = scalar_f64(value, "yfinal", BUILTIN_NAME)?;
    if !result.is_finite() {
        return Err(error("yfinal must be finite"));
    }
    Ok(result)
}

fn threshold_scalar(value: &Value) -> BuiltinResult<f64> {
    let result = scalar_f64(value, "SettlingTimeThreshold", BUILTIN_NAME)?;
    if !result.is_finite() || result <= 0.0 || result >= 1.0 {
        return Err(error(
            "SettlingTimeThreshold must lie strictly between zero and one",
        ));
    }
    Ok(result)
}

fn error(message: impl Into<String>) -> crate::RuntimeError {
    control_error(
        BUILTIN_NAME,
        "RunMat:stepinfo:InvalidData",
        format!("stepinfo: {}", message.into()),
    )
}
