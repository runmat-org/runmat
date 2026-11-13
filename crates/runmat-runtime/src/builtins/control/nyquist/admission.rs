//! Mode, shape and exactness admission before control-model provider reads.

use super::*;
use crate::builtins::common::validation::{
    native_integer_value_is_exact_f64, value_has_native_integer_class,
};
use crate::builtins::control::tf_model::preflight_model_operand;

const NUMERIC_PROPERTIES: [&str; 5] = [
    "Numerator",
    "Denominator",
    "Ts",
    "InputDelay",
    "OutputDelay",
];

pub(super) fn preflight(system: &Value, frequency: Option<&Value>) -> BuiltinResult<()> {
    let Value::Object(object) = system else {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidModel",
            "nyquist: expected a scalar tf model",
        ));
    };
    preflight_model_operand(system, BUILTIN_NAME)?;
    for name in NUMERIC_PROPERTIES {
        let value = object
            .properties()
            .get(name)
            .expect("preflight required property");
        gate_numeric_role(value)?;
        if !matches!(value, Value::GpuTensor(_)) {
            exact_integer(value, name)?;
        }
    }
    if object
        .properties()
        .get("Variable")
        .is_some_and(|value| matches!(value, Value::GpuTensor(_)))
    {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidModel",
            "nyquist: transfer-function Variable must be host text",
        ));
    }
    if let Some(value) = frequency {
        gate_numeric_role(value)?;
        if !matches!(value, Value::GpuTensor(_)) {
            exact_integer(value, "frequency")?;
            super::frequency::validate_host(value)?;
        }
    }
    for name in ["InputDelay", "OutputDelay"] {
        if let Some(value) = object
            .properties()
            .get(name)
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
        {
            let delay = crate::builtins::control::tf_model::scalar_f64(value, name, BUILTIN_NAME)?;
            if delay != 0.0 {
                return Err(control_error(
                    BUILTIN_NAME,
                    "RunMat:nyquist:UnsupportedModel",
                    "nyquist: nonzero transfer-function delays are not supported",
                ));
            }
        }
    }
    Ok(())
}

pub(super) async fn parse_model(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<TfModel> {
    let Value::Object(mut object) = value else {
        unreachable!("preflight requires tf object")
    };
    for name in NUMERIC_PROPERTIES {
        if object
            .properties()
            .get(name)
            .is_some_and(|value| matches!(value, Value::GpuTensor(_)))
        {
            let gathered = crate::dispatcher::gather_if_needed_async(
                object.properties().get(name).expect("required"),
            )
            .await?;
            exact_integer(&gathered, name)?;
            crate::execution::object::replacement::replace_property(
                runtime,
                &mut object,
                name,
                gathered,
                |detail, source| {
                    crate::build_runtime_error(format!("{BUILTIN_NAME}: {detail}"))
                        .with_builtin(BUILTIN_NAME)
                        .with_source(source)
                        .build()
                },
                |detail| {
                    crate::build_runtime_error(detail)
                        .with_builtin(BUILTIN_NAME)
                        .build()
                },
            )?;
        }
    }
    let model = TfModel::from_value_async(Value::Object(object), BUILTIN_NAME).await?;
    if model.input_delay != 0.0 || model.output_delay != 0.0 {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:UnsupportedModel",
            "nyquist: nonzero transfer-function delays are not supported",
        ));
    }
    Ok(model)
}

fn gate_numeric_role(value: &Value) -> BuiltinResult<()> {
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NYQUIST_EXTENSIONS[0],
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NYQUIST_EXTENSIONS[1],
            BUILTIN_NAME,
        )?;
    }
    if matches!(value, Value::GpuTensor(_)) {
        gate_resident()?;
    }
    Ok(())
}

fn gate_resident() -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(&NYQUIST_EXTENSIONS[2], BUILTIN_NAME)
}

fn exact_integer(value: &Value, role: &str) -> BuiltinResult<()> {
    if native_integer_value_is_exact_f64(value) {
        Ok(())
    } else {
        Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidArgument",
            format!(
                "nyquist: {role} native integer values must be exactly representable as double"
            ),
        ))
    }
}

pub(super) fn exact_gathered_frequency(value: &Value) -> BuiltinResult<()> {
    exact_integer(value, "frequency")
}
