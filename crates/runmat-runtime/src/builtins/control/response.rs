//! Shared admission for real SISO response simulation.
//!
//! Transfer-function structure belongs to `tf_model`; this boundary checks
//! exact integer conversion and gathers only properties that the simulator
//! consumes. Time grids and response equations remain with each operation.
use crate::builtins::common::tensor::complex_tensor_values_complex64;
use crate::builtins::common::validation::native_integer_value_is_exact_f64;
use crate::builtins::control::tf_model::{
    preflight_model_operand, scalar_f64, RealTfModel, TfModel,
};
use crate::{build_runtime_error, dispatcher, BuiltinResult};
use runmat_value::Value;

pub(crate) mod step;

#[derive(Clone, Copy)]
pub(crate) enum ResponseKind {
    Step,
    Impulse,
    StepInfo,
}

impl ResponseKind {
    fn name(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::Impulse => "impulse",
            Self::StepInfo => "stepinfo",
        }
    }

    fn integer_error(self) -> &'static str {
        match self {
            Self::Step => "RunMat:step:InvalidModel",
            Self::Impulse => "RunMat:impulse:InvalidArgument",
            Self::StepInfo => "RunMat:stepinfo:InvalidSystem",
        }
    }

    fn unsupported_model_error(self) -> &'static str {
        match self {
            Self::Step => "RunMat:step:UnsupportedModel",
            Self::Impulse => "RunMat:impulse:UnsupportedModel",
            Self::StepInfo => "RunMat:stepinfo:UnsupportedModel",
        }
    }
}

pub(crate) struct ParsedResponseModel {
    pub model: TfModel,
    pub real: RealTfModel,
}

pub(crate) async fn parse_real_model(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    kind: ResponseKind,
) -> BuiltinResult<RealTfModel> {
    Ok(parse_admitted_model(runtime, value, kind).await?.real)
}

pub(crate) async fn parse_admitted_model(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    kind: ResponseKind,
) -> BuiltinResult<ParsedResponseModel> {
    let builtin = kind.name();
    if !matches!(value, Value::Object(_)) {
        let model = TfModel::from_value(value, builtin)?;
        let real = model.to_real(builtin)?;
        real.ensure_zero_delays(builtin)?;
        return Ok(ParsedResponseModel { model, real });
    }
    preflight_response_model_operand(&value, kind)?;
    let Value::Object(mut object) = value else {
        unreachable!()
    };
    const CONSUMED: [&str; 5] = [
        "Numerator",
        "Denominator",
        "Ts",
        "InputDelay",
        "OutputDelay",
    ];
    for name in CONSUMED {
        let value = object
            .properties()
            .get(name)
            .expect("preflight required property");
        if matches!(value, Value::GpuTensor(_)) {
            let gathered = dispatcher::gather_if_needed_async(value).await?;
            ensure_exact_integer(&gathered, name, kind)?;
            if matches!(name, "Numerator" | "Denominator") {
                preflight_real_coefficients(&gathered, name, kind)?;
            }
            crate::execution::object::replacement::replace_property(
                runtime,
                &mut object,
                name,
                gathered,
                |detail, source| {
                    build_runtime_error(format!("{builtin}: {detail}"))
                        .with_builtin(builtin)
                        .with_source(source)
                        .build()
                },
                |detail| build_runtime_error(detail).with_builtin(builtin).build(),
            )?;
        }
    }
    let model = TfModel::from_value_async(Value::Object(object), builtin).await?;
    let real = model.to_real(builtin)?;
    real.ensure_zero_delays(builtin)?;
    Ok(ParsedResponseModel { model, real })
}

/// Reject every host-known error that can invalidate a response model before
/// time or coefficient downloads. The shared tf parser owns model structure;
/// this operation boundary additionally disallows nonzero delays and lossy
/// integer conversion for the host-only response equations.
pub(crate) fn preflight_response_model_operand(
    value: &Value,
    kind: ResponseKind,
) -> BuiltinResult<()> {
    let builtin = kind.name();
    preflight_model_operand(value, builtin)?;
    let Value::Object(object) = value else {
        return Ok(());
    };
    for name in [
        "Numerator",
        "Denominator",
        "Ts",
        "InputDelay",
        "OutputDelay",
    ] {
        if let Some(value) = object
            .properties()
            .get(name)
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
        {
            ensure_exact_integer(value, name, kind)?;
            if matches!(name, "Numerator" | "Denominator") {
                preflight_real_coefficients(value, name, kind)?;
            }
        }
    }
    for name in ["InputDelay", "OutputDelay"] {
        if let Some(value) = object
            .properties()
            .get(name)
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
        {
            if scalar_f64(value, name, builtin)? != 0.0 {
                return Err(build_runtime_error(format!(
                    "{builtin}: transfer functions with input or output delays are not supported"
                ))
                .with_builtin(builtin)
                .with_identifier(kind.unsupported_model_error())
                .build());
            }
        }
    }
    Ok(())
}

fn ensure_exact_integer(value: &Value, name: &str, kind: ResponseKind) -> BuiltinResult<()> {
    if native_integer_value_is_exact_f64(value) {
        Ok(())
    } else {
        Err(build_runtime_error(format!(
            "{}: {name} integer values must be exactly representable as double",
            kind.name()
        ))
        .with_builtin(kind.name())
        .with_identifier(kind.integer_error())
        .build())
    }
}

fn preflight_real_coefficients(value: &Value, name: &str, kind: ResponseKind) -> BuiltinResult<()> {
    let nonreal = match value {
        Value::Complex(_, imaginary) => *imaginary != 0.0,
        Value::ComplexTensor(tensor) => complex_tensor_values_complex64(tensor)
            .iter()
            .any(|coefficient| coefficient.im != 0.0),
        _ => false,
    };
    if nonreal {
        Err(build_runtime_error(format!(
            "{}: complex tf {name} coefficients are not supported",
            kind.name()
        ))
        .with_builtin(kind.name())
        .with_identifier(kind.unsupported_model_error())
        .build())
    } else {
        Ok(())
    }
}
