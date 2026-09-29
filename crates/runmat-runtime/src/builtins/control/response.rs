//! Shared admission for real SISO response simulation.
//!
//! Transfer-function structure belongs to `tf_model`; this boundary checks
//! exact integer conversion and gathers only properties that the simulator
//! consumes. Time grids and response equations remain with each operation.
use crate::builtins::control::tf_model::{
    preflight_model_operand, scalar_f64, RealTfModel, TfModel,
};
use crate::{build_runtime_error, dispatcher, BuiltinResult};
use runmat_value::{NumericScalar, Value};

#[derive(Clone, Copy)]
pub(crate) enum ResponseKind {
    Step,
    Impulse,
}

impl ResponseKind {
    fn name(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::Impulse => "impulse",
        }
    }

    fn integer_error(self) -> &'static str {
        match self {
            Self::Step => "RunMat:step:InvalidModel",
            Self::Impulse => "RunMat:impulse:InvalidArgument",
        }
    }
}

pub(crate) async fn parse_real_model(
    value: Value,
    kind: ResponseKind,
) -> BuiltinResult<RealTfModel> {
    let builtin = kind.name();
    if !matches!(value, Value::Object(_)) {
        return TfModel::from_value(value, builtin).and_then(|model| model.to_real(builtin));
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
            .properties
            .get(name)
            .expect("preflight required property");
        if matches!(value, Value::GpuTensor(_)) {
            let gathered = dispatcher::gather_if_needed_async(value).await?;
            ensure_exact_integer(&gathered, name, kind)?;
            object.properties.insert(name.into(), gathered);
        }
    }
    let model = TfModel::from_value_async(Value::Object(object), builtin).await?;
    let real = model.to_real(builtin)?;
    real.ensure_zero_delays(builtin)?;
    Ok(real)
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
            .properties
            .get(name)
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
        {
            ensure_exact_integer(value, name, kind)?;
        }
    }
    for name in ["InputDelay", "OutputDelay"] {
        if let Some(value) = object
            .properties
            .get(name)
            .filter(|value| !matches!(value, Value::GpuTensor(_)))
        {
            if scalar_f64(value, name, builtin)? != 0.0 {
                return Err(build_runtime_error(format!(
                    "{builtin}: transfer functions with input or output delays are not supported"
                ))
                .with_builtin(builtin)
                .with_identifier(match kind {
                    ResponseKind::Step => "RunMat:step:UnsupportedModel",
                    ResponseKind::Impulse => "RunMat:impulse:UnsupportedModel",
                })
                .build());
            }
        }
    }
    Ok(())
}

fn ensure_exact_integer(value: &Value, name: &str, kind: ResponseKind) -> BuiltinResult<()> {
    let exact = match value {
        Value::Int(value) => value.to_i128().unsigned_abs() <= (1_u128 << 53),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            (0..tensor.len()).all(|index| tensor.numeric_value_at(index).is_some_and(exact_scalar))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => (0..tensor.len())
            .all(|index| {
                tensor
                    .numeric_value_at(index)
                    .is_some_and(|(real, imaginary)| exact_scalar(real) && exact_scalar(imaginary))
            }),
        _ => true,
    };
    if exact {
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

fn exact_scalar(value: NumericScalar) -> bool {
    value
        .into_int_value()
        .is_none_or(|integer| integer.to_i128().unsigned_abs() <= (1_u128 << 53))
}
