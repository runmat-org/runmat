use runmat_builtins::{
    BSXFUN_ERROR_FUNCTION_ERROR, BSXFUN_ERROR_INTERNAL, BSXFUN_ERROR_INVALID_FUNCTION,
};
use runmat_value::{NumericDType, Value};

use super::input::ArrayInput;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum OutputContract {
    #[default]
    Dynamic,
    Logical,
    Numeric(NumericDType),
    Complex(NumericDType),
    Character,
}

impl OutputContract {
    pub(super) fn infer(callable: &Value, left: &ArrayInput, right: &ArrayInput) -> Self {
        let Some(inference) = crate::call::catalog::infer_builtin_callback(
            callable,
            vec![left.scalar_fact(), right.scalar_fact()],
            1,
        ) else {
            return Self::Dynamic;
        };
        if inference
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == runmat_types::InferenceSeverity::Error)
        {
            return Self::Dynamic;
        }
        match inference.outputs.first().map(|output| &output.kind) {
            Some(runmat_types::ValueKindFact::Logical) => Self::Logical,
            Some(runmat_types::ValueKindFact::Character) => Self::Character,
            Some(runmat_types::ValueKindFact::Numeric(numeric)) => match numeric.domain {
                runmat_types::NumericDomain::Real => Self::Numeric(numeric.class.into()),
                runmat_types::NumericDomain::Complex => Self::Complex(numeric.class.into()),
            },
            _ => Self::Dynamic,
        }
    }

    pub(super) fn normalize(self, value: Value) -> crate::BuiltinResult<Value> {
        if self != Self::Logical {
            return Ok(value);
        }
        match value {
            Value::Bool(_) | Value::LogicalArray(_) => Ok(value),
            Value::Num(value) => Ok(Value::Bool(value != 0.0)),
            Value::Int(value) => Ok(Value::Bool(!value.is_zero())),
            Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(&tensor) => {
                Ok(Value::Bool(
                    !tensor
                        .numeric_value_at(0)
                        .ok_or_else(|| super::error::from_descriptor(&BSXFUN_ERROR_INTERNAL))?
                        .is_zero(),
                ))
            }
            other => Err(super::error::detail(
                &BSXFUN_ERROR_FUNCTION_ERROR,
                Some(format!(
                    "logical callback must return scalar logical values (got {other:?})"
                )),
            )),
        }
    }
}

pub(super) fn validate(value: &Value) -> crate::BuiltinResult<()> {
    match value {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_) => Ok(()),
        other => Err(super::error::detail(
            &BSXFUN_ERROR_INVALID_FUNCTION,
            Some(format!("got {other:?}")),
        )),
    }
}
