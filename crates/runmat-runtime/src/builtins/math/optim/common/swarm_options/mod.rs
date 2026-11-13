//! Selected swarm controls share one typed metadata/value owner between the
//! public constructor and the actual solver. This is not a global algorithm.
use crate::BuiltinResult;
pub(crate) use runmat_builtins::OptimOptionField as Field;
use runmat_value::{Tensor, Value};
mod metadata;
mod preparation;
mod time;
pub(crate) use metadata::{metadata, validate};
pub(crate) use preparation::{prepare, Prepared};
pub(crate) use time::{time_limit, TimeLimit};

#[derive(Clone, Copy)]
pub(crate) enum Owner {
    Constructor,
    Solver,
}
impl Owner {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Constructor => "optimoptions",
            Self::Solver => "particleswarm",
        }
    }
    pub(crate) fn time_policy(self) -> super::global_time::Policy {
        super::global_time::Policy {
            identity: runmat_builtins::BuiltinCatalogIdentity { name: self.name() },
            error: self.error(),
        }
    }
    fn error(self) -> &'static runmat_builtins::BuiltinErrorDescriptor {
        match self {
            Self::Constructor => &runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            Self::Solver => &runmat_builtins::PARTICLESWARM_ERROR_INVALID_ARGUMENT,
        }
    }
}
pub(crate) fn invalid(owner: Owner, message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(owner.name())
        .with_identifier(owner.error().identifier.expect("public option identifier"))
        .build()
}
/// Only synchronous, already-admitted scalar/grammar parsing is translated.
/// Provider, callback, cancellation and compatibility results never enter here.
pub(crate) fn own_parse<T>(owner: Owner, result: BuiltinResult<T>) -> BuiltinResult<T> {
    result.map_err(|source| {
        crate::build_runtime_error(format!("{}: invalid control", owner.name()))
            .with_builtin(owner.name())
            .with_identifier(owner.error().identifier.expect("public option identifier"))
            .with_source(source)
            .build()
    })
}
pub(crate) fn text(owner: Owner, value: &Value, field: Field) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .map(|v| v.to_owned())
        .ok_or_else(|| invalid(owner, format!("{} must be scalar text", field.name())))
}
pub(crate) fn switch(owner: Owner, value: &Value, field: Field) -> BuiltinResult<bool> {
    let numeric = |v: f64| match v {
        0.0 => Some(false),
        1.0 => Some(true),
        _ => None,
    };
    let parsed = match value {
        Value::Bool(v) => Some(*v),
        Value::LogicalArray(v) if v.data.len() == 1 => Some(v.data[0] != 0),
        Value::Num(v) => numeric(*v),
        Value::Int(v) => match v.try_to_u64() {
            Some(0) => Some(false),
            Some(1) => Some(true),
            _ => None,
        },
        Value::Tensor(v) if v.len() == 1 => {
            numeric(crate::builtins::common::tensor::tensor_value_f64(v, 0))
        }
        _ => crate::builtins::common::validation::value_text_scalar_string(value)
            .and_then(|s| runmat_builtins::OptimSwitch::parse(&s))
            .map(|s| s.enabled()),
    };
    parsed.ok_or_else(|| {
        invalid(
            owner,
            format!("{} must be a logical scalar or on/off switch", field.name()),
        )
    })
}
pub(crate) enum Floating<'a> {
    Scalar(f64),
    Tensor(&'a Tensor),
}
impl Floating<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(v) => v.len(),
        }
    }
    pub(crate) fn get(&self, index: usize) -> f64 {
        match self {
            Self::Scalar(v) => {
                assert_eq!(index, 0);
                *v
            }
            Self::Tensor(v) => crate::builtins::common::tensor::tensor_value_f64(v, index),
        }
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}
pub(crate) fn floating(owner: Owner, value: &Value) -> BuiltinResult<Floating<'_>> {
    if super::is_typed_integer_value(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(invalid(
            owner,
            "integer floating controls must be exactly representable as double",
        ));
    }
    match value {
        Value::Num(v) => Ok(Floating::Scalar(*v)),
        Value::Int(v) => Ok(Floating::Scalar(v.to_f64())),
        Value::Tensor(v) => Ok(Floating::Tensor(v)),
        _ => Err(invalid(
            owner,
            "numeric control must be host-visible after preparation",
        )),
    }
}
pub(crate) fn plot_form() -> super::output_callback::CallbackForm {
    const PLOT: &[runmat_builtins::BuiltinCatalogIdentity] =
        &[runmat_builtins::BuiltinCatalogIdentity {
            name: "pswplotbestf",
        }];
    super::output_callback::CallbackForm::HandlesOrPredefined(PLOT)
}
pub(crate) use super::global_hybrid::Hybrid;
pub(crate) fn hybrid(owner: Owner, value: &Value) -> BuiltinResult<Option<Hybrid>> {
    super::global_hybrid::parse(value, |detail: &str| invalid(owner, detail))
}
