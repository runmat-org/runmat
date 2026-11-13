//! One selected options generation with algorithm-specific typed controls.
use super::{common, invalid, ScalarCooperation, NAME};
use crate::BuiltinResult;
use runmat_value::Value;
pub(in crate::builtins::math::optim) mod admission;
mod operator;
mod preparation;
mod resolve;
mod publication;
mod ranges;
mod parameters;
mod nonlinear_creation;
#[cfg(test)]
mod tests;
pub(super) use nonlinear_creation::default as default_nonlinear_creation;
pub(super) use nonlinear_creation::Prepared as NonlinearCreation;
pub(super) use operator::Operator;
pub(super) use preparation::preflight_field;
pub(super) use preparation::Activity;
pub(super) use preparation::Prepared;
pub(super) use resolve::Options;
pub(super) use runmat_builtins::OptimOptionField as Field;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum Owner {
    Constructor,
    Solver,
}
impl Owner {
    pub fn name(self) -> &'static str {
        match self {
            Self::Constructor => "optimoptions",
            Self::Solver => NAME,
        }
    }
    pub fn policy(self) -> common::global_time::Policy {
        common::global_time::Policy {
            identity: runmat_builtins::BuiltinCatalogIdentity { name: self.name() },
            error: match self {
                Self::Constructor => &runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                Self::Solver => &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
            },
        }
    }
    pub fn invalid(self, detail: impl std::fmt::Display) -> crate::RuntimeError {
        let policy = self.policy();
        crate::build_runtime_error(format!("{}: {detail}", self.name()))
            .with_builtin(self.name())
            .with_identifier(policy.error.identifier.expect("canonical option error"))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    }
}
pub(super) fn text(value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .ok_or_else(|| invalid("option must contain scalar text"))
}
pub(super) fn real(owner: Owner, value: &Value) -> BuiltinResult<f64> {
    Ok(match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_exact_f64().ok_or_else(|| {
            owner.invalid("integer real control must be exactly representable as double")
        })?,
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(storage) = tensor.integer_storage() {
                storage
                    .value_at(0)
                    .and_then(|n| n.to_exact_f64())
                    .ok_or_else(|| {
                        owner
                            .invalid("integer real control must be exactly representable as double")
                    })?
            } else {
                crate::builtins::common::tensor::tensor_value_f64(tensor, 0)
            }
        }
        _ => return Err(owner.invalid("control must be a host real scalar")),
    })
}
pub(super) fn switch(owner: Owner, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::LogicalArray(values) if values.data.len() == 1 => Ok(values.data[0] != 0),
        value if crate::builtins::common::validation::value_text_scalar_string(value).is_some() => {
            runmat_builtins::OptimSwitch::parse(&text(value)?)
                .map(|v| v.enabled())
                .ok_or_else(|| owner.invalid("flag must be on/off or logical scalar"))
        }
        value => match real(owner, value)? {
            0. => Ok(false),
            1. => Ok(true),
            _ => Err(owner.invalid("flag must be logical scalar")),
        },
    }
}
