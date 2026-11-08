//! Unit-preserving nonnegative limits; finite duration never becomes fake Inf.
use crate::BuiltinResult;
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::Value;
use std::time::Duration;
#[derive(Clone, Copy)]
pub(crate) struct Policy {
    pub identity: BuiltinCatalogIdentity,
    pub error: &'static BuiltinErrorDescriptor,
}
impl Policy {
    fn invalid(self, message: &str) -> crate::RuntimeError {
        let mut builder = crate::build_runtime_error(message).with_builtin(self.identity.name);
        if let Some(identifier) = self.error.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}
#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub(crate) enum TimeLimit {
    #[default]
    Unbounded,
    Seconds(f64),
    Days(f64),
}
impl TimeLimit {
    pub(crate) fn is_zero(self) -> bool {
        matches!(self, Self::Seconds(0.0) | Self::Days(0.0))
    }
    pub(crate) fn validate(self, policy: Policy) -> BuiltinResult<Self> {
        let v = match self {
            Self::Unbounded => return Ok(self),
            Self::Seconds(v) | Self::Days(v) => v,
        };
        if v.is_nan() || v < 0.0 {
            return Err(policy.invalid("time limits must be nonnegative"));
        }
        Ok(if v == f64::INFINITY {
            Self::Unbounded
        } else {
            self
        })
    }
    /// A unit-preserving elapsed fraction. Unbounded limits have no fraction.
    /// Zero is already reached; callers decide whether to clamp display values.
    #[cfg_attr(not(any(test, feature = "plot-core")), allow(dead_code))]
    pub(crate) fn fraction(self, elapsed: Duration) -> Option<f64> {
        match self {
            Self::Unbounded => None,
            Self::Seconds(0.0) | Self::Days(0.0) => Some(1.0),
            Self::Seconds(v) => Some(elapsed.as_secs_f64() / v),
            Self::Days(v) => Some((elapsed.as_secs_f64() / 86400.0) / v),
        }
    }
    pub(crate) fn reached(self, elapsed: Duration) -> bool {
        match self {
            Self::Unbounded => false,
            Self::Seconds(v) => elapsed.as_secs_f64() >= v,
            Self::Days(v) => elapsed.as_secs_f64() / 86400.0 >= v,
        }
    }
}
pub(crate) fn parse(policy: Policy, value: &Value) -> BuiltinResult<TimeLimit> {
    let limit = if crate::builtins::duration::is_duration_object(value) {
        let tensor = crate::builtins::duration::duration_tensor_from_duration_value(value)?;
        if tensor.len() != 1 || tensor.as_f64_slice().is_none() {
            return Err(policy.invalid("time limit duration must be scalar double"));
        }
        TimeLimit::Days(tensor.as_f64_slice().expect("double duration")[0])
    } else {
        super::preflight_option_scalar(
            policy.identity.name,
            value,
            runmat_builtins::OptimOptionField::MaxTime,
        )?;
        let seconds = match value {
            Value::Num(v) => *v,
            Value::Tensor(v) if v.len() == 1 && v.integer_storage().is_none() => {
                crate::builtins::common::tensor::tensor_value_f64(v, 0)
            }
            _ => super::option_f64_value(policy.identity.name, Some(value), "time limit", 0.0)?,
        };
        TimeLimit::Seconds(seconds)
    };
    limit.validate(policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    const POLICY: Policy = Policy {
        identity: BuiltinCatalogIdentity {
            name: "particleswarm",
        },
        error: &runmat_builtins::PARTICLESWARM_ERROR_INVALID_ARGUMENT,
    };
    #[test]
    fn unit_preserving_limits_and_fractions() {
        let huge = TimeLimit::Days(f64::MAX).validate(POLICY).unwrap();
        assert!(matches!(huge, TimeLimit::Days(v) if v == f64::MAX));
        assert!(!huge.reached(Duration::MAX));
        assert!(huge.fraction(Duration::MAX).unwrap() > 0.0);
        assert_eq!(
            TimeLimit::Days(2.0).fraction(Duration::from_secs(86400)),
            Some(0.5)
        );
        assert_eq!(
            TimeLimit::Seconds(2.0).fraction(Duration::from_secs(1)),
            Some(0.5)
        );
        for limit in [TimeLimit::Days(-0.0), TimeLimit::Seconds(0.0)] {
            assert!(limit.reached(Duration::ZERO));
            assert_eq!(limit.fraction(Duration::ZERO), Some(1.0));
        }
        assert!(matches!(
            TimeLimit::Seconds(f64::INFINITY).validate(POLICY).unwrap(),
            TimeLimit::Unbounded
        ));
        assert_eq!(TimeLimit::Unbounded.fraction(Duration::MAX), None);
        assert!(TimeLimit::Seconds(f64::NAN).validate(POLICY).is_err());
        assert!(TimeLimit::Days(-1.0).validate(POLICY).is_err());
        assert!(parse(POLICY, &Value::Num(3.0))
            .unwrap()
            .reached(Duration::from_secs(3)));
    }
}
