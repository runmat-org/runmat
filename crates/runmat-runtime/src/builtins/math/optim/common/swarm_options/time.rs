//! Swarm diagnostics delegate to the shared unit-preserving time-limit owner.
pub(crate) use super::super::global_time::TimeLimit;
use super::Owner;
use crate::BuiltinResult;
use runmat_value::Value;
pub(crate) fn time_limit(owner: Owner, value: &Value) -> BuiltinResult<TimeLimit> {
    super::super::global_time::parse(owner.time_policy(), value)
}
