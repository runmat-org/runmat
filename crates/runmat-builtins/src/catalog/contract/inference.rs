use serde::Serialize;

use super::{
    AccelerationInferenceRule, ArrayInferenceRule, IntrospectionInferenceRule, IoInferenceRule,
    LogicalInferenceRule, MathInferenceRule, ParallelInferenceRule, StatsInferenceRule,
};
#[path = "inference/identity.rs"]
mod identity;

pub use identity::IdentityInferenceRule;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinInferenceRule {
    Identity(IdentityInferenceRule),
    Acceleration(AccelerationInferenceRule),
    Array(ArrayInferenceRule),
    Introspection(IntrospectionInferenceRule),
    Io(IoInferenceRule),
    Logical(LogicalInferenceRule),
    Math(MathInferenceRule),
    Parallel(ParallelInferenceRule),
    Stats(StatsInferenceRule),
}
