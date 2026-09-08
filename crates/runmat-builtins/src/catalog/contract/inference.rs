use runmat_types::{CallInference, CallRequest};
use serde::Serialize;

use super::{
    AccelerationInferenceRule, AggregateInferenceRule, ArrayInferenceRule,
    IntrospectionInferenceRule, IoInferenceRule, LogicalInferenceRule, MathInferenceRule,
    ParallelInferenceRule, StatsInferenceRule,
};
use crate::BuiltinCatalogEntry;

pub(crate) type BuiltinInferenceHandler = fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference;

/// Connects one catalog entry directly to inference owned by that identity.
///
/// The executable pointer is omitted from serialized catalog data because a
/// process address is not a stable contract. The surrounding entry supplies
/// the stable builtin identity, while generated products fingerprint source.
#[derive(Clone, Copy, Serialize)]
pub struct IdentityInferenceRule {
    #[serde(skip)]
    handler: BuiltinInferenceHandler,
}

impl IdentityInferenceRule {
    pub(crate) const fn new(handler: BuiltinInferenceHandler) -> Self {
        Self { handler }
    }

    pub(crate) fn infer(self, request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
        (self.handler)(request, entry)
    }
}

impl std::fmt::Debug for IdentityInferenceRule {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("IdentityInferenceRule")
    }
}

impl PartialEq for IdentityInferenceRule {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.handler, other.handler)
    }
}

impl Eq for IdentityInferenceRule {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinInferenceRule {
    Identity(IdentityInferenceRule),
    Acceleration(AccelerationInferenceRule),
    Aggregate(AggregateInferenceRule),
    Array(ArrayInferenceRule),
    Introspection(IntrospectionInferenceRule),
    Io(IoInferenceRule),
    Logical(LogicalInferenceRule),
    Math(MathInferenceRule),
    Parallel(ParallelInferenceRule),
    Stats(StatsInferenceRule),
}
