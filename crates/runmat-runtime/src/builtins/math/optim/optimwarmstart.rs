//! Immutable initial/solution warm data; every invocation uses current owners.
use super::common::{model_policy::Policy, native_class_registration, ScalarCooperation};
use crate::{context::RuntimeContext, execution::RootedValue, BuiltinResult};
use runmat_builtins::{
    OptimWarmStartProperty as Property, LSQLIN_WARM_START_CLASS, OPTIMWARMSTART_CATALOG_ENTRY,
    OPTIMWARMSTART_ERROR_INVALID_INPUT, QUADPROG_WARM_START_CLASS,
};
use runmat_value::{
    warm_start::{ActiveSetWarmStartData, LinearWorkingSetData, WarmStartMemoryLimits},
    ObjectInstance, Tensor, Value,
};
pub(crate) mod bindings;
mod constructor;
mod properties;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum Kind {
    LeastSquares,
    Quadratic,
}
impl Kind {
    fn class(self) -> runmat_types::StaticClassIdentity {
        match self {
            Self::LeastSquares => LSQLIN_WARM_START_CLASS,
            Self::Quadratic => QUADPROG_WARM_START_CLASS,
        }
    }
}
fn policy() -> Policy {
    Policy {
        identity: OPTIMWARMSTART_CATALOG_ENTRY.identity,
        invalid: &OPTIMWARMSTART_ERROR_INVALID_INPUT,
    }
}
pub(in crate::builtins::math::optim) fn checked<'a>(
    object: &'a ObjectInstance,
    kind: Kind,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<&'a ActiveSetWarmStartData> {
    if !object.class_name().is(kind.class()) || object.selected_class().is_some() {
        return Err(
            policy().invalid("expected an original native warm start of the selected solver")
        );
    }
    let data = object
        .active_set_warm_start_payload()
        .ok_or_else(|| policy().invalid("warm start has no native solver data"))?;
    native_class_registration::ensure(kind.class(), &[], policy(), runtime)?;
    Ok(data)
}
pub(in crate::builtins::math::optim) fn updated(
    kind: Kind,
    x: Tensor,
    previous: &ActiveSetWarmStartData,
    working: Option<LinearWorkingSetData>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    let Value::Struct(options) = previous.options() else {
        return Err(policy().invalid("warm options lost their structure"));
    };
    publish_value(
        kind,
        x,
        options.clone(),
        previous.limits(),
        working,
        runtime,
    )
}
fn publish_value(
    kind: Kind,
    x: Tensor,
    options: runmat_value::StructValue,
    limits: WarmStartMemoryLimits,
    working: Option<LinearWorkingSetData>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    native_class_registration::ensure(kind.class(), &[], policy(), runtime)?;
    let data = ActiveSetWarmStartData::new(x, options, limits, working)
        .map_err(|e| policy().invalid(e))?;
    Ok(Value::Object(
        ObjectInstance::new(kind.class().owned()).with_active_set_warm_start_payload(data),
    ))
}
