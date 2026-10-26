//! Problem-based termination values use the existing enumeration object owner,
//! class issuance and rooted object publication, not a new Value payload.
use super::*;
use runmat_builtins::OptimizationExitflag as Flag;
pub(super) fn register(policy: Policy, runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    registration::ensure(standard::OPTIMIZATION_EXITFLAG, policy, runtime)?;
    crate::context::with_original_scope(runtime, || {
        crate::class_registry::register_class_enumerations(
            &standard::OPTIMIZATION_EXITFLAG.owned(),
            Flag::ALL.into_iter().map(|flag| flag.name().to_owned()),
        );
    });
    Ok(())
}
pub(super) fn publish(
    flag: Flag,
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValue> {
    register(policy, runtime)?;
    crate::object::resolve::enumeration_value_in(
        runtime,
        &standard::OPTIMIZATION_EXITFLAG.owned(),
        flag.name(),
    )
}
pub(super) fn read(
    value: &Value,
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Flag> {
    let Value::Object(object) = value else {
        return Err(policy.invalid("expected an OptimizationExitflag scalar"));
    };
    if !object.class_name().is(standard::OPTIMIZATION_EXITFLAG) || object.selected_class().is_some()
    {
        return Err(policy.invalid("expected an original OptimizationExitflag value"));
    }
    registration::ensure(standard::OPTIMIZATION_EXITFLAG, policy, runtime)?;
    let Some(name) = crate::object::resolve::enumeration_member_name(object) else {
        return Err(policy.invalid("termination value has no enumeration member"));
    };
    Flag::parse_member(name).ok_or_else(|| policy.invalid("unknown OptimizationExitflag member"))
}
