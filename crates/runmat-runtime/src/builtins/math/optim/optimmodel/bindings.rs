//! Public dispatch retains the original explicit context and rooted inputs.
use super::*;
pub(crate) mod arithmetic;
pub(crate) mod creation;
pub(crate) mod equation;
pub(crate) mod exitflag;
pub(crate) mod functions;
pub(crate) mod problem;
pub(crate) mod protocol;
fn policy(entry: &runmat_builtins::BuiltinCatalogEntry) -> Policy {
    Policy {
        identity: entry.identity,
        invalid: &runmat_builtins::OPTIMMODEL_ERROR_INVALID_INPUT,
    }
}
fn validate(
    entry: &runmat_builtins::BuiltinCatalogEntry,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    crate::output_count::validate_bounded_outputs_in(
        entry.descriptor,
        &runmat_builtins::OPTIMMODEL_ERROR_INVALID_INPUT,
        entry.identity.name,
        runtime,
    )
}
#[runmat_macros::runtime_builtin(
    name = "optimexpr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings"
)]
async fn constructor(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    construct::empty(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "evaluate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings"
)]
async fn evaluate(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMMODEL_EVALUATE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    if args.len() != 2 {
        return Err(policy
            .invalid("evaluate expects a model expression or comparison and a point structure"));
    }
    evaluate_model(&args[0], &args[1], policy, &mut cooperation).await
}
async fn evaluate_model(
    value: &Value,
    point: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let _inputs = policy.roots([value, point].into_iter(), cooperation.runtime())?;
    if let Value::Object(object) = value {
        if matches!(
            object.optimization_model_payload(),
            Some(Payload::Constraint(_))
        ) {
            return constraints::evaluate(value, point, policy, cooperation).await;
        }
        if is_expression(object) || super::super::optimvar::is_reference(object) {
            let expression = owner::as_expression(value, policy, cooperation).await?;
            return evaluation::evaluate(
                owner::reference(expression.value(), policy)?,
                point,
                policy,
                cooperation,
            )
            .await;
        }
    }
    Err(policy
        .invalid("evaluate requires an original optimization expression, variable or comparison"))
}
