//! Model targets for the canonical class-creation argument ABI.
use super::*;
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.zerosLike",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::creation"
)]
async fn zeros_like(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ZEROS_LIKE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let Some(prototype) = args.get(0) else {
        return Err(policy.invalid("zerosLike requires an original model prototype"));
    };
    match prototype {
        Value::Object(object) if is_expression(object) => {
            let _node = owner::OwnedNode::read(owner::reference(prototype, policy)?, policy)?;
        }
        Value::Object(object) if super::super::super::optimvar::is_reference(object) => {
            let _view = super::super::super::optimvar::View::new(object, &mut cooperation).await?;
        }
        _ => {
            return Err(
                policy.invalid("zerosLike requires an original expression or variable prototype")
            )
        }
    }
    super::super::construct::zeros(
        args.view().range(1..).expect("prototype was checked"),
        policy,
        &mut cooperation,
    )
    .await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.zeros",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::creation"
)]
async fn zeros(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_ZEROS_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    super::super::construct::zeros(args.view(), policy, &mut cooperation).await
}
