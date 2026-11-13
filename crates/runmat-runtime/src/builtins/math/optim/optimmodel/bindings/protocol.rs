//! Public standard-substruct and evaluation methods.
use super::*;
use crate::object::indexing::{ObjectIndexSelector, ObjectSubscriptPath};
use runmat_builtins::OptimizationExpressionProperty as Property;

async fn expression_read(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.len() != 2 {
        return Err(
            policy.invalid("subsref expects an expression and standard subscript descriptor")
        );
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    super::super::protocol::read(&args[0], &path, policy, cooperation).await
}
async fn expression_write(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.len() != 3 {
        return Err(
            policy.invalid("subsasgn expects expression, standard subscript descriptor and value")
        );
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    super::super::protocol::assign(&args[0], &path, &args[2], policy, cooperation).await
}
async fn comparison_read(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.len() != 2 {
        return Err(
            policy.invalid("subsref expects a constraint and standard subscript descriptor")
        );
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    let (step, remaining) = path
        .steps()
        .split_first()
        .ok_or_else(|| policy.invalid("constraint path is empty"))?;
    let ObjectIndexSelector::Member(name) = step.selector() else {
        return Err(policy.invalid("constraint element indexing is not yet implemented"));
    };
    let property =
        Property::parse(name).ok_or_else(|| policy.invalid("unknown constraint property"))?;
    let root = constraints::read_property(&args[0], property, policy, cooperation).await?;
    if remaining.is_empty() {
        return Ok(root);
    }
    super::super::protocol::read_ordinary(
        root,
        ObjectSubscriptPath::new(remaining.to_vec())?,
        cooperation,
    )
    .await
}
async fn comparison_write(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.len() != 3 {
        return Err(policy.invalid("subsasgn expects constraint, descriptor and value"));
    }
    let _inputs = policy.roots(args.iter(), cooperation.runtime())?;
    let path = crate::object::indexing::parse_standard_substruct(&args[1])?;
    comparison_assign_path(&args[0], &path, &args[2], policy, cooperation).await
}
pub(in crate::builtins::math::optim::optimmodel) async fn comparison_assign_path(
    value: &Value,
    path: &ObjectSubscriptPath,
    rhs: &Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let (step, remaining) = path
        .steps()
        .split_first()
        .ok_or_else(|| policy.invalid("constraint path is empty"))?;
    let ObjectIndexSelector::Member(name) = step.selector() else {
        return Err(policy.invalid("constraint element assignment is not yet implemented"));
    };
    let property =
        Property::parse(name).ok_or_else(|| policy.invalid("unknown constraint property"))?;
    if property == Property::Variables {
        return Err(policy.invalid("Variables is a read-only property"));
    }
    if remaining.is_empty() {
        return constraints::set_names(value, rhs, policy, cooperation).await;
    }
    let names = constraints::read_property(value, property, policy, cooperation).await?;
    let names = super::super::protocol::assign_ordinary(
        names,
        ObjectSubscriptPath::new(remaining.to_vec())?,
        rhs,
        cooperation,
    )
    .await?;
    constraints::set_names(value, names.value(), policy, cooperation).await
}
async fn evaluation(
    args: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    if args.len() != 2 {
        return Err(policy.invalid("evaluate expects an expression/constraint and point structure"));
    }
    super::evaluate_model(&args[0], &args[1], policy, cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::protocol"
)]
async fn optimizationexpression_subsref(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SUBSREF_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    expression_read(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::protocol"
)]
async fn optimizationexpression_subsasgn(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_SUBSASGN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    expression_write(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationExpression.evaluate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::protocol"
)]
async fn optimizationexpression_evaluate(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMEXPR_EVALUATE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    evaluation(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationConstraint.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::protocol"
)]
async fn optimizationconstraint_subsref(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMCONSTRAINT_SUBSREF_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    comparison_read(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationConstraint.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::protocol"
)]
async fn optimizationconstraint_subsasgn(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMCONSTRAINT_SUBSASGN_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    comparison_write(args.view(), policy, &mut cooperation).await
}
#[runmat_macros::runtime_builtin(
    name = "optim.problemdef.OptimizationConstraint.evaluate",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimmodel::bindings::protocol"
)]
async fn optimizationconstraint_evaluate(
    runtime: Option<RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let entry = &runmat_builtins::OPTIMCONSTRAINT_EVALUATE_CATALOG_ENTRY;
    validate(entry, runtime.as_ref())?;
    let policy = policy(entry);
    let mut cooperation = policy.cooperation(runtime.as_ref()).await?;
    evaluation(args.view(), policy, &mut cooperation).await
}
