//! Public typed affine, residual and quadratic model solver selection.
use super::*;
pub(in crate::builtins::math::optim::optimmodel) async fn execute(
    arguments: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    if arguments.is_empty() {
        return Err(policy.invalid("solve expects an optimization problem"));
    }
    let _inputs = policy.roots(arguments.iter(), cooperation.runtime())?;
    let model = data(&arguments[0], policy)?;
    let super::super::solve_arguments::Parsed {
        starting,
        options,
        selected,
    } = super::super::solve_arguments::parse(
        arguments,
        super::super::solve_arguments::Family::Optimization,
        policy,
        cooperation,
    )
    .await?;
    let count = crate::output_count::current_output_count_in(cooperation.runtime());
    let products = if count.is_some_and(|n| n >= 5) {
        super::super::lowering::Products::WithMultipliers
    } else if count.is_some_and(|n| n >= 4) {
        super::super::lowering::Products::Information
    } else {
        super::super::lowering::Products::Primal
    };
    let solved = super::super::lowering::solve(
        model,
        options,
        products,
        selected,
        starting,
        policy,
        cooperation,
    )?
    .await?;
    super::super::lowering::publish(solved, count, policy, cooperation).await
}
