//! Equation solve uses the original native generation and public solve grammar.
use super::*;
pub(in crate::builtins::math::optim::optimmodel) async fn execute(
    arguments: crate::call::FunctionArgumentView<'_>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let _inputs = policy.roots(arguments.iter(), c.runtime())?;
    let model = data(
        arguments
            .get(0)
            .ok_or_else(|| policy.invalid("solve expects an equation problem"))?,
        policy,
    )?;
    let count = crate::output_count::current_output_count_in(c.runtime());
    if count.is_some_and(|count| count > 4) {
        return Err(policy.invalid(
            "equation solve returns at most four outputs; equation problems do not return lambda",
        ));
    }
    let super::super::solve_arguments::Parsed {
        starting,
        options,
        selected,
    } = super::super::solve_arguments::parse(
        arguments,
        super::super::solve_arguments::Family::Equation,
        policy,
        c,
    )
    .await?;
    let solved = super::super::lowering::equation::solve(
        model,
        selected,
        starting,
        options,
        count.is_some_and(|count| count > 3),
        policy,
        c,
    )?
    .await?;
    super::super::lowering::equation::publish(solved, count, policy, c).await
}
