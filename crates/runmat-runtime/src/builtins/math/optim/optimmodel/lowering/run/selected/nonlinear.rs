//! A nonlinear stopped candidate has no affine feasibility/certificate cache.
use super::super::*;
pub(super) async fn run<'a>(
    prepared: NonlinearPrepared,
    invocation: Invocation<'a>,
) -> BuiltinResult<Completed<'a>> {
    let NonlinearPrepared {
        model,
        formulation,
        solver,
    } = prepared;
    let Invocation {
        options,
        starting,
        products,
        policy,
        cooperation,
    } = invocation;
    let solved = nonlinear::solve(
        model,
        &formulation,
        solver,
        starting,
        options,
        products,
        policy,
        cooperation,
    )
    .await?;
    Ok(Completed {
        family: Either::Right(Either::Left(NonlinearSolved {
            solved,
            formulation,
        })),
        policy,
        cooperation,
    })
}
