//! One selected original numerical body; other solver futures do not exist here.
use super::super::*;
pub(super) async fn run<'a>(
    prepared: Algebraic,
    invocation: Invocation<'a>,
) -> BuiltinResult<Completed<'a>> {
    let Algebraic {
        program,
        formulation,
        rows,
        assembled,
        ..
    } = prepared;
    let Invocation {
        options,
        products,
        policy,
        cooperation,
        ..
    } = invocation;
    let mut validation =
        validation::Validator::new(&program, &formulation, &rows, policy, cooperation).await?;
    let outcome = {
        let assembly::Cost::Affine(f) = assembled.cost else {
            return Err(policy.invalid("linear objective assembly is unavailable"));
        };
        let problem = lp::Problem {
            f,
            a_ineq: assembled.a_ineq,
            b_ineq: assembled.b_ineq,
            a_eq: assembled.a_eq,
            b_eq: assembled.b_eq,
            lb: assembled.lb,
            ub: assembled.ub,
        };
        report::Outcome::Linear(
            lp::solve(
                problem,
                options,
                if products == Products::WithMultipliers {
                    lp::Products::WithMultipliers
                } else {
                    lp::Products::Primal
                },
                cooperation,
                &mut validation,
            )
            .await?,
        )
    };
    let retained = validation.into_point();
    Ok(Completed {
        family: Either::Left(AlgebraicCompleted {
            program,
            formulation,
            rows,
            retained,
            outcome,
        }),
        policy,
        cooperation,
    })
}
