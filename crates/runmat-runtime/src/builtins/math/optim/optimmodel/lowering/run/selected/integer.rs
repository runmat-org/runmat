//! Same original source/layout and the existing integer branch-and-bound port.
use super::super::*;
use crate::builtins::math::optim::intlinprog::compiled as mip;
pub(super) async fn run<'a>(
    prepared: Algebraic,
    invocation: Invocation<'a>,
) -> BuiltinResult<Completed<'a>> {
    let Algebraic {
        mut program,
        formulation,
        rows,
        assembled,
        ..
    } = prepared;
    let Invocation {
        options,
        starting,
        products,
        policy,
        cooperation,
    } = invocation;
    let assembly::Cost::Affine(f) = assembled.cost else {
        return Err(policy.invalid("integer affine solver received a nonlinear objective"));
    };
    // All numerical rows/cost now have their one assembled owner. Original
    // proof and publication use the retained source/facts/layout, not proposal
    // coefficient tables. Do not keep a second numerical matrix mirror.
    program.arrays.clear();
    let prepared = mip::prepare(
        mip::Problem {
            f,
            a_ineq: assembled.a_ineq,
            b_ineq: assembled.b_ineq,
            a_eq: assembled.a_eq,
            b_eq: assembled.b_eq,
            lower: assembled.lb,
            upper: assembled.ub,
            integers: assembled.integers,
            semis: assembled.semis,
        },
        options,
        cooperation,
    )
    .await?;
    // The selected original clock/options precede actual named x0 conversion,
    // just as raw intlinprog prepares controls before its numeric x0 conversion.
    let initial = match starting {
        Some(value) => Some(
            layout::initial_point(&program.source, &program.layout, value, policy, cooperation)
                .await?,
        ),
        None => None,
    };
    let mut validation =
        validation::Validator::new(&program, &formulation, &rows, policy, cooperation).await?;
    let outcome = mip::solve(
        prepared,
        initial,
        if products == Products::Primal {
            mip::Products::Primal
        } else {
            mip::Products::Information
        },
        cooperation,
        &mut validation,
    )
    .await?;
    let retained = validation.into_point();
    Ok(Completed {
        family: Either::Left(AlgebraicCompleted {
            program,
            formulation,
            rows,
            retained,
            outcome: report::Outcome::Integer(outcome),
        }),
        policy,
        cooperation,
    })
}
