//! Construct only the selected typed numerical child.
use super::*;
mod integer;
mod linear;
mod nonlinear;
mod quadratic;
mod residual;
pub(super) fn run<'a>(
    prepared: Prepared<'a>,
) -> impl std::future::Future<Output = BuiltinResult<Completed<'a>>> + 'a {
    let Prepared { family, invocation } = prepared;
    match family {
        Either::Left(prepared) => Either::Left(algebraic(prepared, invocation)),
        Either::Right(prepared) => Either::Right(nonlinear::run(prepared, invocation)),
    }
}
fn algebraic<'a>(
    prepared: Algebraic,
    invocation: Invocation<'a>,
) -> impl std::future::Future<Output = BuiltinResult<Completed<'a>>> + 'a {
    if prepared.solver == OptimSolver::Intlinprog {
        Either::Left(integer::run(prepared, invocation))
    } else {
        Either::Right(match prepared.solver {
            OptimSolver::Linprog => Either::Left(linear::run(prepared, invocation)),
            OptimSolver::Lsqlin => Either::Right(Either::Left(residual::run(prepared, invocation))),
            OptimSolver::Quadprog => {
                Either::Right(Either::Right(quadratic::run(prepared, invocation)))
            }
            _ => unreachable!("classified algebraic model solver"),
        })
    }
}
