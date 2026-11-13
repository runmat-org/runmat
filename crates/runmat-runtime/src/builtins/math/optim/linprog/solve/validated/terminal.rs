//! Only the actual terminal status constructs a numerical/validation child.
use super::*;
use futures::future::Either;
mod solved;
mod infeasible;
mod descent;
mod incumbent;
pub(super) fn finish<'a, V: super::compiled::Validation + 'a>(
    prepared: Prepared<'a, V>,
) -> impl std::future::Future<Output = BuiltinResult<LinprogOutcome>> + 'a {
    match prepared.point.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => Either::Left(solved::run(prepared)),
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
            Either::Right(Either::Left(infeasible::run(prepared)))
        }
        SolverStatus::DualInfeasible | SolverStatus::AlmostDualInfeasible => {
            Either::Right(Either::Right(Either::Left(descent::run(prepared))))
        }
        SolverStatus::MaxIterations | SolverStatus::MaxTime | SolverStatus::CallbackTerminated => {
            Either::Right(Either::Right(Either::Right(incumbent::run(
                prepared,
                0,
                "Stopped before an optimality or infeasibility certificate was established.",
            ))))
        }
        SolverStatus::Unsolved
        | SolverStatus::NumericalError
        | SolverStatus::InsufficientProgress => {
            Either::Right(Either::Right(Either::Right(incumbent::run(
                prepared,
                -4,
                "Numerical progress was insufficient for a verified certificate.",
            ))))
        }
    }
}
