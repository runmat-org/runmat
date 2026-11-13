//! Retain each actual solver's owned outcome, never disguise one domain as another.
use super::*;
use crate::builtins::math::optim::common::solver_timing::SolverTiming;
use crate::builtins::math::optim::{
    intlinprog::compiled as mip, linprog::compiled as lp, lsqlin::compiled as lsq,
    quadprog::compiled as qp,
};
use runmat_builtins::OptimizationExitflag as Flag;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim::optimmodel) enum Products {
    Primal,
    Information,
    WithMultipliers,
}
pub(super) enum Outcome {
    Integer(mip::Outcome),
    Quadratic {
        outcome: qp::Outcome,
        timing: Option<SolverTiming>,
    },
    Linear(lp::Outcome),
    LeastSquares {
        outcome: lsq::Outcome,
        timing: Option<SolverTiming>,
    },
}
pub(super) struct Multipliers {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub eqlin: Vec<f64>,
    pub ineqlin: Vec<f64>,
}
impl Outcome {
    pub fn point(&mut self) -> Option<Vec<f64>> {
        match self {
            Self::Integer(outcome) => {
                outcome.fval = None;
                outcome.x.take()
            }
            Self::Quadratic { outcome, .. } => {
                outcome.fval = None;
                outcome.x.take()
            }
            Self::Linear(out) => {
                out.fval = None;
                out.x.take()
            }
            Self::LeastSquares { outcome, .. } => {
                outcome.residual = None;
                outcome.resnorm = None;
                outcome.x.take()
            }
        }
    }
    pub fn flag(&self, policy: Policy) -> BuiltinResult<Flag> {
        if let Self::Quadratic { outcome, .. } = self {
            if let qp::Method::ActiveSet(reason) = outcome.method {
                return match reason {
                    qp::ActiveTermination::Optimal => Ok(Flag::OptimalSolution),
                    qp::ActiveTermination::IterationLimit | qp::ActiveTermination::TimeLimit => {
                        Ok(Flag::SolverLimitExceeded)
                    }
                    qp::ActiveTermination::ObjectiveLimit => Ok(Flag::ObjectiveLimitReached),
                    qp::ActiveTermination::StepStagnation => Ok(Flag::IllConditioned),
                    qp::ActiveTermination::Numerical => Ok(Flag::NoDescentDirectionFound),
                    qp::ActiveTermination::Infeasible => Ok(Flag::NoFeasiblePointFound),
                    qp::ActiveTermination::Unbounded => Ok(Flag::Unbounded),
                    qp::ActiveTermination::Nonconvex => Ok(Flag::Nonconvex),
                    qp::ActiveTermination::Unverified => {
                        Err(run::unverified(policy, outcome.message))
                    }
                };
            }
        }
        if let Self::LeastSquares { outcome, .. } = self {
            if let lsq::Method::ActiveSet { termination } = outcome.method {
                return match termination {
                    lsq::ActiveTermination::Optimal => Ok(Flag::OptimalSolution),
                    lsq::ActiveTermination::IterationLimit => Ok(Flag::SolverLimitExceeded),
                    lsq::ActiveTermination::ObjectiveLimit => Ok(Flag::ObjectiveLimitReached),
                    lsq::ActiveTermination::IllConditioned => Ok(Flag::IllConditioned),
                    lsq::ActiveTermination::NoDirection => Ok(Flag::NoDescentDirectionFound),
                    lsq::ActiveTermination::Infeasible => Ok(Flag::NoFeasiblePointFound),
                    lsq::ActiveTermination::Unverified => {
                        Err(run::unverified(policy, outcome.message))
                    }
                };
            }
        }
        let (flag, message) = match self {
            Self::Integer(outcome) => {
                return Ok(match outcome.flag {
                    1 => Flag::OptimalSolution,
                    2 => Flag::IntegerFeasible,
                    0 => Flag::SolverLimitExceeded,
                    -1 => Flag::OutputFcnStop,
                    -2 => Flag::NoFeasiblePointFound,
                    -3 => Flag::Unbounded,
                    -9 => Flag::FeasibilityLost,
                    _ => {
                        return Err(
                            policy.invalid("unsupported termination returned by integer solver")
                        )
                    }
                })
            }
            Self::Quadratic { outcome, .. } => (
                if outcome.flag == -8 { -4 } else { outcome.flag },
                outcome.message,
            ),
            Self::Linear(out) => (out.exitflag, out.message.as_str()),
            Self::LeastSquares { outcome, .. } => (
                if outcome.flag == -8 { -4 } else { outcome.flag },
                outcome.message,
            ),
        };
        Ok(match flag {
            1 => Flag::OptimalSolution,
            3 if matches!(self,Self::LeastSquares{outcome,..} if matches!(&outcome.method,lsq::Method::Reflective{..})) => {
                Flag::FunctionChangeBelowTolerance
            }
            0 => Flag::SolverLimitExceeded,
            -2 => Flag::NoFeasiblePointFound,
            -3 => Flag::Unbounded,
            -6 => Flag::Nonconvex,
            -4 => return Err(run::unverified(policy, message)),
            _ => return Err(policy.invalid("unsupported termination returned by selected solver")),
        })
    }
    pub fn multipliers(&mut self) -> Option<Multipliers> {
        match self {
            Self::Integer(_) => None,
            Self::Quadratic { outcome, .. } => outcome.lambda.take().map(|lambda| Multipliers {
                lower: lambda.lower,
                upper: lambda.upper,
                eqlin: lambda.eqlin,
                ineqlin: lambda.ineqlin,
            }),
            Self::Linear(out) => out.lambda.take().map(|lambda| Multipliers {
                lower: lambda.lower,
                upper: lambda.upper,
                eqlin: lambda.eqlin,
                ineqlin: lambda.ineqlin,
            }),
            Self::LeastSquares { outcome, .. } => outcome.lambda.take().map(|lambda| Multipliers {
                lower: lambda.lower,
                upper: lambda.upper,
                eqlin: lambda.eqlin,
                ineqlin: lambda.ineqlin,
            }),
        }
    }
}
