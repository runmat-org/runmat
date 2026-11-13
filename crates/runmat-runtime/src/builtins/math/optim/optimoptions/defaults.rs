//! Concrete public defaults; variable-dependent limits resolve in the solver.
use super::*;
pub(super) fn default_options(
    runtime: Option<&crate::context::RuntimeContext>,
    solver: Solver,
) -> BuiltinResult<OptionState> {
    let mut out = OptionState::new();
    if solver != Solver::Generic {
        out.set_solver(state::text(runtime, solver.name())?);
    }
    match solver {
        Solver::Surrogateopt => {
            for field in OptimOptionField::ALL {
                if solver.accepts(field) {
                    if let Some(value) =
                        crate::builtins::math::optim::common::surrogate_options::default_value(
                            field, runtime,
                        )?
                    {
                        out.set(field, value);
                    }
                }
            }
        }
        Solver::Ga => {
            for (field, value) in [
                (OptimOptionField::FunctionTolerance, 1e-6),
                (OptimOptionField::TolCon, 1e-3),
                (OptimOptionField::CrossoverFraction, 0.8),
                (OptimOptionField::MaxStallIterations, 50.),
                (OptimOptionField::MaxTime, f64::INFINITY),
                (OptimOptionField::MaxStallTime, f64::INFINITY),
                (OptimOptionField::ObjectiveLimit, f64::NEG_INFINITY),
                (OptimOptionField::MigrationFraction, 0.2),
                (OptimOptionField::MigrationInterval, 20.),
                (OptimOptionField::InitialPenalty, 10.),
                (OptimOptionField::PenaltyFactor, 100.),
                (OptimOptionField::DisplayInterval, 1.),
                (OptimOptionField::PlotInterval, 1.),
            ] {
                out.set(field, Value::Num(value));
            }
            for (field, value) in [
                (OptimOptionField::PopulationType, "doubleVector"),
                (OptimOptionField::StallTest, "averageChange"),
                (OptimOptionField::FitnessScalingFcn, "fitscalingrank"),
                (OptimOptionField::MigrationDirection, "forward"),
                (OptimOptionField::FunValCheck, "off"),
                (OptimOptionField::UseParallel, "off"),
                (OptimOptionField::Display, "final"),
            ] {
                out.set(field, state::text(runtime, value)?);
            }
            out.set(OptimOptionField::UseVectorized, Value::Bool(false));
            for field in [
                OptimOptionField::MaxIter,
                OptimOptionField::PopulationSize,
                OptimOptionField::EliteCount,
                OptimOptionField::InitialPopulationMatrix,
                OptimOptionField::InitialScoresMatrix,
                OptimOptionField::InitialPopulationRange,
                OptimOptionField::CreationFcn,
                OptimOptionField::CrossoverFcn,
                OptimOptionField::MutationFcn,
                OptimOptionField::SelectionFcn,
                OptimOptionField::NonlinearConstraintAlgorithm,
                OptimOptionField::HybridFcn,
                OptimOptionField::OutputFcn,
                OptimOptionField::PlotFcn,
            ] {
                out.set(
                    field,
                    crate::builtins::math::optim::common::output_callback::empty(runtime)?,
                );
            }
        }
        Solver::Patternsearch => pattern::defaults(runtime, &mut out)?,
        Solver::Simulannealbnd => {
            for field in OptimOptionField::ALL {
                if let Some(value) =
                    crate::builtins::math::optim::common::anneal_options::default_value(
                        field, runtime,
                    )?
                {
                    out.set(field, value);
                }
            }
        }
        Solver::Particleswarm => {
            for (field, value) in [
                (OptimOptionField::FunctionTolerance, 1e-6),
                (OptimOptionField::DisplayInterval, 1.0),
                (OptimOptionField::MaxStallIterations, 20.0),
                (OptimOptionField::MaxStallTime, f64::INFINITY),
                (OptimOptionField::MaxTime, f64::INFINITY),
                (OptimOptionField::ObjectiveLimit, f64::NEG_INFINITY),
                (OptimOptionField::MinNeighborsFraction, 0.25),
                (OptimOptionField::SelfAdjustmentWeight, 1.49),
                (OptimOptionField::SocialAdjustmentWeight, 1.49),
                (OptimOptionField::InitialSwarmSpan, 2000.0),
            ] {
                out.set(field, Value::Num(value));
            }
            for field in [
                OptimOptionField::MaxIter,
                OptimOptionField::SwarmSize,
                OptimOptionField::InitialPoints,
                OptimOptionField::HybridFcn,
                OptimOptionField::OutputFcn,
                OptimOptionField::PlotFcn,
            ] {
                out.set(
                    field,
                    crate::builtins::math::optim::common::output_callback::empty(runtime)?,
                );
            }
            out.set(
                OptimOptionField::CreationFcn,
                state::text(runtime, "pswcreationuniform")?,
            );
            out.set(OptimOptionField::FunValCheck, state::text(runtime, "off")?);
            out.set(OptimOptionField::UseParallel, state::text(runtime, "off")?);
            out.set(OptimOptionField::UseVectorized, Value::Bool(false));
            out.set(OptimOptionField::Display, state::text(runtime, "final")?);
            out.set(
                OptimOptionField::InertiaRange,
                Value::Tensor(
                    runmat_value::Tensor::new(vec![0.1, 1.1], vec![1, 2]).map_err(|source| {
                        optimoptions_error_with(
                            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                            source.to_string(),
                        )
                    })?,
                ),
            );
        }
        Solver::Fseminf => {
            for (field, value) in [
                (OptimOptionField::TolX, 1e-4),
                (OptimOptionField::FunctionTolerance, 1e-4),
                (OptimOptionField::OptimalityTolerance, 1e-6),
                (OptimOptionField::TolCon, 1e-6),
                (OptimOptionField::TolConSQP, 1e-6),
                (OptimOptionField::MaxIter, 400.0),
                (OptimOptionField::DiffMinChange, 0.0),
                (OptimOptionField::DiffMaxChange, f64::INFINITY),
                (OptimOptionField::RelLineSrchBndDuration, 1.0),
            ] {
                out.set(field, Value::Num(value));
            }
            for (field, value) in [
                (OptimOptionField::Display, "final"),
                (OptimOptionField::FiniteDifferenceType, "forward"),
                (OptimOptionField::Diagnostics, "off"),
                (OptimOptionField::FunValCheck, "off"),
            ] {
                out.set(field, state::text(runtime, value)?);
            }
            out.set(
                OptimOptionField::SpecifyObjectiveGradient,
                Value::Bool(false),
            );
            for field in [
                OptimOptionField::MaxFunEvals,
                OptimOptionField::MaxSQPIter,
                OptimOptionField::RelLineSrchBnd,
                OptimOptionField::OutputFcn,
                OptimOptionField::PlotFcn,
            ] {
                out.set(
                    field,
                    crate::builtins::math::optim::common::output_callback::empty(runtime)?,
                );
            }
        }
        Solver::Fminimax | Solver::Fgoalattain => {
            for field in [
                OptimOptionField::TolX,
                OptimOptionField::OptimalityTolerance,
                OptimOptionField::FunctionTolerance,
                OptimOptionField::TolCon,
            ] {
                out.set(field, Value::Num(1e-6));
            }
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(
                OptimOptionField::MaxFunEvals,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
            out.set(OptimOptionField::Display, state::text(runtime, "final")?);
            out.set(
                OptimOptionField::FiniteDifferenceType,
                state::text(runtime, "forward")?,
            );
            out.set(
                OptimOptionField::MeritFunction,
                state::text(runtime, "multiobj")?,
            );
            out.set(
                if solver == Solver::Fminimax {
                    OptimOptionField::AbsoluteMaxObjectiveCount
                } else {
                    OptimOptionField::EqualityGoalCount
                },
                Value::Num(0.0),
            );
            for field in [
                OptimOptionField::SpecifyObjectiveGradient,
                OptimOptionField::SpecifyConstraintGradient,
            ] {
                out.set(field, Value::Bool(false));
            }
            out.set(
                OptimOptionField::OutputFcn,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
        }
        Solver::Fmincon => {
            out.set(
                OptimOptionField::FiniteDifferenceType,
                state::text(runtime, "forward")?,
            );
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "interior-point")?,
            );
            for (field, value) in [
                (OptimOptionField::TolX, 1e-10),
                (OptimOptionField::OptimalityTolerance, 1e-6),
                (OptimOptionField::TolCon, 1e-6),
                (OptimOptionField::MaxIter, 1000.0),
            ] {
                out.set(field, Value::Num(value));
            }
            // The public default remains interior-point. Selecting actual SQP
            // changes only still-default controls in final normalization.
            out.set(OptimOptionField::MaxFunEvals, Value::Num(3000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
            out.set(
                OptimOptionField::OutputFcn,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
            for field in [
                OptimOptionField::SpecifyObjectiveGradient,
                OptimOptionField::SpecifyConstraintGradient,
            ] {
                out.set(field, Value::Bool(false));
            }
        }
        Solver::Intlinprog => {
            for (field, value) in [
                (OptimOptionField::AbsoluteGapTolerance, 1e-6),
                (OptimOptionField::RelativeGapTolerance, 1e-4),
                (OptimOptionField::TolCon, 1e-6),
                (OptimOptionField::MaxFeasiblePoints, f64::INFINITY),
                (OptimOptionField::MaxNodes, 10_000_000.0),
                (OptimOptionField::MaxTime, 7200.0),
                (OptimOptionField::ObjectiveCutOff, f64::INFINITY),
            ] {
                out.set(field, Value::Num(value));
            }
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
            for field in [OptimOptionField::OutputFcn, OptimOptionField::PlotFcn] {
                out.set(
                    field,
                    crate::builtins::math::optim::common::output_callback::empty(runtime)?,
                );
            }
        }
        Solver::Quadprog | Solver::Lsqlin => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(
                    runtime,
                    if solver == Solver::Lsqlin {
                        "interior-point"
                    } else {
                        "interior-point-convex"
                    },
                )?,
            );
            out.set(OptimOptionField::TolFun, Value::Num(1e-8));
            out.set(OptimOptionField::TolCon, Value::Num(1e-8));
            out.set(OptimOptionField::MaxIter, Value::Num(200.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Coneprog => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "interior-point")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-7));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-7));
            out.set(OptimOptionField::MaxIter, Value::Num(200.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Linprog => {
            out.set(OptimOptionField::MaxIter, Value::Num(100000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Fminbnd => {
            out.set(OptimOptionField::TolX, Value::Num(1.0e-4));
            out.set(OptimOptionField::MaxIter, Value::Num(500.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(500.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Fminunc => {
            out.set(
                OptimOptionField::OutputFcn,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "quasi-newton")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::OptimalityTolerance, Value::Num(1.0e-6));
            out.set(
                OptimOptionField::HessianApproximation,
                state::text(runtime, "bfgs")?,
            );
            out.set(OptimOptionField::ObjectiveLimit, Value::Num(-1e20));
            out.set(OptimOptionField::UseParallel, Value::Bool(false));
            for field in OptimOptionField::ALL
                .into_iter()
                .filter(|field| field.trust_region_only() && solver.accepts(*field))
            {
                let value = match field {
                    OptimOptionField::FunctionTolerance => Value::Num(1e-6),
                    OptimOptionField::TolPCG => Value::Num(0.1),
                    OptimOptionField::PrecondBandWidth => Value::Num(0.0),
                    OptimOptionField::SubproblemAlgorithm => state::text(runtime, "cg")?,
                    _ => crate::builtins::math::optim::common::output_callback::empty(runtime)?,
                };
                out.set(field, value);
            }
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
            out.set(
                OptimOptionField::SpecifyObjectiveGradient,
                Value::Bool(false),
            );
        }
        Solver::Fzero => {
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(500.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Fsolve => {
            out.set(
                OptimOptionField::SpecifyObjectiveGradient,
                Value::Bool(false),
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Lsqcurvefit => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "levenberg-marquardt")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::TolFun, Value::Num(1.0e-6));
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(OptimOptionField::MaxFunEvals, Value::Num(40000.0));
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Lsqnonlin => {
            out.set(
                OptimOptionField::Algorithm,
                state::text(runtime, "trust-region-reflective")?,
            );
            out.set(OptimOptionField::TolX, Value::Num(1.0e-6));
            out.set(OptimOptionField::FunctionTolerance, Value::Num(1.0e-6));
            out.set(OptimOptionField::OptimalityTolerance, Value::Num(1.0e-6));
            out.set(
                OptimOptionField::SpecifyObjectiveGradient,
                Value::Bool(false),
            );
            out.set(
                OptimOptionField::SubproblemAlgorithm,
                state::text(runtime, "factorization")?,
            );
            out.set(
                OptimOptionField::PrecondBandWidth,
                Value::Num(f64::INFINITY),
            );
            out.set(OptimOptionField::TolPCG, Value::Num(0.1));
            out.set(
                OptimOptionField::JacobianMultiplyFcn,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
            out.set(
                OptimOptionField::MaxPCGIter,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
            out.set(OptimOptionField::MaxIter, Value::Num(400.0));
            out.set(
                OptimOptionField::MaxFunEvals,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
            out.set(
                OptimOptionField::Display,
                state::text(runtime, solver.default_display())?,
            );
        }
        Solver::Generic => {}
    }
    for field in [
        OptimOptionField::FiniteDifferenceStepSize,
        OptimOptionField::TypicalX,
    ] {
        if solver != Solver::Generic && solver.accepts(field) {
            // Dimension/mode-dependent defaults are resolved by the numerical owner.
            out.set(
                field,
                crate::builtins::math::optim::common::output_callback::empty(runtime)?,
            );
        }
    }
    Ok(out)
}
