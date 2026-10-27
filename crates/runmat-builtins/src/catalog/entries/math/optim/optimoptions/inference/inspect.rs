use super::*;

pub(super) fn inspect_value(
    request: &CallRequest,
    field: Field,
    solver: Option<OptimSolver>,
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if solver == Some(OptimSolver::Surrogateopt) {
        surrogate::inspect(request, field, value, index, diagnostics);
        return;
    }
    if solver == Some(OptimSolver::Ga) {
        genetic::inspect(request, field, value, index, diagnostics);
        return;
    }
    if solver == Some(OptimSolver::Simulannealbnd) {
        annealing::inspect(request, field, value, index, diagnostics);
        return;
    }
    if !solver_value_metadata(field, solver, value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPTIONS-TYPE",
            "option value has unsupported class or geometry",
            index,
        ));
    }
    // A dynamic function-handle selector does not establish a static profile.
    if solver.is_none() {
        return;
    }
    if solver == Some(OptimSolver::Intlinprog) {
        if matches!(field, Field::OutputFcn | Field::PlotFcn) {
            let kind = if field == Field::OutputFcn {
                crate::MilpObserverKind::SaveSolutions
            } else {
                crate::MilpObserverKind::PlotBounds
            };
            if request
                .literals
                .literal_string_at(index)
                .is_some_and(|name| name != kind.name())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MIP-CALLBACK",
                    "use a function handle or the selected predefined MILP observer name",
                    index,
                ));
            }
            return;
        }
        if field != Field::Display {
            if request
                .literals
                .numeric_at(index)
                .is_some_and(|number| !field.intlinprog_scalar_range(number))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MIP-RANGE",
                    "option is outside its intlinprog scalar range",
                    index,
                ));
            }
            return;
        }
    }
    if pattern::inspect(solver, request, field, index, diagnostics) {
        return;
    }
    if swarm::inspect(solver, request, field, index, diagnostics) {
        return;
    }
    if semi_infinite::inspect(request, field, index, diagnostics) {
        return;
    }
    match field {
        Field::HessianApproximation | Field::HessUpdate => {
            if let Some(text) = request.literals.literal_string_at(index) {
                if crate::OptimQuasiNewtonUpdate::parse(&text, false).is_none() {
                    diagnostics.push(argument_error("RM-CATALOG-QN-UPDATE","HessianApproximation must be bfgs/lbfgs or a valid limited-memory specification",index));
                }
            }
        }
        Field::ObjectiveLimit => {
            if request
                .literals
                .numeric_at(index)
                .is_some_and(|x| x.is_nan())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OBJECTIVE-LIMIT",
                    "ObjectiveLimit must not be NaN",
                    index,
                ));
            }
        }
        Field::FiniteDifferenceStepSize | Field::TypicalX => {
            if request
                .literals
                .numeric_at(index)
                .is_some_and(|x| !x.is_finite() || x <= 0.0)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-DERIVATIVE-SCALING",
                    "derivative scaling entries must be finite and strictly positive",
                    index,
                ));
            }
        }
        Field::UseParallel => {
            if request
                .literals
                .literal_string_at(index)
                .is_some_and(|s| !matches!(s.to_ascii_lowercase().as_str(), "off" | "on" | "auto"))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PARALLEL",
                    "UseParallel must be off/on/auto or logical",
                    index,
                ));
            }
        }
        Field::Display
        | Field::Algorithm
        | Field::HessianFcn
        | Field::SubproblemAlgorithm
        | Field::MeritFunction => {
            if let Some(text) = request.literals.literal_string_at(index) {
                let valid = if field == Field::Display {
                    let text = text.to_ascii_lowercase();
                    let detailed = text.strip_suffix("-detailed");
                    let base = detailed.unwrap_or(&text);
                    let allowed_detail = detailed.is_none()
                        || (solver == Some(OptimSolver::Fseminf)
                            && matches!(base, "iter" | "notify" | "final"));
                    allowed_detail
                        && OptimDisplay::parse(base).is_some_and(|display| {
                            solver.is_none_or(|solver| solver.accepts_display(display))
                        })
                } else if field == Field::Algorithm {
                    OptimAlgorithm::parse(&text).is_some_and(|algorithm| {
                        solver.is_none_or(|solver| solver.accepts_algorithm(algorithm))
                    })
                } else if field == Field::HessianFcn {
                    crate::OptimHessianSource::parse(&text).is_some()
                } else if field == Field::MeritFunction {
                    crate::OptimMeritFunction::parse(&text).is_some()
                } else {
                    crate::OptimSubproblemAlgorithm::parse(&text).is_some()
                };
                if !valid {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIONS-CONTROL",
                        "unsupported solver control value",
                        index,
                    ));
                }
            }
        }
        Field::TolX
        | Field::TolFun
        | Field::TolCon
        | Field::OptimalityTolerance
        | Field::FunctionTolerance
        | Field::TolPCG
        | Field::PrecondBandWidth
        | Field::MaxIter
        | Field::MaxFunEvals
        | Field::AbsoluteMaxObjectiveCount
        | Field::EqualityGoalCount
        | Field::MaxPCGIter => {
            if request.literals.numeric_at(index).is_some_and(|number| {
                let bandwidth_infinity =
                    field == Field::PrecondBandWidth && number == f64::INFINITY;
                !bandwidth_infinity
                    && (!number.is_finite()
                        || number < 0.0
                        || (number == 0.0
                            && solver.is_some_and(|solver| !field.allows_zero(solver)))
                        || (matches!(
                            field,
                            Field::MaxIter
                                | Field::MaxFunEvals
                                | Field::AbsoluteMaxObjectiveCount
                                | Field::EqualityGoalCount
                                | Field::MaxPCGIter
                                | Field::PrecondBandWidth
                        ) && number.fract() != 0.0))
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-RANGE",
                    "option must be finite, satisfy the selected solver's range, and counts must be integral",
                    index,
                ));
            }
            if matches!(
                field,
                Field::TolX
                    | Field::TolFun
                    | Field::TolCon
                    | Field::OptimalityTolerance
                    | Field::FunctionTolerance
                    | Field::TolPCG
            ) && request
                .literals
                .literal_args
                .get(index)
                .is_some_and(|literal| literal.integer_is_exact_binary64() == Some(false))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-EXACT",
                    "integer tolerances must be exactly representable as binary64",
                    index,
                ));
            }
        }
        Field::FiniteDifferenceType => {
            if request
                .literals
                .literal_string_at(index)
                .is_some_and(|value| crate::OptimDifferenceType::parse(&value).is_none())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-DIFFERENCE",
                    "FiniteDifferenceType must be forward or central",
                    index,
                ));
            }
        }
        Field::OutputFcn
        | Field::PlotFcn
        | Field::HessianMultiplyFcn
        | Field::JacobianMultiplyFcn
        | Field::HessPattern => {}
        Field::AbsoluteGapTolerance
        | Field::RelativeGapTolerance
        | Field::MaxFeasiblePoints
        | Field::MaxNodes
        | Field::MaxTime
        | Field::ObjectiveCutOff => {}
        Field::Diagnostics
        | Field::FunValCheck
        | Field::DiffMaxChange
        | Field::DiffMinChange
        | Field::MaxSQPIter
        | Field::RelLineSrchBnd
        | Field::RelLineSrchBndDuration
        | Field::TolConSQP => unreachable!("semi-infinite scalar controls were handled above"),
        Field::SpecifyObjectiveGradient | Field::SpecifyConstraintGradient => {
            if request
                .literals
                .numeric_at(index)
                .is_some_and(|value| value != 0.0 && value != 1.0)
                || request
                    .literals
                    .literal_string_at(index)
                    .is_some_and(|text| crate::OptimSwitch::parse(&text).is_none())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-FLAG",
                    "gradient control must be logical 0/1 or a supported on/off spelling",
                    index,
                ));
            }
        }
        _ => {}
    }
}
