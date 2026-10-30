//! Typed constructor value validation and normalization.
use super::*;
pub(super) fn prepared_option_value(
    solver: Solver,
    field: OptimOptionField,
    value: &Value,
) -> BuiltinResult<state::PreparedOptionValue<'_>> {
    use state::PreparedOptionValue;
    if solver == Solver::Quadprog && matches!(value, Value::Tensor(t) if t.is_empty()) {
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Quadprog && field == OptimOptionField::MaxTime {
        crate::builtins::math::optim::common::global_time::parse(
            crate::builtins::math::optim::common::global_time::Policy {
                identity: runmat_builtins::OPTIMOPTIONS_CATALOG_ENTRY.identity,
                error: &runmat_builtins::OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            },
            value,
        )?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Surrogateopt {
        use crate::builtins::math::optim::common::surrogate_options as controls;
        controls::validate(controls::Policy::constructor(), field, value)?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Ga {
        crate::builtins::math::optim::ga::options::admission::validate(
            crate::builtins::math::optim::ga::options::Owner::Constructor,
            field,
            value,
        )?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Patternsearch {
        pattern::validate(field, value)?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Simulannealbnd {
        crate::builtins::math::optim::common::anneal_options::validate(
            crate::builtins::math::optim::common::anneal_options::Owner::Constructor,
            field,
            value,
        )?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Particleswarm {
        crate::builtins::math::optim::common::swarm_options::validate(
            crate::builtins::math::optim::common::swarm_options::Owner::Constructor,
            field,
            value,
        )?;
        return Ok(PreparedOptionValue::Callback(value));
    }
    if solver == Solver::Intlinprog {
        return intlinprog::prepared(field, value);
    }
    if matches!(value, Value::Tensor(t) if t.is_empty())
        && (matches!(
            solver,
            Solver::Fminunc
                | Solver::Fmincon
                | Solver::Fseminf
                | Solver::Fminimax
                | Solver::Fgoalattain
        ) || (solver == Solver::Lsqnonlin && field == OptimOptionField::MaxFunEvals)
            || field.trust_region_only()
            || field == OptimOptionField::OptimalityTolerance)
    {
        return Ok(PreparedOptionValue::Callback(value));
    }
    if matches!(
        field,
        OptimOptionField::FiniteDifferenceStepSize | OptimOptionField::TypicalX
    ) && matches!(value,Value::Tensor(t) if t.is_empty())
    {
        return Ok(PreparedOptionValue::Callback(value));
    }
    match field {
        OptimOptionField::BatchUpdateInterval
        | OptimOptionField::CheckpointFile
        | OptimOptionField::MinSampleDistance
        | OptimOptionField::MinSurrogatePoints => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "option requires surrogateopt",
        )),
        OptimOptionField::PopulationType
        | OptimOptionField::PopulationSize
        | OptimOptionField::EliteCount
        | OptimOptionField::CrossoverFraction
        | OptimOptionField::InitialPopulationMatrix
        | OptimOptionField::InitialScoresMatrix
        | OptimOptionField::InitialPopulationRange
        | OptimOptionField::CrossoverFcn
        | OptimOptionField::MutationFcn
        | OptimOptionField::SelectionFcn
        | OptimOptionField::FitnessScalingFcn
        | OptimOptionField::NonlinearConstraintAlgorithm
        | OptimOptionField::MigrationDirection
        | OptimOptionField::MigrationFraction
        | OptimOptionField::MigrationInterval
        | OptimOptionField::StallTest => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "option requires ga",
        )),

        OptimOptionField::MeshTolerance
        | OptimOptionField::InitialMeshSize
        | OptimOptionField::MaxMeshSize
        | OptimOptionField::MeshContractionFactor
        | OptimOptionField::MeshExpansionFactor
        | OptimOptionField::MeshRotate
        | OptimOptionField::AccelerateMesh
        | OptimOptionField::ScaleMesh
        | OptimOptionField::PollMethod
        | OptimOptionField::PollOrderAlgorithm
        | OptimOptionField::UseCompletePoll
        | OptimOptionField::SearchFcn
        | OptimOptionField::UseCompleteSearch
        | OptimOptionField::TolBind
        | OptimOptionField::InitialPenalty
        | OptimOptionField::PenaltyFactor
        | OptimOptionField::Cache
        | OptimOptionField::CacheSize
        | OptimOptionField::CacheTol
        | OptimOptionField::PlotInterval => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "option requires patternsearch",
        )),
        OptimOptionField::AcceptanceFcn
        | OptimOptionField::AnnealingFcn
        | OptimOptionField::DataType
        | OptimOptionField::HybridInterval
        | OptimOptionField::InitialTemperature
        | OptimOptionField::ReannealInterval
        | OptimOptionField::TemperatureFcn
        | OptimOptionField::CreationFcn
        | OptimOptionField::HybridFcn
        | OptimOptionField::DisplayInterval
        | OptimOptionField::InertiaRange
        | OptimOptionField::InitialPoints
        | OptimOptionField::InitialSwarmSpan
        | OptimOptionField::MaxStallIterations
        | OptimOptionField::MaxStallTime
        | OptimOptionField::MinNeighborsFraction
        | OptimOptionField::SelfAdjustmentWeight
        | OptimOptionField::SocialAdjustmentWeight
        | OptimOptionField::SwarmSize
        | OptimOptionField::UseVectorized => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "option requires particleswarm",
        )),
        OptimOptionField::Diagnostics
        | OptimOptionField::DiffMaxChange
        | OptimOptionField::DiffMinChange
        | OptimOptionField::FunValCheck
        | OptimOptionField::MaxSQPIter
        | OptimOptionField::RelLineSrchBnd
        | OptimOptionField::RelLineSrchBndDuration
        | OptimOptionField::TolConSQP => {
            crate::builtins::math::optim::common::semi_options::validate(field, value)?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::PlotFcn if solver == Solver::Fseminf => {
            crate::builtins::math::optim::common::output_callback::validate_for(
                value,
                crate::builtins::math::optim::common::output_callback::CallbackForm::HandlesOrNames,
            )?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::Display if solver == Solver::Fseminf => {
            crate::builtins::math::optim::common::semi_options::display(value)?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::FiniteDifferenceStepSize | OptimOptionField::TypicalX => {
            crate::builtins::math::optim::common::derivative_scaling::validate(NAME, value)?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::HessianApproximation => {
            crate::builtins::math::optim::common::quasi_newton_controls::update(
                NAME, field, value,
            )?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::HessUpdate => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_UNKNOWN_OPTION,
            "optimoptions: HessUpdate is a legacy optimset option; use HessianApproximation",
        )),
        OptimOptionField::ObjectiveLimit => Ok(PreparedOptionValue::Number(
            crate::builtins::math::optim::common::quasi_newton_controls::objective_limit(
                NAME, value,
            )?,
        )),
        OptimOptionField::UseParallel => {
            crate::builtins::math::optim::common::quasi_newton_controls::parallel_enabled(
                NAME, value,
            )?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::AbsoluteGapTolerance
        | OptimOptionField::RelativeGapTolerance
        | OptimOptionField::MaxFeasiblePoints
        | OptimOptionField::MaxNodes
        | OptimOptionField::MaxTime
        | OptimOptionField::ObjectiveCutOff
        | OptimOptionField::PlotFcn => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "optimoptions: option requires intlinprog",
        )),
        OptimOptionField::TolX
        | OptimOptionField::TolFun
        | OptimOptionField::TolCon
        | OptimOptionField::OptimalityTolerance
        | OptimOptionField::FunctionTolerance
        | OptimOptionField::TolPCG => {
            let parsed = numeric_scalar(field.name(), value)?;
            let zero = field.allows_zero(solver);
            if parsed < 0.0 || (parsed == 0.0 && !zero) {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: invalid {} range", field.name()),
                ));
            }
            Ok(PreparedOptionValue::Number(parsed))
        }
        OptimOptionField::MaxIter
        | OptimOptionField::MaxFunEvals
        | OptimOptionField::AbsoluteMaxObjectiveCount
        | OptimOptionField::EqualityGoalCount
        | OptimOptionField::MaxPCGIter => {
            let count = positive_integer_scalar(field, value, solver)?;
            Ok(tensor::scalar_integer_value(value)
                .map(PreparedOptionValue::Integer)
                .unwrap_or_else(|| PreparedOptionValue::Number(count as f64)))
        }
        OptimOptionField::Display => {
            Ok(PreparedOptionValue::Display(display_value(solver, value)?))
        }
        OptimOptionField::Algorithm => Ok(PreparedOptionValue::Algorithm(algorithm_value(
            solver, value,
        )?)),
        OptimOptionField::FiniteDifferenceType => {
            let text = expect_string_scalar(
                value,
                "optimoptions: FiniteDifferenceType must be scalar text",
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            let mode = runmat_builtins::OptimDifferenceType::parse(&text).ok_or_else(|| {
                optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: FiniteDifferenceType must be forward or central",
                )
            })?;
            Ok(PreparedOptionValue::Difference(mode))
        }
        OptimOptionField::MeritFunction => {
            let text = expect_string_scalar(
                value,
                "optimoptions: MeritFunction must be scalar text",
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            let mode = runmat_builtins::OptimMeritFunction::parse(&text).ok_or_else(|| {
                optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: MeritFunction must be multiobj or singleobj",
                )
            })?;
            Ok(PreparedOptionValue::Merit(mode))
        }
        OptimOptionField::OutputFcn => {
            crate::builtins::math::optim::common::output_callback::validate(value)?;
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::SpecifyObjectiveGradient
        | OptimOptionField::SpecifyConstraintGradient => Ok(PreparedOptionValue::Logical(
            logical_value(field.name(), value)?,
        )),
        OptimOptionField::HessianFcn => {
            let text = expect_string_scalar(
                value,
                "optimoptions: HessianFcn must be [] or 'objective'",
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            let source = runmat_builtins::OptimHessianSource::parse(&text).ok_or_else(|| {
                optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: HessianFcn must be [] or 'objective'",
                )
            })?;
            Ok(PreparedOptionValue::Hessian(source))
        }
        OptimOptionField::JacobianMultiplyFcn => {
            if !value.is_function_handle() {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: JacobianMultiplyFcn must be a single function handle or []",
                ));
            }
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::HessianMultiplyFcn => {
            if !matches!(
                value,
                Value::ScopedFunctionHandle(_)
                    | Value::FunctionHandle(_)
                    | Value::ExternalFunctionHandle(_)
                    | Value::MethodFunctionHandle(_)
                    | Value::BoundFunctionHandle(_)
                    | Value::Closure(_)
            ) {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: HessianMultiplyFcn must be a single function handle or []",
                ));
            }
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::HessPattern => {
            if !pattern_metadata(value) {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "optimoptions: HessPattern must be a real numeric or logical square matrix",
                ));
            }
            Ok(PreparedOptionValue::Callback(value))
        }
        OptimOptionField::SubproblemAlgorithm => {
            let text = expect_string_scalar(
                value,
                "optimoptions: SubproblemAlgorithm must be scalar text",
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            let algorithm =
                runmat_builtins::OptimSubproblemAlgorithm::parse(&text).ok_or_else(|| {
                    optimoptions_error_with(
                        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "optimoptions: SubproblemAlgorithm must be 'cg' or 'factorization'",
                    )
                })?;
            Ok(PreparedOptionValue::Subproblem(algorithm))
        }
        OptimOptionField::PrecondBandWidth => {
            if matches!(value, Value::Num(n) if *n == f64::INFINITY)
                || matches!(value, Value::Tensor(t) if t.len() == 1 && t.integer_storage().is_none() && tensor::tensor_value_f64(t, 0) == f64::INFINITY)
            {
                return Ok(PreparedOptionValue::Number(f64::INFINITY));
            }
            let count = crate::builtins::math::optim::common::option_usize_value(
                NAME,
                Some(value),
                field.name(),
                0,
            )?;
            Ok(tensor::scalar_integer_value(value)
                .map(PreparedOptionValue::Integer)
                .unwrap_or_else(|| PreparedOptionValue::Number(count as f64)))
        }
    }
}

pub(super) fn pattern_metadata(value: &Value) -> bool {
    if !matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::SparseTensor(_)
            | Value::GpuTensor(_)
    ) {
        return false;
    }
    let fact = crate::value_fact::value_fact(value);
    let real = matches!(fact.kind, runmat_types::ValueKindFact::Numeric(n) if n.domain == runmat_types::NumericDomain::Real)
        || matches!(fact.kind, runmat_types::ValueKindFact::Logical);
    real && (fact.shape.element_count() == Some(1)
        || fact
            .shape
            .known_dims()
            .is_some_and(|dims| dims.len() == 2 && dims[0] == dims[1]))
}

pub(super) fn solver_label(solver: Solver) -> &'static str {
    match solver {
        Solver::Generic => "optimization solvers",
        _ => solver.name(),
    }
}

pub(super) fn positive_finite_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    let parsed = numeric_scalar(field, value)?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be a finite positive scalar"),
        ))
    }
}

pub(super) fn positive_integer_scalar(
    identity: OptimOptionField,
    value: &Value,
    solver: Solver,
) -> BuiltinResult<usize> {
    let zero = identity.allows_zero(solver);
    let field = identity.name();
    if let Some(integer) = tensor::scalar_integer_value(value) {
        if let Some(parsed) = integer.try_to_usize() {
            if parsed == 0 && !zero {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: option {field} must be a finite positive scalar"),
                ));
            }
            return Ok(parsed);
        }
        if integer.try_to_i64().is_some_and(|value| value < 0) {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be a finite positive scalar"),
            ));
        }
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} is too large"),
        ));
    }
    let parsed = if zero {
        let parsed = numeric_scalar(field, value)?;
        if parsed < 0.0 {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be nonnegative"),
            ));
        }
        parsed
    } else {
        positive_finite_scalar(field, value)?
    };
    if parsed.fract() != 0.0 {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be an integer scalar"),
        ));
    }
    if parsed >= 2f64.powi(usize::BITS as i32) {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} is too large"),
        ));
    }
    Ok(parsed as usize)
}

pub(super) fn numeric_scalar(field: &str, value: &Value) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::value_contains_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
    {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: integer option {field} must be exactly representable as double"),
        ));
    }
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be a numeric scalar, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be finite"),
        ))
    }
}

pub(super) fn logical_value(field: &str, value: &Value) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!("optimoptions: option {field} must be logical 0 or 1"),
            )),
        };
    }
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Num(n) => logical_from_number(field, *n),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            logical_from_number(field, tensor::tensor_value_f64(tensor, 0))
        }
        Value::String(runmat_value::StringElement::Text(s)) => logical_from_text(field, s),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            let text = sa.data()[0].as_str().ok_or_else(|| {
                optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: option {field} must be nonmissing text"),
                )
            })?;
            logical_from_text(field, text)
        }
        Value::CharArray(array) if array.rows == 1 => {
            let text = runmat_value::decode_utf16_code_units(array.data()).map_err(|error| {
                optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("optimoptions: option {field} contains {error}"),
                )
            })?;
            logical_from_text(field, &text)
        }
        other => Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be logical, got {other:?}"),
        )),
    }
}

pub(super) fn logical_from_number(field: &str, value: f64) -> BuiltinResult<bool> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!("optimoptions: option {field} must be logical 0 or 1"),
        ))
    }
}

pub(super) fn logical_from_text(field: &str, value: &str) -> BuiltinResult<bool> {
    OptimSwitch::parse(value)
        .map(OptimSwitch::enabled)
        .ok_or_else(|| {
            optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!(
                    "optimoptions: option {field} must be 'on' or 'off', got '{}'",
                    value.trim().to_ascii_lowercase()
                ),
            )
        })
}

pub(super) fn display_value(solver: Solver, value: &Value) -> BuiltinResult<OptimDisplay> {
    let display = expect_string_scalar(
        value,
        "optimoptions: Display must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?
    .trim()
    .to_ascii_lowercase();
    if let Some(display) =
        OptimDisplay::parse(&display).filter(|display| solver.accepts_display(*display))
    {
        Ok(display)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported Display '{}' for {}",
                display,
                solver_label(solver)
            ),
        ))
    }
}

pub(super) fn algorithm_value(solver: Solver, value: &Value) -> BuiltinResult<OptimAlgorithm> {
    let algorithm = expect_string_scalar(
        value,
        "optimoptions: Algorithm must be a character vector or string scalar",
        &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?
    .trim()
    .to_ascii_lowercase();
    if let Some(algorithm) =
        OptimAlgorithm::parse(&algorithm).filter(|algorithm| solver.accepts_algorithm(*algorithm))
    {
        Ok(algorithm)
    } else {
        Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "optimoptions: unsupported Algorithm '{}' for {}",
                algorithm,
                solver_label(solver)
            ),
        ))
    }
}
