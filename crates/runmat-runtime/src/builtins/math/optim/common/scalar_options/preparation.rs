//! Materialize only the controls consumed by the canonical solver option owner.

use super::super::{optim_error, ScalarCooperation};
use super::{PreparedSolverOptions, ScalarOptionFields};
use crate::BuiltinResult;
use runmat_builtins::{
    BuiltinExtensionDescriptor, OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver,
};
use runmat_value::{StructValue, Value};
mod plan;

pub(crate) async fn prepare_solver_options(
    solver: OptimSolver,
    options: Option<StructValue>,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<PreparedSolverOptions>> {
    prepare_with_inputs(
        solver,
        options,
        numeric_extension,
        resident_extension,
        runtime,
        cooperation,
        plan::InputContext::Unspecified,
    )
    .await
}

pub(crate) async fn prepare_modeled_fminunc_options(
    options: Option<StructValue>,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<PreparedSolverOptions>> {
    prepare_with_inputs(
        OptimSolver::Fminunc,
        options,
        numeric_extension,
        resident_extension,
        runtime,
        cooperation,
        plan::InputContext::FminuncModel,
    )
    .await
}

pub(crate) async fn prepare_modeled_residual_options(
    options: Option<StructValue>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<PreparedSolverOptions>> {
    prepare_with_inputs(
        OptimSolver::Lsqnonlin,
        options,
        None,
        None,
        runtime,
        cooperation,
        plan::InputContext::ResidualModel,
    )
    .await
}

pub(crate) async fn prepare_fmincon_options(
    options: Option<StructValue>,
    nonlinear_constraints: bool,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<PreparedSolverOptions>> {
    prepare_with_inputs(
        OptimSolver::Fmincon,
        options,
        numeric_extension,
        resident_extension,
        runtime,
        cooperation,
        plan::InputContext::Fmincon {
            nonlinear_constraints,
        },
    )
    .await
}

pub(crate) async fn prepare_multiobjective_options(
    solver: OptimSolver,
    options: Option<StructValue>,
    nonlinear_constraints: bool,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<PreparedSolverOptions>> {
    assert!(matches!(
        solver,
        OptimSolver::Fminimax | OptimSolver::Fgoalattain
    ));
    prepare_with_inputs(
        solver,
        options,
        numeric_extension,
        resident_extension,
        runtime,
        cooperation,
        plan::InputContext::Multiobjective {
            nonlinear_constraints,
        },
    )
    .await
}

async fn prepare_with_inputs(
    solver: OptimSolver,
    options: Option<StructValue>,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
    inputs: plan::InputContext,
) -> BuiltinResult<Option<PreparedSolverOptions>> {
    let Some(options) = options else {
        return Ok(None);
    };
    let source =
        crate::execution::RootedValue::in_optional_context(Value::Struct(options), runtime)?;
    let Value::Struct(options) = source.value() else {
        unreachable!("option input structure")
    };
    if solver == OptimSolver::Lsqnonlin {
        for (name, value) in &options.fields {
            if name.eq_ignore_ascii_case("Solver") {
                let text_carrier = match value {
                    Value::String(runmat_value::StringElement::Text(_)) => true,
                    Value::StringArray(value) => value.data().len() == 1,
                    Value::CharArray(value) => value.rows == 1,
                    _ => false,
                };
                let text = if text_carrier {
                    super::super::option_string_value(Some(value), "Solver", "")?
                } else {
                    return Err(optim_error(
                        solver.name(),
                        "options Solver must be a scalar solver name",
                    ));
                };
                if !OptimSolver::parse(&text).is_some_and(|profile| {
                    matches!(profile, OptimSolver::Generic | OptimSolver::Lsqnonlin)
                }) {
                    return Err(optim_error(
                        solver.name(),
                        "options Solver must be lsqnonlin or generic",
                    ));
                }
                continue;
            }
            if OptimOptionField::parse_for_solver(name, solver).is_some() {
                continue;
            }
            // A generic optimset record can carry other documented empty
            // reset markers, but never an active unimplemented control.
            if OptimOptionField::parse(name).is_some()
                && matches!(value, Value::Tensor(value) if value.is_empty())
            {
                continue;
            }
            return Err(optim_error(
                solver.name(),
                format!("unsupported lsqnonlin option '{name}'"),
            ));
        }
    }
    if solver != OptimSolver::Lsqnonlin
        && options.fields.iter().any(|(name, value)| {
            OptimOptionField::parse(name) == Some(OptimOptionField::JacobianMultiplyFcn)
                && !matches!(value,Value::Tensor(value) if value.is_empty())
        })
    {
        return Err(optim_error(
            solver.name(),
            "JacobianMultiplyFcn is unavailable for this solver profile",
        ));
    }
    // Preserve solver-specific public spelling semantics before canonical
    // scalar-field selection loses aliases (and before any control gather).
    for (name, value) in &options.fields {
        if OptimOptionField::parse(name).is_some_and(|field| {
            matches!(
                field,
                OptimOptionField::FiniteDifferenceStepSize | OptimOptionField::TypicalX
            ) && !solver.accepts(field)
                && !matches!(value,Value::Tensor(value) if value.is_empty())
        }) {
            return Err(optim_error(
                solver.name(),
                "derivative scaling controls are not implemented by this solver",
            ));
        }
        if OptimOptionField::parse(name).is_some_and(|field| solver.accepts(field))
            && OptimOptionField::parse_for_solver(name, solver).is_none()
        {
            return Err(optim_error(
                solver.name(),
                format!(
                    "{}: option {name} is not supported by this algorithm",
                    solver.name()
                ),
            ));
        }
    }
    let fields = if solver == OptimSolver::Lsqnonlin {
        ScalarOptionFields::from_struct_for_solver(Some(options), solver)
    } else {
        ScalarOptionFields::from_struct(Some(options))
    };
    use OptimOptionField::*;
    let mut plan = plan::prepare(
        solver,
        &fields,
        inputs,
        numeric_extension,
        resident_extension,
        runtime,
        cooperation,
    )
    .await?;
    let selected = OptimOptionField::ALL.map(|field| {
        if !solver.accepts(field) || !plan.applies(field) {
            return None;
        }
        fields
            .binding_for(solver, field)
            .map(|binding| (field, binding))
    });

    // After the necessary role-selection reads, validate every active control's
    // host-visible metadata before its remaining provider operations.
    for (field, binding) in selected.iter().flatten() {
        let value = binding.value;
        if matches!(field, HessianApproximation | HessUpdate) {
            super::super::quasi_newton_controls::update(solver.name(), *field, value)?;
        }
        if super::super::is_typed_integer_value(value)
            || matches!(value, Value::SparseTensor(t) if t.integer_storage().is_some())
        {
            if let Some(extension) = numeric_extension {
                crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                    runtime,
                    extension,
                    solver.name(),
                )?;
            }
        }
        super::super::preflight_host_residency(solver.name(), value, resident_extension, runtime)?;
        match field {
            CreationFcn
            | HybridFcn
            | DisplayInterval
            | InertiaRange
            | InitialPoints
            | InitialSwarmSpan
            | MaxStallIterations
            | MaxStallTime
            | MinNeighborsFraction
            | SelfAdjustmentWeight
            | SocialAdjustmentWeight
            | SwarmSize
            | UseVectorized
            | MeshTolerance
            | InitialMeshSize
            | MaxMeshSize
            | MeshContractionFactor
            | MeshExpansionFactor
            | MeshRotate
            | AccelerateMesh
            | ScaleMesh
            | PollMethod
            | PollOrderAlgorithm
            | UseCompletePoll
            | SearchFcn
            | UseCompleteSearch
            | TolBind
            | InitialPenalty
            | PenaltyFactor
            | Cache
            | CacheSize
            | CacheTol
            | PlotInterval
            | AcceptanceFcn
            | AnnealingFcn
            | DataType
            | HybridInterval
            | InitialTemperature
            | ReannealInterval
            | TemperatureFcn
            | PopulationType
            | PopulationSize
            | EliteCount
            | CrossoverFraction
            | InitialPopulationMatrix
            | InitialScoresMatrix
            | InitialPopulationRange
            | CrossoverFcn
            | MutationFcn
            | SelectionFcn
            | FitnessScalingFcn
            | NonlinearConstraintAlgorithm
            | MigrationDirection
            | MigrationFraction
            | MigrationInterval
            | StallTest
            | BatchUpdateInterval
            | CheckpointFile
            | MinSampleDistance
            | MinSurrogatePoints => {
                return Err(optim_error(
                    solver.name(),
                    "global-search controls require their selected solver owner",
                ))
            }
            Diagnostics
            | DiffMaxChange
            | DiffMinChange
            | FunValCheck
            | MaxSQPIter
            | RelLineSrchBnd
            | RelLineSrchBndDuration
            | TolConSQP => {
                if !matches!(field, Diagnostics | FunValCheck) {
                    super::super::preflight_option_scalar(solver.name(), value, *field)?;
                }
                if !matches!(value, Value::GpuTensor(_)) {
                    super::super::semi_options::validate(*field, value)?;
                } else if matches!(field, Diagnostics | FunValCheck) {
                    return Err(optim_error(
                        solver.name(),
                        "on/off options must be scalar text",
                    ));
                }
            }
            FiniteDifferenceStepSize | TypicalX => {
                super::super::derivative_scaling::metadata(solver.name(), value)?;
                if !matches!(value, Value::GpuTensor(_)) {
                    super::super::derivative_scaling::validate(solver.name(), value)?;
                }
            }
            HessianApproximation | HessUpdate => {
                super::super::quasi_newton_controls::update(solver.name(), *field, value)?;
                if super::super::quasi_newton_controls::history_operand(value)
                    .is_some_and(super::super::is_typed_integer_value)
                {
                    if let Some(extension) = numeric_extension {
                        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                            runtime,
                            extension,
                            solver.name(),
                        )?;
                    }
                }
            }
            UseParallel => {
                if matches!(value, Value::GpuTensor(_)) {
                    validate_switch_metadata(solver.name(), *field, value)?;
                } else {
                    super::super::quasi_newton_controls::parallel(solver.name(), value)?;
                }
            }
            ObjectiveLimit => {
                super::super::preflight_option_scalar(solver.name(), value, *field)?;
                if !matches!(value, Value::GpuTensor(_)) {
                    super::super::quasi_newton_controls::objective_limit(solver.name(), value)?;
                }
            }
            TolX
            | TolFun
            | TolCon
            | OptimalityTolerance
            | FunctionTolerance
            | MaxIter
            | MaxFunEvals
            | MaxPCGIter
            | AbsoluteMaxObjectiveCount
            | EqualityGoalCount
            | TolPCG
            | PrecondBandWidth
            | AbsoluteGapTolerance
            | RelativeGapTolerance
            | MaxFeasiblePoints
            | MaxNodes
            | MaxTime
            | ObjectiveCutOff => {
                super::super::preflight_option_scalar(solver.name(), value, *field)?;
                if !matches!(value, Value::GpuTensor(_)) {
                    match field {
                        TolX
                        | TolFun
                        | TolCon
                        | OptimalityTolerance
                        | FunctionTolerance
                        | TolPCG
                        | PrecondBandWidth
                        | MaxPCGIter
                        | MaxIter
                        | MaxFunEvals
                        | AbsoluteMaxObjectiveCount
                        | EqualityGoalCount
                        | AbsoluteGapTolerance
                        | RelativeGapTolerance
                        | MaxFeasiblePoints
                        | MaxNodes
                        | MaxTime
                        | ObjectiveCutOff => validate_numeric(solver, *field, value)?,
                        _ => unreachable!(),
                    }
                }
            }
            OutputFcn | PlotFcn => {
                if solver == OptimSolver::Fseminf && *field == PlotFcn {
                    super::super::output_callback::validate_for(
                        value,
                        super::super::output_callback::CallbackForm::HandlesOrNames,
                    )?;
                } else if solver == OptimSolver::Intlinprog {
                    super::super::output_callback::validate_for(
                        value,
                        super::super::output_callback::milp_form(*field),
                    )?;
                } else {
                    super::super::output_callback::validate(value)?;
                }
            }
            JacobianMultiplyFcn => {
                if !value.is_function_handle() {
                    return Err(optim_error(
                        solver.name(),
                        "JacobianMultiplyFcn must be a single function handle or []",
                    ));
                }
            }
            HessianMultiplyFcn => validate_hessian_multiply(solver.name(), value)?,
            HessianFcn => {
                hessian_source(solver.name(), value)?;
            }
            HessPattern => validate_pattern(solver.name(), value)?,
            SubproblemAlgorithm => {
                subproblem(solver.name(), value)?;
            }
            Display => {
                let text = super::super::option_string_value(Some(value), field.name(), "")?;
                if solver == OptimSolver::Fseminf {
                    super::super::semi_options::display(value)?;
                    continue;
                }
                if !OptimDisplay::parse(&text)
                    .is_some_and(|display| solver.accepts_display(display))
                {
                    return Err(optim_error(
                        solver.name(),
                        format!("{}: unsupported Display value '{text}'", solver.name()),
                    ));
                }
            }
            Algorithm => {
                let text = super::super::option_string_value(Some(value), field.name(), "")?;
                if !OptimAlgorithm::parse(&text)
                    .is_some_and(|algorithm| solver.accepts_algorithm(algorithm))
                {
                    return Err(optim_error(
                        solver.name(),
                        format!("{}: unsupported Algorithm value '{text}'", solver.name()),
                    ));
                }
            }
            MeritFunction => {
                let text = super::super::option_string_value(Some(value), field.name(), "")?;
                if runmat_builtins::OptimMeritFunction::parse(&text).is_none() {
                    return Err(optim_error(
                        solver.name(),
                        "MeritFunction must be multiobj or singleobj",
                    ));
                }
            }
            FiniteDifferenceType => {
                let text = super::super::option_string_value(Some(value), field.name(), "")?;
                if runmat_builtins::OptimDifferenceType::parse(&text).is_none() {
                    return Err(optim_error(
                        solver.name(),
                        "FiniteDifferenceType must be forward or central",
                    ));
                }
            }
            SpecifyObjectiveGradient | SpecifyConstraintGradient => {
                if matches!(value, Value::GpuTensor(_)) {
                    validate_switch_metadata(solver.name(), *field, value)?;
                } else {
                    switch_value(solver.name(), field.name(), value)?;
                }
            }
        }
    }
    // Gathered controls are numeric/logical only; the original generation keeps
    // all other descendants rooted until the one final generation is published.
    let mut replacements: [Option<(&str, Value)>; OptimOptionField::ALL.len()] =
        std::array::from_fn(|_| None);
    for (index, selected) in selected.into_iter().enumerate() {
        let Some((field, binding)) = selected else {
            continue;
        };
        let value = binding.value;
        if !matches!(value, Value::GpuTensor(_)) {
            continue;
        }
        let materialized = if let Some(value) = plan.take(field) {
            value
        } else {
            match field {
                CreationFcn
                | HybridFcn
                | DisplayInterval
                | InertiaRange
                | InitialPoints
                | InitialSwarmSpan
                | MaxStallIterations
                | MaxStallTime
                | MinNeighborsFraction
                | SelfAdjustmentWeight
                | SocialAdjustmentWeight
                | SwarmSize
                | UseVectorized
                | MeshTolerance
                | InitialMeshSize
                | MaxMeshSize
                | MeshContractionFactor
                | MeshExpansionFactor
                | MeshRotate
                | AccelerateMesh
                | ScaleMesh
                | PollMethod
                | PollOrderAlgorithm
                | UseCompletePoll
                | SearchFcn
                | UseCompleteSearch
                | TolBind
                | InitialPenalty
                | PenaltyFactor
                | Cache
                | CacheSize
                | CacheTol
                | PlotInterval
                | AcceptanceFcn
                | AnnealingFcn
                | DataType
                | HybridInterval
                | InitialTemperature
                | ReannealInterval
                | TemperatureFcn
                | PopulationType
                | PopulationSize
                | EliteCount
                | CrossoverFraction
                | InitialPopulationMatrix
                | InitialScoresMatrix
                | InitialPopulationRange
                | CrossoverFcn
                | MutationFcn
                | SelectionFcn
                | FitnessScalingFcn
                | NonlinearConstraintAlgorithm
                | MigrationDirection
                | MigrationFraction
                | MigrationInterval
                | StallTest
                | BatchUpdateInterval
                | CheckpointFile
                | MinSampleDistance
                | MinSurrogatePoints => {
                    return Err(optim_error(
                        solver.name(),
                        "global-search controls require their selected solver owner",
                    ))
                }
                DiffMaxChange | DiffMinChange | RelLineSrchBnd | TolConSQP | TolX | TolFun
                | TolCon | OptimalityTolerance | FunctionTolerance | TolPCG | PrecondBandWidth
                | AbsoluteGapTolerance | RelativeGapTolerance | MaxTime | ObjectiveCutOff
                | ObjectiveLimit => {
                    super::super::gather_exact_floating_value_with_cooperation(
                        solver.name(),
                        value.clone(),
                        field.name(),
                        cooperation,
                    )
                    .await?
                }
                MaxSQPIter
                | RelLineSrchBndDuration
                | MaxIter
                | MaxFunEvals
                | AbsoluteMaxObjectiveCount
                | EqualityGoalCount
                | MaxPCGIter
                | MaxFeasiblePoints
                | MaxNodes
                | FiniteDifferenceStepSize
                | TypicalX
                | SpecifyObjectiveGradient
                | SpecifyConstraintGradient
                | UseParallel
                | HessPattern => {
                    crate::dispatcher::gather_if_needed_in_context_async(
                        value,
                        cooperation.runtime(),
                    )
                    .await?
                }
                Diagnostics | FunValCheck | Display | Algorithm | FiniteDifferenceType
                | MeritFunction | OutputFcn | PlotFcn | JacobianMultiplyFcn
                | HessianMultiplyFcn | HessianFcn | SubproblemAlgorithm | HessianApproximation
                | HessUpdate => {
                    unreachable!("text controls were admitted before provider access")
                }
            }
        };
        match field {
            Diagnostics
            | DiffMaxChange
            | DiffMinChange
            | FunValCheck
            | MaxSQPIter
            | RelLineSrchBnd
            | RelLineSrchBndDuration
            | TolConSQP => super::super::semi_options::validate(field, &materialized)?,
            FiniteDifferenceStepSize | TypicalX => {
                super::super::derivative_scaling::validate(solver.name(), &materialized)?;
            }
            ObjectiveLimit => {
                super::super::quasi_newton_controls::objective_limit(solver.name(), &materialized)?;
            }
            UseParallel => {
                super::super::quasi_newton_controls::parallel(solver.name(), &materialized)?;
            }
            MaxIter
            | MaxFunEvals
            | MaxPCGIter
            | AbsoluteMaxObjectiveCount
            | EqualityGoalCount
            | MaxFeasiblePoints
            | MaxNodes => validate_numeric(solver, field, &materialized)?,
            SpecifyObjectiveGradient | SpecifyConstraintGradient => {
                switch_value(solver.name(), field.name(), &materialized)?;
            }
            TolX | TolFun | TolCon | OptimalityTolerance | FunctionTolerance | TolPCG
            | PrecondBandWidth | AbsoluteGapTolerance | RelativeGapTolerance | MaxTime
            | ObjectiveCutOff => validate_numeric(solver, field, &materialized)?,
            HessPattern => validate_pattern(solver.name(), &materialized)?,
            _ => unreachable!(),
        }
        replacements[index] = Some((binding.name, materialized));
        cooperation.completed(1).await?;
    }
    let output = if replacements.iter().any(Option::is_some) {
        let fields = options.fields.iter().map(|(name, original)| {
            let value = replacements
                .iter()
                .flatten()
                .find_map(|(replaced, value)| (name == *replaced).then_some(value))
                .unwrap_or(original);
            (name, value)
        });
        crate::execution::construct_struct_from_fields_in_optional_context(runtime, fields)?
    } else {
        source
    };
    Ok(Some(PreparedSolverOptions(output)))
}

fn validate_switch_metadata(
    name: &str,
    field: OptimOptionField,
    value: &Value,
) -> BuiltinResult<()> {
    let fact = crate::value_fact::value_fact(value);
    let real = runmat_builtins::nonlinear_real_numeric_metadata(&fact, false)
        || (matches!(fact.kind, runmat_types::ValueKindFact::Logical)
            && fact.storage != runmat_types::StorageFact::Sparse);
    if real && fact.shape.element_count() == Some(1) {
        Ok(())
    } else {
        Err(optim_error(
            name,
            format!("{} must be a logical scalar", field.name()),
        ))
    }
}

fn validate_numeric(
    solver: OptimSolver,
    field: OptimOptionField,
    value: &Value,
) -> BuiltinResult<()> {
    use OptimOptionField::*;
    let invalid = || {
        optim_error(
            solver.name(),
            format!("{}: invalid {} range", solver.name(), field.name()),
        )
    };
    if solver == OptimSolver::Intlinprog {
        if matches!(field, MaxFeasiblePoints | MaxNodes) {
            if field == MaxFeasiblePoints && infinite_bandwidth(value) {
                return Ok(());
            }
            let count =
                super::super::option_usize_value(solver.name(), Some(value), field.name(), 0)?;
            return if count > 0 { Ok(()) } else { Err(invalid()) };
        }
        let number = if infinite_bandwidth(value) {
            f64::INFINITY
        } else {
            super::super::option_f64_value(solver.name(), Some(value), field.name(), 0.0)?
        };
        return if field.intlinprog_scalar_range(number) {
            Ok(())
        } else {
            Err(invalid())
        };
    }
    if matches!(
        field,
        MaxIter
            | MaxFunEvals
            | MaxPCGIter
            | AbsoluteMaxObjectiveCount
            | EqualityGoalCount
            | PrecondBandWidth
    ) {
        if field == PrecondBandWidth && infinite_bandwidth(value) {
            return Ok(());
        }
        let count = super::super::option_usize_value(solver.name(), Some(value), field.name(), 0)?;
        if count == 0 && !field.allows_zero(solver) {
            return Err(invalid());
        }
    } else {
        let number = super::super::option_f64_value(solver.name(), Some(value), field.name(), 0.0)?;
        let zero = field.allows_zero(solver);
        if number < 0.0 || (number == 0.0 && !zero) {
            return Err(invalid());
        }
    }
    Ok(())
}

fn infinite_bandwidth(value: &Value) -> bool {
    match value {
        Value::Num(n) => *n == f64::INFINITY,
        Value::Tensor(t) if t.len() == 1 && t.integer_storage().is_none() => {
            crate::builtins::common::tensor::tensor_value_f64(t, 0) == f64::INFINITY
        }
        _ => false,
    }
}

fn hessian_source(name: &str, value: &Value) -> BuiltinResult<runmat_builtins::OptimHessianSource> {
    if matches!(value, Value::Tensor(t) if t.is_empty()) {
        return Ok(runmat_builtins::OptimHessianSource::FiniteDifference);
    }
    let text = super::super::option_string_value(Some(value), "HessianFcn", "")?;
    runmat_builtins::OptimHessianSource::parse(&text)
        .ok_or_else(|| optim_error(name, "HessianFcn must be [] or 'objective'"))
}

fn subproblem(
    name: &str,
    value: &Value,
) -> BuiltinResult<runmat_builtins::OptimSubproblemAlgorithm> {
    let text = super::super::option_string_value(Some(value), "SubproblemAlgorithm", "")?;
    runmat_builtins::OptimSubproblemAlgorithm::parse(&text)
        .ok_or_else(|| optim_error(name, "SubproblemAlgorithm must be 'cg' or 'factorization'"))
}

fn validate_hessian_multiply(name: &str, value: &Value) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    ) {
        Ok(())
    } else {
        Err(optim_error(
            name,
            "HessianMultiplyFcn must be a single function handle or []",
        ))
    }
}

fn validate_pattern(name: &str, value: &Value) -> BuiltinResult<()> {
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
        return Err(optim_error(
            name,
            "HessPattern must be a real numeric or logical square matrix",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    let real = matches!(fact.kind, runmat_types::ValueKindFact::Numeric(n) if n.domain == runmat_types::NumericDomain::Real)
        || matches!(fact.kind, runmat_types::ValueKindFact::Logical);
    let square = fact
        .shape
        .known_dims()
        .is_some_and(|dims| dims.len() == 2 && dims[0] == dims[1]);
    if real && square {
        Ok(())
    } else {
        Err(optim_error(
            name,
            "HessPattern must be a real numeric or logical square matrix",
        ))
    }
}

pub(crate) fn switch_value(name: &str, field: &str, value: &Value) -> BuiltinResult<bool> {
    use crate::builtins::common::tensor;
    let numeric = |number: f64| match number {
        0.0 => Some(false),
        1.0 => Some(true),
        _ => None,
    };
    let text = |text: &str| runmat_builtins::OptimSwitch::parse(text).map(|value| value.enabled());
    let parsed = if let Some(integer) = tensor::scalar_integer_value(value) {
        match integer.try_to_u64() {
            Some(0) => Some(false),
            Some(1) => Some(true),
            _ => None,
        }
    } else {
        match value {
            Value::Bool(value) => Some(*value),
            Value::Num(number) => numeric(*number),
            Value::LogicalArray(values) if values.data.len() == 1 => Some(values.data[0] != 0),
            Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
                numeric(tensor::tensor_value_f64(value, 0))
            }
            Value::String(runmat_value::StringElement::Text(value)) => text(value),
            Value::StringArray(values) if values.data().len() == 1 => {
                values.data()[0].as_str().and_then(text)
            }
            Value::CharArray(chars) if chars.rows == 1 => text(
                &runmat_value::decode_utf16_code_units(chars.data()).map_err(|source| {
                    crate::build_runtime_error(source.to_string())
                        .with_source(source)
                        .build()
                })?,
            ),
            _ => None,
        }
    };
    parsed.ok_or_else(|| {
        optim_error(
            name,
            format!("{name}: option {field} must be a logical scalar or on/off switch"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn switches_reject_missing_without_conflating_literal_text() {
        for (element, expected) in [
            (runmat_value::StringElement::text("on"), Some(true)),
            (runmat_value::StringElement::text("off"), Some(false)),
            (runmat_value::StringElement::text("<missing>"), None),
            (runmat_value::StringElement::Missing, None),
        ] {
            let scalar = Value::String(element.clone());
            let array = Value::StringArray(
                runmat_value::StringArray::from_elements_unbounded(vec![element], vec![1, 1])
                    .unwrap(),
            );
            for value in [scalar, array] {
                assert_eq!(switch_value("fsolve", "FunValCheck", &value).ok(), expected);
            }
        }
    }

    #[test]
    fn preparation_preserves_public_keys_and_ignores_unconsumed_controls() {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert("StepTolerance", Value::Num(0.125));
        options.insert("Algorithm", Value::Num(42.0));
        options.insert(
            "UnknownResidentLikeField",
            Value::String("untouched".into()),
        );
        let mut cooperation = ScalarCooperation::new(OptimSolver::Fsolve, None);
        let options = options.finish();
        let prepared = block_on(prepare_solver_options(
            OptimSolver::Fsolve,
            Some(options),
            None,
            None,
            None,
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        assert!(matches!(
            prepared.structure().fields.get("StepTolerance"),
            Some(Value::Num(0.125))
        ));
        assert!(matches!(
            prepared.structure().fields.get("Algorithm"),
            Some(Value::Num(42.0))
        ));
        assert!(
            matches!(prepared.structure().fields.get("UnknownResidentLikeField"), Some(Value::String(value)) if value == "untouched")
        );
        assert!(prepared.structure().fields.get("TolX").is_none());
    }
    #[test]
    fn derivative_scaling_is_selected_only_for_missing_derivatives() {
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("SpecifyObjectiveGradient", Value::Bool(true));
        fields.insert("FiniteDifferenceStepSize", Value::from("dormant"));
        fields.insert("TypicalX", Value::Num(-1.0));
        let fields = fields.finish();
        let mut cooperation = ScalarCooperation::new(OptimSolver::Fsolve, None);
        let prepared = block_on(prepare_solver_options(
            OptimSolver::Fsolve,
            Some(fields.clone()),
            None,
            None,
            None,
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        assert!(matches!(
            prepared.structure().fields.get("TypicalX"),
            Some(Value::Num(-1.0))
        ));
        let mut active = runmat_value::UnboundedStructValueBuilder::new();
        for (name, value) in &fields.fields {
            active.insert(
                name,
                if name == "SpecifyObjectiveGradient" {
                    Value::Bool(false)
                } else {
                    value.clone()
                },
            );
        }
        assert!(block_on(prepare_solver_options(
            OptimSolver::Fsolve,
            Some(active.finish()),
            None,
            None,
            None,
            &mut cooperation
        ))
        .is_err());
    }
}
