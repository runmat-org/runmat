use super::*;

pub(super) struct BayesoptOutcome {
    pub(super) variables: Vec<Variable>,
    pub(super) options: BayesoptOptions,
    pub(super) observations: Vec<Observation>,
    pub(super) surrogate: SurrogateSummary,
    pub(super) total_time: f64,
    pub(super) exhausted_candidates: bool,
}

pub(super) fn finalize(
    runtime: Option<&crate::context::RuntimeContext>,
    outcome: BayesoptOutcome,
    requested_outputs: Option<usize>,
) -> BuiltinResult<Value> {
    let result = result_object(runtime, outcome)?;
    match requested_outputs {
        None => Ok(result),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![result])),
        Some(requested) => Err(bayesopt_error(
            format!("bayesopt: requested {requested} outputs, but only one is supported"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn result_object(
    runtime: Option<&crate::context::RuntimeContext>,
    outcome: BayesoptOutcome,
) -> BuiltinResult<Value> {
    let best_index = best_observation_index(&outcome.observations).ok_or_else(|| {
        bayesopt_error(
            "bayesopt: no finite objective observations were recorded",
            &ERROR_OBJECTIVE,
        )
    })?;
    let best = &outcome.observations[best_index];
    let x_trace = trace_table(&outcome.variables, &outcome.observations)?;
    let x_at_min = best.point.to_table(&outcome.variables)?;
    let objective_trace = tensor_column(
        outcome
            .observations
            .iter()
            .map(|obs| obs.objective)
            .collect(),
    )?;
    let objective_minimum_trace = tensor_column(best_so_far_trace(&outcome.observations))?;
    let feasibility_trace = LogicalArray::new(
        outcome
            .observations
            .iter()
            .map(|obs| u8::from(obs.feasible()))
            .collect(),
        vec![outcome.observations.len(), 1],
    )
    .map(Value::LogicalArray)
    .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))?;
    let objective_time_trace = tensor_column(
        outcome
            .observations
            .iter()
            .map(|obs| obs.objective_time)
            .collect(),
    )?;
    let optimization_trace = optimization_trace_table(&outcome.observations)?;

    let mut output = runmat_value::UnboundedStructValueBuilder::new();
    output.insert("iterations", Value::Num(outcome.observations.len() as f64));
    output.insert("funcCount", Value::Num(outcome.observations.len() as f64));
    output.insert(
        "algorithm",
        Value::String("deterministic Gaussian-process expected improvement".into()),
    );
    output.insert(
        "message",
        Value::String(
            if outcome.exhausted_candidates {
                "Optimization completed after exhausting the finite candidate set."
            } else {
                "Optimization completed MaxObjectiveEvaluations."
            }
            .into(),
        ),
    );

    let mut surrogate = runmat_value::UnboundedStructValueBuilder::new();
    surrogate.insert("Kernel", Value::String("squared-exponential".into()));
    surrogate.insert("LengthScale", Value::Num(outcome.surrogate.length_scale));
    surrogate.insert("Noise", Value::Num(outcome.surrogate.noise));

    let fields = [
        ("XTrace", x_trace),
        ("ObjectiveTrace", objective_trace),
        ("ObjectiveMinimumTrace", objective_minimum_trace),
        ("ObjectiveEvaluationTimeTrace", objective_time_trace),
        ("FeasibilityTrace", feasibility_trace),
        ("OptimizationTrace", optimization_trace),
        ("XAtMinObjective", x_at_min),
        ("MinObjective", Value::Num(best.objective)),
        ("IndexOfMinimumTrace", Value::Num((best_index + 1) as f64)),
        (
            "NumObjectiveEvaluations",
            Value::Num(outcome.observations.len() as f64),
        ),
        ("TotalElapsedTime", Value::Num(outcome.total_time)),
        ("Options", Value::Struct(outcome.options.raw_options)),
        ("SurrogateModel", Value::Struct(surrogate.finish())),
        (
            "VariableDescriptions",
            variable_descriptions(runtime, &outcome.variables)?,
        ),
        ("Output", Value::Struct(output.finish())),
        ("ExitFlag", Value::Num(1.0)),
        ("BestPoint", best.point.to_table(&outcome.variables)?),
        ("BestObjective", Value::Num(best.objective)),
        (
            "IsObjectiveDeterministic",
            Value::Bool(outcome.options.deterministic),
        ),
        ("Verbose", Value::Bool(outcome.options.verbose)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        RESULT_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

fn best_observation_index(observations: &[Observation]) -> Option<usize> {
    observations
        .iter()
        .enumerate()
        .filter(|(_, obs)| obs.feasible() && obs.objective.is_finite())
        .min_by(|(_, a), (_, b)| {
            a.objective
                .partial_cmp(&b.objective)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .or_else(|| {
            observations
                .iter()
                .enumerate()
                .filter(|(_, obs)| obs.objective.is_finite())
                .min_by(|(_, a), (_, b)| {
                    a.penalized_objective()
                        .partial_cmp(&b.penalized_objective())
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(idx, _)| idx)
        })
}

pub(super) fn best_feasible_objective(observations: &[Observation]) -> Option<f64> {
    observations
        .iter()
        .filter(|obs| obs.feasible() && obs.objective.is_finite())
        .map(|obs| obs.objective)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
}

fn best_so_far_trace(observations: &[Observation]) -> Vec<f64> {
    let mut best = f64::INFINITY;
    let mut trace = Vec::with_capacity(observations.len());
    for obs in observations {
        if obs.feasible() {
            best = best.min(obs.objective);
        }
        trace.push(best);
    }
    trace
}

fn trace_table(variables: &[Variable], observations: &[Observation]) -> BuiltinResult<Value> {
    let mut names = Vec::with_capacity(variables.len());
    let mut columns = Vec::with_capacity(variables.len());
    for (idx, variable) in variables.iter().enumerate() {
        names.push(variable.name.clone());
        match &variable.kind {
            VariableKind::Categorical { .. } => {
                let data = observations
                    .iter()
                    .map(|obs| match &obs.point.values[idx] {
                        VariableValue::Categorical(value) => value.clone(),
                        VariableValue::Numeric(value) => value.to_string(),
                    })
                    .collect::<Vec<_>>();
                columns.push(Value::StringArray(
                    StringArray::new(data, vec![observations.len(), 1]).map_err(|err| {
                        bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT)
                    })?,
                ));
            }
            _ => {
                let data = observations
                    .iter()
                    .map(|obs| match &obs.point.values[idx] {
                        VariableValue::Numeric(value) => *value,
                        VariableValue::Categorical(_) => f64::NAN,
                    })
                    .collect::<Vec<_>>();
                columns.push(tensor_column(data)?);
            }
        }
    }
    crate::builtins::table::table_from_columns(names, columns).map_err(|err| {
        bayesopt_error_with_source(
            format!(
                "bayesopt: trace table construction failed: {}",
                err.message()
            ),
            &ERROR_INVALID_ARGUMENT,
            err,
        )
    })
}

fn optimization_trace_table(observations: &[Observation]) -> BuiltinResult<Value> {
    let n = observations.len();
    let iterations = tensor_column((1..=n).map(|idx| idx as f64).collect())?;
    let objective = tensor_column(observations.iter().map(|obs| obs.objective).collect())?;
    let best = tensor_column(best_so_far_trace(observations))?;
    let feasible = LogicalArray::new(
        observations
            .iter()
            .map(|obs| u8::from(obs.feasible()))
            .collect(),
        vec![n, 1],
    )
    .map(Value::LogicalArray)
    .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))?;
    crate::builtins::table::table_from_columns(
        vec![
            "Iteration".into(),
            "Objective".into(),
            "BestSoFar".into(),
            "Feasible".into(),
        ],
        vec![iterations, objective, best, feasible],
    )
    .map_err(|err| {
        bayesopt_error_with_source(
            format!(
                "bayesopt: optimization trace construction failed: {}",
                err.message()
            ),
            &ERROR_INVALID_ARGUMENT,
            err,
        )
    })
}

fn variable_descriptions(
    runtime: Option<&crate::context::RuntimeContext>,
    variables: &[Variable],
) -> BuiltinResult<Value> {
    let values = variables
        .iter()
        .map(|var| {
            let fields = [
                ("Name", Value::string(var.name.clone())),
                (
                    "Type",
                    Value::String(
                        match &var.kind {
                            VariableKind::Real { .. } => VariableType::Real,
                            VariableKind::Integer { .. } => VariableType::Integer,
                            VariableKind::Categorical { .. } => VariableType::Categorical,
                        }
                        .public_name()
                        .into(),
                    ),
                ),
                (
                    "Transform",
                    Value::String(var.transform.public_name().into()),
                ),
                ("Optimize", Value::Bool(var.optimize)),
                ("Range", variable_range_value(var)?),
            ];
            let object = crate::execution::object::construct_object_from_borrowed_fields(
                runtime,
                OPTIMIZABLE_VARIABLE_CLASS.display_name(),
                fields.iter().map(|(name, value)| (*name, value)),
            )?;
            Ok(object.into_value_and_lease().0)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    CellArray::new(values, variables.len(), 1)
        .map(Value::Cell)
        .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
}

fn variable_range_value(var: &Variable) -> BuiltinResult<Value> {
    match &var.kind {
        VariableKind::Real { lower, upper } => tensor_column(vec![*lower, *upper]),
        VariableKind::Integer { lower, upper } => tensor_column(vec![*lower as f64, *upper as f64]),
        VariableKind::Categorical { categories } => {
            StringArray::new(categories.clone(), vec![1, categories.len()])
                .map(Value::StringArray)
                .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
        }
    }
}

fn tensor_column(values: Vec<f64>) -> BuiltinResult<Value> {
    Tensor::new(values.clone(), vec![values.len(), 1])
        .map(Value::Tensor)
        .map_err(|err| bayesopt_error(format!("bayesopt: {err}"), &ERROR_INVALID_ARGUMENT))
}
