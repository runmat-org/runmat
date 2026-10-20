use super::*;

pub(super) fn fit_info_value(
    spec: &FitSpec,
    lambdas: &[f64],
    models: &[TrainedModel],
) -> BuiltinResult<Value> {
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    info.insert("Lambda", row_tensor(lambdas.to_vec())?);
    info.insert(
        "Objective",
        row_tensor(models.iter().map(|model| model.objective).collect())?,
    );
    info.insert(
        "NumIterations",
        row_tensor(models.iter().map(|model| model.iterations as f64).collect())?,
    );
    info.insert(
        "NumPasses",
        row_tensor(models.iter().map(|model| model.iterations as f64).collect())?,
    );
    info.insert(
        "GradientNorm",
        row_tensor(models.iter().map(|model| model.gradient_norm).collect())?,
    );
    info.insert("PassLimit", Value::Num(spec.options.iteration_limit as f64));
    info.insert(
        "Solver",
        Value::String(spec.options.solver.public_name().into()),
    );
    info.insert(
        "Learner",
        Value::string(spec.options.learner.as_str().to_string()),
    );
    info.insert(
        "Regularization",
        Value::string(spec.options.regularization.as_str().to_string()),
    );
    info.insert("TerminationCode", row_tensor(vec![1.0; models.len()])?);
    info.insert(
        "TerminationStatus",
        Value::string("Iteration limit reached or gradient tolerance satisfied".to_string()),
    );
    Ok(Value::Struct(info.finish()))
}
