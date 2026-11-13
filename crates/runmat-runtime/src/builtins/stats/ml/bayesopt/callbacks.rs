use super::*;

pub(super) struct ObjectiveValue {
    pub(super) objective: f64,
    pub(super) constraint_violations: Vec<f64>,
}

pub(super) async fn parse_objective_value(value: Value) -> BuiltinResult<ObjectiveValue> {
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        bayesopt_error_with_source(
            format!(
                "bayesopt: objective output gather failed: {}",
                err.message()
            ),
            &ERROR_OBJECTIVE,
            err,
        )
    })?;
    match value {
        Value::Struct(st) => {
            let objective = st
                .fields
                .get("Objective")
                .or_else(|| st.fields.get("objective"))
                .cloned()
                .ok_or_else(|| {
                    bayesopt_error(
                        "bayesopt: objective struct must contain Objective",
                        &ERROR_OBJECTIVE,
                    )
                })?;
            let objective = objective_scalar(objective)?;
            let constraint_violations = match st
                .fields
                .get("ConstraintViolations")
                .or_else(|| st.fields.get("constraintViolations"))
                .or_else(|| st.fields.get("Constraints"))
            {
                Some(value) => finite_vector(value.clone(), "ConstraintViolations")?,
                None => Vec::new(),
            };
            Ok(ObjectiveValue {
                objective,
                constraint_violations,
            })
        }
        other => Ok(ObjectiveValue {
            objective: objective_scalar(other)?,
            constraint_violations: Vec::new(),
        }),
    }
}

pub(super) async fn constraint_allows(handle: &Value, row: Value) -> BuiltinResult<bool> {
    let value = call_function(handle, vec![row]).await.map_err(|err| {
        bayesopt_error_with_source(
            format!(
                "bayesopt: XConstraintFcn evaluation failed: {}",
                err.message()
            ),
            &ERROR_OBJECTIVE,
            err,
        )
    })?;
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        bayesopt_error_with_source(
            format!("bayesopt: XConstraintFcn gather failed: {}", err.message()),
            &ERROR_OBJECTIVE,
            err,
        )
    })?;
    match value {
        Value::Bool(value) => Ok(value),
        Value::LogicalArray(logical) if logical.data.len() == 1 => Ok(logical.data[0] != 0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            Ok(tensor::tensor_value_f64(&tensor, 0) != 0.0)
        }
        Value::Num(value) => Ok(value != 0.0),
        other => Err(bayesopt_error(
            format!("bayesopt: XConstraintFcn must return a logical scalar, got {other:?}"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}
