use super::*;

pub(super) fn model_object(
    runtime: Option<&crate::context::RuntimeContext>,
    spec: FitSpec,
    fit: OlsFit,
) -> BuiltinResult<Value> {
    let coefficients = coefficients_table(runtime, &spec.coefficient_names, &fit)?;
    let coefficient_names = string_row(spec.coefficient_names.clone())?;
    let formula = Value::string(spec.model_formula);
    let rsquared_values = [
        ("Ordinary", Value::Num(fit.r_squared)),
        ("Adjusted", Value::Num(fit.adjusted_r_squared)),
    ];
    let rsquared = rooted_struct(runtime, &rsquared_values)?;
    let criterion_values = [("MSE", Value::Num(fit.mse)), ("SSE", Value::Num(fit.sse))];
    let model_criterion = rooted_struct(runtime, &criterion_values)?;
    let fitted = Value::Tensor(
        Tensor::new(fit.fitted.clone(), vec![fit.fitted.len(), 1]).map_err(|err| {
            fitlm_internal(format!("fitlm: failed to construct fitted values: {err}"))
        })?,
    );
    let raw_residuals = Value::Tensor(
        Tensor::new(fit.residuals.clone(), vec![fit.residuals.len(), 1]).map_err(|err| {
            fitlm_internal(format!("fitlm: failed to construct residuals: {err}"))
        })?,
    );
    let residual_values = [("Raw", raw_residuals)];
    let residuals = rooted_struct(runtime, &residual_values)?;
    let predictor_names = string_row(spec.predictor_names.clone())?;
    let response_name = Value::string(spec.response_name);
    let terms = terms_tensor(&spec.terms)?;
    let beta = numeric_column(&fit.beta)?;
    let covariance = Value::Tensor(
        Tensor::new(
            fit.covariance.clone(),
            vec![spec.terms.len(), spec.terms.len()],
        )
        .map_err(|err| fitlm_internal(format!("fitlm: covariance shape error: {err}")))?,
    );
    let fields = [
        ("Coefficients", coefficients),
        ("CoefficientNames", coefficient_names),
        ("Formula", formula),
        ("NumObservations", Value::Num(fit.observations as f64)),
        ("DFE", Value::Num(fit.dfe)),
        ("RMSE", Value::Num(fit.rmse)),
        ("Rsquared", rsquared),
        ("ModelCriterion", model_criterion),
        ("Fitted", fitted),
        ("Residuals", residuals),
        ("PredictorNames", predictor_names),
        ("ResponseName", response_name),
        (
            "NumPredictors",
            Value::Num(spec.predictor_names.len() as f64),
        ),
        ("HasIntercept", Value::Bool(has_intercept(&spec.terms))),
        ("DesignRank", Value::Num(fit.rank as f64)),
        ("__RunMatTerms", terms),
        ("__RunMatBeta", beta),
        ("__RunMatCovariance", covariance),
        ("__RunMatMSE", Value::Num(fit.mse)),
        ("__RunMatDFE", Value::Num(fit.dfe)),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::LINEAR_MODEL.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

fn rooted_struct(
    runtime: Option<&crate::context::RuntimeContext>,
    fields: &[(&str, Value)],
) -> BuiltinResult<Value> {
    let value = crate::execution::construct_struct_from_fields_in_optional_context(
        runtime,
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(value.into_value_and_lease().0)
}

fn coefficients_table(
    runtime: Option<&crate::context::RuntimeContext>,
    names: &[String],
    fit: &OlsFit,
) -> BuiltinResult<Value> {
    let columns = vec![
        numeric_column(&fit.beta)?,
        numeric_column(&fit.se)?,
        numeric_column(&fit.tstat)?,
        numeric_column(&fit.pvalue)?,
    ];
    let table = table_from_columns(
        vec![
            "Estimate".to_string(),
            "SE".to_string(),
            "tStat".to_string(),
            "pValue".to_string(),
        ],
        columns,
    )
    .map_err(|err| fitlm_internal(format!("fitlm: {err}")))?;
    let mut object = match table {
        Value::Object(object) => object,
        _ => unreachable!("table_from_columns returns object"),
    };
    let row_names = string_column(names.to_vec())?;
    let mut updates = Vec::new();
    for name in ["Properties", "__table_properties"] {
        if let Some(Value::Struct(props)) = object.properties().get(name) {
            let fields = props
                .fields
                .iter()
                .filter(|(field, _)| *field != "RowNames")
                .chain(std::iter::once(("RowNames", &row_names)));
            let replacement = crate::execution::construct_struct_from_fields_in_optional_context(
                runtime, fields,
            )?;
            updates.push((name, replacement.into_value_and_lease().0));
        }
    }
    updates.push(("__CoefficientNames", row_names));
    crate::execution::object::replacement::replace_properties(
        runtime,
        &mut object,
        updates.iter().map(|(name, value)| (*name, value)),
        |detail, source| fitlm_internal(format!("fitlm: {detail}: {source}")),
        |detail| fitlm_internal(format!("fitlm: {detail}")),
    )?;
    Ok(Value::Object(object))
}
