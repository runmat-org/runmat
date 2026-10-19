use super::*;

pub(super) fn model_object(spec: FitSpec, fit: OlsFit) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(runmat_types::standard::LINEAR_MODEL.to_string());
    object.properties.insert(
        "Coefficients".to_string(),
        coefficients_table(&spec.coefficient_names, &fit)?,
    );
    object.properties.insert(
        "CoefficientNames".to_string(),
        string_row(spec.coefficient_names.clone())?,
    );
    object
        .properties
        .insert("Formula".to_string(), Value::String(spec.model_formula));
    object.properties.insert(
        "NumObservations".to_string(),
        Value::Num(fit.observations as f64),
    );
    object
        .properties
        .insert("DFE".to_string(), Value::Num(fit.dfe));
    object
        .properties
        .insert("RMSE".to_string(), Value::Num(fit.rmse));
    let mut rsquared = StructValue::new();
    rsquared.insert("Ordinary", Value::Num(fit.r_squared));
    rsquared.insert("Adjusted", Value::Num(fit.adjusted_r_squared));
    object
        .properties
        .insert("Rsquared".to_string(), Value::Struct(rsquared));
    let mut model_criterion = StructValue::new();
    model_criterion.insert("MSE", Value::Num(fit.mse));
    model_criterion.insert("SSE", Value::Num(fit.sse));
    object
        .properties
        .insert("ModelCriterion".to_string(), Value::Struct(model_criterion));
    object.properties.insert(
        "Fitted".to_string(),
        Value::Tensor(
            Tensor::new(fit.fitted.clone(), vec![fit.fitted.len(), 1]).map_err(|err| {
                fitlm_internal(format!("fitlm: failed to construct fitted values: {err}"))
            })?,
        ),
    );
    let mut residuals = StructValue::new();
    residuals.insert(
        "Raw",
        Value::Tensor(
            Tensor::new(fit.residuals.clone(), vec![fit.residuals.len(), 1]).map_err(|err| {
                fitlm_internal(format!("fitlm: failed to construct residuals: {err}"))
            })?,
        ),
    );
    object
        .properties
        .insert("Residuals".to_string(), Value::Struct(residuals));
    object.properties.insert(
        "PredictorNames".to_string(),
        string_row(spec.predictor_names.clone())?,
    );
    object.properties.insert(
        "ResponseName".to_string(),
        Value::String(spec.response_name),
    );
    object.properties.insert(
        "NumPredictors".to_string(),
        Value::Num(spec.predictor_names.len() as f64),
    );
    object.properties.insert(
        "HasIntercept".to_string(),
        Value::Bool(has_intercept(&spec.terms)),
    );
    object
        .properties
        .insert("DesignRank".to_string(), Value::Num(fit.rank as f64));
    object
        .properties
        .insert("__RunMatTerms".to_string(), terms_tensor(&spec.terms)?);
    object
        .properties
        .insert("__RunMatBeta".to_string(), numeric_column(&fit.beta)?);
    object.properties.insert(
        "__RunMatCovariance".to_string(),
        Value::Tensor(
            Tensor::new(
                fit.covariance.clone(),
                vec![spec.terms.len(), spec.terms.len()],
            )
            .map_err(|err| fitlm_internal(format!("fitlm: covariance shape error: {err}")))?,
        ),
    );
    object
        .properties
        .insert("__RunMatMSE".to_string(), Value::Num(fit.mse));
    object
        .properties
        .insert("__RunMatDFE".to_string(), Value::Num(fit.dfe));
    Ok(object)
}

fn coefficients_table(names: &[String], fit: &OlsFit) -> BuiltinResult<Value> {
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
    if let Some(Value::Struct(props)) = object.properties.get_mut("Properties") {
        props.insert("RowNames", row_names.clone());
    }
    if let Some(Value::Struct(props)) = object.properties.get_mut("__table_properties") {
        props.insert("RowNames", row_names.clone());
    }
    object
        .properties
        .insert("__CoefficientNames".to_string(), row_names);
    Ok(Value::Object(object))
}
