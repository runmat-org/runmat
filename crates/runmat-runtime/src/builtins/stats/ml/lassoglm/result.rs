use super::*;

pub(super) fn coefficients_value(path: &FitPath) -> BuiltinResult<Value> {
    let rows = path
        .coefficients
        .first()
        .map(|coeffs| coeffs.len())
        .unwrap_or(0);
    let cols = path.coefficients.len();
    let mut data = Vec::with_capacity(rows * cols);
    for lambda_idx in 0..cols {
        for row in 0..rows {
            data.push(path.coefficients[lambda_idx][row]);
        }
    }
    Tensor::new(data, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

pub(super) fn row_tensor(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![1, values.len()])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

pub(super) fn scalar_tensor(value: f64) -> BuiltinResult<Value> {
    Tensor::new(vec![value], vec![1, 1])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

pub(super) fn logical_scalar(value: bool) -> BuiltinResult<Value> {
    LogicalArray::new(vec![if value { 1 } else { 0 }], vec![1, 1])
        .map(Value::LogicalArray)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}

pub(super) fn fit_info_value(
    path: &FitPath,
    distribution: Distribution,
    options: &Options,
    cv: Option<CvInfo>,
) -> BuiltinResult<Value> {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert("Intercept", row_tensor(&path.intercepts)?);
    st.insert("Lambda", row_tensor(&path.lambdas)?);
    st.insert("Alpha", Value::Num(options.alpha));
    st.insert(
        "Deviance",
        row_tensor(
            cv.as_ref()
                .map(|info| info.deviance.as_slice())
                .unwrap_or(&path.deviance),
        )?,
    );
    st.insert("DF", row_tensor(&path.df)?);
    st.insert("Iterations", row_tensor(&path.iterations)?);
    st.insert(
        "Distribution",
        Value::string(distribution.public_name().to_string()),
    );
    st.insert("PredictorNames", predictor_names_value(options)?);
    st.insert("UseCovariance", logical_scalar(false)?);
    if let Some(cv) = cv {
        st.insert("SE", row_tensor(&cv.se)?);
        st.insert("LambdaMinDeviance", scalar_tensor(cv.lambda_min_deviance)?);
        st.insert("Lambda1SE", scalar_tensor(cv.lambda_1se)?);
        st.insert("IndexMinDeviance", Value::Num(cv.index_min_deviance as f64));
        st.insert("Index1SE", Value::Num(cv.index_1se as f64));
    }
    Ok(Value::Struct(st.finish()))
}

pub(super) fn predictor_names_value(options: &Options) -> BuiltinResult<Value> {
    let names = options.predictor_names.as_deref().unwrap_or(&[]);
    let data = names
        .iter()
        .map(|name| Value::CharArray(CharArray::new_row(name)))
        .collect::<Vec<_>>();
    CellArray::new(data, 1, names.len())
        .map(Value::Cell)
        .map_err(|err| internal(format!("lassoglm: {err}")))
}
