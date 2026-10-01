use super::*;

pub(super) fn coefficients_value(beta: &[f64], rows: usize, cols: usize) -> BuiltinResult<Value> {
    Tensor::new(beta.to_vec(), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("mnrfit: {err}")))
}

pub(super) fn stats_value(fit: &NominalFit, data: &PreparedData) -> BuiltinResult<Value> {
    let mut st = StructValue::new();
    let class_count = data.classes.len();
    let logits = data.outcomes.len();
    st.insert("beta", coefficients_value(&fit.beta, data.cols, logits)?);
    st.insert("se", coefficients_value(&fit.se, data.cols, logits)?);
    st.insert("t", coefficients_value(&fit.t, data.cols, logits)?);
    st.insert("p", coefficients_value(&fit.p, data.cols, logits)?);
    st.insert(
        "covb",
        matrix_value(&fit.covb, fit.beta.len(), fit.beta.len())?,
    );
    st.insert("coeffcorr", coeffcorr_value(&fit.covb, &fit.se)?);
    st.insert("dfe", Value::Num(fit.dfe));
    st.insert("s", Value::Num(1.0));
    st.insert("estdisp", Value::Bool(false));
    st.insert("classNames", class_names_value(&data.classes, class_count)?);
    Ok(Value::Struct(st))
}

fn matrix_value(data: &[f64], rows: usize, cols: usize) -> BuiltinResult<Value> {
    Tensor::new(data.to_vec(), vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn row_tensor(values: &[f64]) -> BuiltinResult<Value> {
    Tensor::new(values.to_vec(), vec![1, values.len()])
        .map(Value::Tensor)
        .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn class_names_value(classes: &ClassNames, len: usize) -> BuiltinResult<Value> {
    match classes {
        ClassNames::Numeric(values) => row_tensor(values),
        ClassNames::Text(values) => StringArray::new(values.clone(), vec![1, len])
            .map(Value::StringArray)
            .map_err(|err| internal(format!("mnrfit: {err}"))),
    }
}

fn coeffcorr_value(covb: &[f64], se: &[f64]) -> BuiltinResult<Value> {
    let dim = se.len();
    let len = dim
        .checked_mul(dim)
        .ok_or_else(|| internal("mnrfit: coefficient correlation dimension overflow"))?;
    let mut data = vec![0.0; len];
    for col in 0..dim {
        for row in 0..dim {
            let denom = se[row] * se[col];
            data[row + col * dim] = if denom > 0.0 {
                covb[row + col * dim] / denom
            } else {
                f64::NAN
            };
        }
    }
    matrix_value(&data, dim, dim)
}
