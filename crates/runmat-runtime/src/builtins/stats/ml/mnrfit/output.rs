use super::*;
use runmat_value::{StringArray, StructValue, Tensor};

#[cfg(test)]
mod tests;

pub(super) fn coefficients_value(
    beta: &[f64],
    rows: usize,
    cols: usize,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Value> {
    Tensor::new(
        copy_checked(beta, "coefficient output", check)?,
        vec![rows, cols],
    )
    .map(Value::Tensor)
    .map_err(|err| internal(format!("mnrfit: {err}")))
}

pub(super) fn stats_value(
    fit: &NominalFit,
    data: &PreparedData,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Value> {
    check()?;
    let mut st = StructValue::new();
    let class_count = data.classes.len();
    let logits = data.outcomes.len();
    st.insert(
        "beta",
        coefficients_value(&fit.beta, data.cols, logits, check)?,
    );
    st.insert("se", coefficients_value(&fit.se, data.cols, logits, check)?);
    st.insert("t", coefficients_value(&fit.t, data.cols, logits, check)?);
    st.insert("p", coefficients_value(&fit.p, data.cols, logits, check)?);
    st.insert(
        "covb",
        matrix_value(&fit.covb, fit.beta.len(), fit.beta.len(), check)?,
    );
    st.insert("coeffcorr", coeffcorr_value(&fit.covb, &fit.se, check)?);
    st.insert("dfe", Value::Num(fit.dfe));
    st.insert("s", Value::Num(1.0));
    st.insert("estdisp", Value::Bool(false));
    st.insert(
        "classNames",
        class_names_value(&data.classes, class_count, check)?,
    );
    check()?;
    Ok(Value::Struct(st))
}

fn matrix_value(
    data: &[f64],
    rows: usize,
    cols: usize,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Value> {
    Tensor::new(
        copy_checked(data, "matrix output", check)?,
        vec![rows, cols],
    )
    .map(Value::Tensor)
    .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn row_tensor(values: &[f64], check: &impl Fn() -> BuiltinResult<()>) -> BuiltinResult<Value> {
    Tensor::new(
        copy_checked(values, "class-name output", check)?,
        vec![1, values.len()],
    )
    .map(Value::Tensor)
    .map_err(|err| internal(format!("mnrfit: {err}")))
}

fn class_names_value(
    classes: &ClassNames,
    len: usize,
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Value> {
    match classes {
        ClassNames::Numeric(values) => row_tensor(values, check),
        ClassNames::Text(values) => {
            let mut names = Vec::new();
            names.try_reserve_exact(values.len()).map_err(|_| {
                invalid("mnrfit: class-name output exceeds available host resources")
            })?;
            for (index, name) in values.iter().enumerate() {
                if index.is_multiple_of(4096) {
                    check()?;
                }
                names.push(name.clone());
            }
            StringArray::new(names, vec![1, len])
                .map(Value::StringArray)
                .map_err(|err| internal(format!("mnrfit: {err}")))
        }
    }
}

fn coeffcorr_value(
    covb: &[f64],
    se: &[f64],
    check: &impl Fn() -> BuiltinResult<()>,
) -> BuiltinResult<Value> {
    check()?;
    let dim = se.len();
    let len = checked_elements(dim, dim, "coefficient correlation")?;
    let mut data = filled(len, 0.0, "coefficient correlation")?;
    for col in 0..dim {
        check()?;
        for row in 0..dim {
            let denom = se[row] * se[col];
            data[row + col * dim] = if denom > 0.0 {
                covb[row + col * dim] / denom
            } else {
                f64::NAN
            };
        }
    }
    matrix_value(&data, dim, dim, check)
}
