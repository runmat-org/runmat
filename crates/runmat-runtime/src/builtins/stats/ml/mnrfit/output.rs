use super::*;
use runmat_value::{StringArray, Tensor};

#[cfg(test)]
mod tests;

pub(super) async fn coefficients_value(
    beta: &[f64],
    rows: usize,
    cols: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Value> {
    Tensor::new(
        copy_cooperatively(beta, "coefficient output", check, cooperation).await?,
        vec![rows, cols],
    )
    .map(Value::Tensor)
    .map_err(|err| internal(format!("mnrfit: {err}")))
}

pub(super) async fn stats_value(
    fit: &NominalFit,
    data: &PreparedData,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Value> {
    check()?;
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    let class_count = data.classes.len();
    let logits = data.outcomes.len();
    st.insert(
        "beta",
        coefficients_value(&fit.beta, data.cols, logits, check, cooperation).await?,
    );
    st.insert(
        "se",
        coefficients_value(&fit.se, data.cols, logits, check, cooperation).await?,
    );
    st.insert(
        "t",
        coefficients_value(&fit.t, data.cols, logits, check, cooperation).await?,
    );
    st.insert(
        "p",
        coefficients_value(&fit.p, data.cols, logits, check, cooperation).await?,
    );
    st.insert(
        "covb",
        matrix_value(
            &fit.covb,
            fit.beta.len(),
            fit.beta.len(),
            check,
            cooperation,
        )
        .await?,
    );
    st.insert(
        "coeffcorr",
        coeffcorr_value(&fit.covb, &fit.se, check, cooperation).await?,
    );
    st.insert("dfe", Value::Num(fit.dfe));
    st.insert("s", Value::Num(1.0));
    st.insert("estdisp", Value::Bool(false));
    st.insert(
        "classNames",
        class_names_value(&data.classes, class_count, check, cooperation).await?,
    );
    check()?;
    Ok(Value::Struct(st.finish()))
}

async fn matrix_value(
    data: &[f64],
    rows: usize,
    cols: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Value> {
    Tensor::new(
        copy_cooperatively(data, "matrix output", check, cooperation).await?,
        vec![rows, cols],
    )
    .map(Value::Tensor)
    .map_err(|err| internal(format!("mnrfit: {err}")))
}

async fn row_tensor(
    values: &[f64],
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Value> {
    Tensor::new(
        copy_cooperatively(values, "class-name output", check, cooperation).await?,
        vec![1, values.len()],
    )
    .map(Value::Tensor)
    .map_err(|err| internal(format!("mnrfit: {err}")))
}

async fn class_names_value(
    classes: &ClassNames,
    len: usize,
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Value> {
    match classes {
        ClassNames::Numeric(values) => row_tensor(values, check, cooperation).await,
        ClassNames::Text(values) => {
            let mut names = Vec::new();
            names.try_reserve_exact(values.len()).map_err(|_| {
                invalid("mnrfit: class-name output exceeds available host resources")
            })?;
            for (index, name) in values.iter().enumerate() {
                names.push(name.clone());
                if (index + 1).is_multiple_of(4096) {
                    check()?;
                    cooperation.completed(4096).await?;
                }
            }
            cooperation.completed((values.len() % 4096) as u64).await?;
            StringArray::new(names, vec![1, len])
                .map(Value::StringArray)
                .map_err(|err| internal(format!("mnrfit: {err}")))
        }
    }
}

async fn coeffcorr_value(
    covb: &[f64],
    se: &[f64],
    check: &impl Fn() -> BuiltinResult<()>,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<Value> {
    check()?;
    let dim = se.len();
    let len = checked_elements(dim, dim, "coefficient correlation")?;
    let mut data =
        filled_cooperatively(len, 0.0, "coefficient correlation", check, cooperation).await?;
    for col in 0..dim {
        check()?;
        cooperation.completed(1).await?;
        for row in 0..dim {
            let denom = se[row] * se[col];
            data[row + col * dim] = if denom > 0.0 {
                covb[row + col * dim] / denom
            } else {
                f64::NAN
            };
            if (row + 1).is_multiple_of(16_384) {
                cooperation.completed(16_384).await?;
            }
        }
        cooperation.completed((dim % 16_384) as u64).await?;
    }
    matrix_value(&data, dim, dim, check, cooperation).await
}
