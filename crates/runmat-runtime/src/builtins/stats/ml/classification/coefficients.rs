use super::*;
use runmat_value::{CellArray, StructValue};

pub(super) fn coeff_value(model: &PreparedModel) -> BuiltinResult<Value> {
    let k = model.classes.len();
    let entry_count = k
        .checked_mul(k)
        .ok_or_else(|| internal_error("classify: coefficient cell shape overflows"))?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|err| internal_error(format!("classify: cannot allocate coefficients: {err}")))?;
    for row in 0..k {
        for col in 0..k {
            let coeff = pairwise_coeff(model, row, col)?;
            let mut st = StructValue::new();
            st.insert(
                "type",
                Value::String(model.discriminant.property_name().to_string()),
            );
            st.insert("name1", labels_value(&model.classes, model.kind, &[row])?);
            st.insert("name2", labels_value(&model.classes, model.kind, &[col])?);
            st.insert("const", Value::Num(coeff.0));
            st.insert(
                "linear",
                Value::Tensor(
                    Tensor::new(coeff.1, vec![model.cols, 1])
                        .map_err(|err| internal_error(format!("classify: {err}")))?,
                ),
            );
            if !matches!(
                model.discriminant,
                DiscriminantType::Linear | DiscriminantType::DiagLinear
            ) {
                st.insert(
                    "quadratic",
                    Value::Tensor(
                        Tensor::new(coeff.2, vec![model.cols, model.cols])
                            .map_err(|err| internal_error(format!("classify: {err}")))?,
                    ),
                );
            }
            entries.push(Value::Struct(st));
        }
    }
    Ok(Value::Cell(CellArray::new(entries, k, k).map_err(
        |err| internal_error(format!("classify: {err}")),
    )?))
}

pub(super) fn pairwise_coeff(
    model: &PreparedModel,
    i: usize,
    j: usize,
) -> BuiltinResult<(f64, Vec<f64>, Vec<f64>)> {
    let quadratic_len = model
        .cols
        .checked_mul(model.cols)
        .ok_or_else(|| internal_error("classify: quadratic coefficient shape overflows"))?;
    if i == j {
        let mut quadratic = Vec::new();
        quadratic.try_reserve_exact(quadratic_len).map_err(|err| {
            internal_error(format!(
                "classify: cannot allocate quadratic coefficients: {err}"
            ))
        })?;
        quadratic.resize(quadratic_len, 0.0);
        return Ok((0.0, vec![0.0; model.cols], quadratic));
    }
    let class_i = &model.classes[i];
    let class_j = &model.classes[j];
    let mean_i = mean_vector(&class_i.mean);
    let mean_j = mean_vector(&class_j.mean);
    let (inv_i, logdet_i) = if let Some(inv) = &model.shared_inverse {
        (inv, model.shared_log_det.unwrap_or(0.0))
    } else {
        (&class_i.inverse, class_i.log_det)
    };
    let (inv_j, logdet_j) = if let Some(inv) = &model.shared_inverse {
        (inv, model.shared_log_det.unwrap_or(0.0))
    } else {
        (&class_j.inverse, class_j.log_det)
    };
    let linear_vec = mean_i.transpose() * inv_i - mean_j.transpose() * inv_j;
    let const_term = if model.discriminant == DiscriminantType::Mahalanobis {
        -0.5 * ((mean_i.transpose() * inv_i * &mean_i)[(0, 0)]
            - (mean_j.transpose() * inv_j * &mean_j)[(0, 0)])
    } else {
        log_prior_ratio(model.priors[i], model.priors[j])
            - 0.5 * (logdet_i - logdet_j)
            - 0.5
                * ((mean_i.transpose() * inv_i * &mean_i)[(0, 0)]
                    - (mean_j.transpose() * inv_j * &mean_j)[(0, 0)])
    };
    let quadratic = if model.discriminant.uses_shared_covariance() {
        let mut values = Vec::new();
        values.try_reserve_exact(quadratic_len).map_err(|err| {
            internal_error(format!(
                "classify: cannot allocate quadratic coefficients: {err}"
            ))
        })?;
        values.resize(quadratic_len, 0.0);
        values
    } else {
        ((inv_j - inv_i) * 0.5).iter().copied().collect()
    };
    Ok((const_term, linear_vec.iter().copied().collect(), quadratic))
}

pub(super) fn log_prior_ratio(left: f64, right: f64) -> f64 {
    match (left > 0.0, right > 0.0) {
        (true, true) => (left / right).ln(),
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => 0.0,
    }
}
