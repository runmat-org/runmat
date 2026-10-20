use super::*;
use runmat_value::StructArray;

pub(super) struct PairwiseCoefficients {
    constant: f64,
    linear: Vec<f64>,
    pub(super) quadratic: Option<Vec<f64>>,
}

pub(super) fn coeff_value(model: &PreparedModel) -> BuiltinResult<Value> {
    let k = model.classes.len();
    let entry_count = k
        .checked_mul(k)
        .ok_or_else(|| internal_error("classify: coefficient structure shape overflows"))?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|err| internal_error(format!("classify: cannot allocate coefficients: {err}")))?;
    // Structure arrays store elements in column-major order: row + col * k.
    for col in 0..k {
        for row in 0..k {
            let coeff = pairwise_coeff(model, row, col)?;
            let mut st = runmat_value::UnboundedStructValueBuilder::new();
            st.insert(
                "type",
                Value::string(model.discriminant.property_name().to_string()),
            );
            st.insert("name1", labels_value(&model.classes, model.kind, &[row])?);
            st.insert("name2", labels_value(&model.classes, model.kind, &[col])?);
            st.insert("const", Value::Num(coeff.constant));
            st.insert(
                "linear",
                Value::Tensor(
                    Tensor::new(coeff.linear, vec![model.cols, 1])
                        .map_err(|err| internal_error(format!("classify: {err}")))?,
                ),
            );
            if let Some(quadratic) = coeff.quadratic {
                st.insert(
                    "quadratic",
                    Value::Tensor(
                        Tensor::new(quadratic, vec![model.cols, model.cols])
                            .map_err(|err| internal_error(format!("classify: {err}")))?,
                    ),
                );
            }
            let st = st.finish();
            entries.push(st);
        }
    }
    Ok(Value::StructArray(
        StructArray::new(entries, vec![k, k])
            .map_err(|err| internal_error(format!("classify: {err}")))?,
    ))
}

pub(super) fn pairwise_coeff(
    model: &PreparedModel,
    i: usize,
    j: usize,
) -> BuiltinResult<PairwiseCoefficients> {
    if i == j {
        return Ok(PairwiseCoefficients {
            constant: 0.0,
            linear: vec![0.0; model.cols],
            quadratic: if model.discriminant.has_quadratic_coefficient() {
                Some(zero_quadratic(model.cols)?)
            } else {
                None
            },
        });
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
    let quadratic = if !model.discriminant.has_quadratic_coefficient() {
        None
    } else if model.discriminant.uses_shared_covariance() {
        Some(zero_quadratic(model.cols)?)
    } else {
        Some(((inv_j - inv_i) * 0.5).iter().copied().collect())
    };
    Ok(PairwiseCoefficients {
        constant: const_term,
        linear: linear_vec.iter().copied().collect(),
        quadratic,
    })
}

fn zero_quadratic(columns: usize) -> BuiltinResult<Vec<f64>> {
    let count = columns
        .checked_mul(columns)
        .ok_or_else(|| internal_error("classify: quadratic coefficient shape overflows"))?;
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|err| {
        internal_error(format!(
            "classify: cannot allocate quadratic coefficients: {err}"
        ))
    })?;
    values.resize(count, 0.0);
    Ok(values)
}

pub(super) fn log_prior_ratio(left: f64, right: f64) -> f64 {
    match (left > 0.0, right > 0.0) {
        (true, true) => (left / right).ln(),
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => 0.0,
    }
}
