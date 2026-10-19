use super::*;

pub(super) fn response_values(
    value: Value,
    distribution: Distribution,
    expected_rows: usize,
    options: &Options,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let tensor = value_to_real_tensor("Y", value)?;
    let values = tensor::tensor_values_f64(&tensor);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("lassoglm: Y must contain finite real values"));
    }
    match distribution {
        Distribution::Normal => {
            if !is_vector_shape(tensor.shape()) {
                return Err(invalid("lassoglm: normal Y must be a vector"));
            }
            Ok((values, Vec::new()))
        }
        Distribution::Poisson => {
            if !is_vector_shape(tensor.shape()) {
                return Err(invalid("lassoglm: poisson Y must be a vector"));
            }
            if values.iter().any(|value| *value < 0.0) {
                return Err(invalid("lassoglm: poisson Y must be nonnegative"));
            }
            Ok((values, Vec::new()))
        }
        Distribution::Binomial => {
            if tensor.cols() == 2 && tensor.rows() == expected_rows {
                if options.binomial_size.is_some() {
                    return Err(invalid(
                        "lassoglm: BinomialSize cannot be combined with two-column binomial counts",
                    ));
                }
                let mut successes = Vec::with_capacity(tensor.rows());
                let mut trials = Vec::with_capacity(tensor.rows());
                for row in 0..tensor.rows() {
                    let y = values[row];
                    let n = values[row + tensor.rows()];
                    if y < 0.0 || n <= 0.0 || y > n {
                        return Err(invalid(
                            "lassoglm: binomial count response must satisfy 0 <= successes <= trials",
                        ));
                    }
                    successes.push(y / n);
                    trials.push(n);
                }
                Ok((successes, trials))
            } else {
                if !is_vector_shape(tensor.shape()) {
                    return Err(invalid(
                        "lassoglm: binomial Y must be a vector or two-column count matrix",
                    ));
                }
                if values.iter().any(|value| *value < 0.0 || *value > 1.0) {
                    return Err(invalid(
                        "lassoglm: binomial vector Y must contain probabilities in [0,1]",
                    ));
                }
                let trials = match &options.binomial_size {
                    Some(values) => expand_binomial_size(values, expected_rows)?,
                    None => Vec::new(),
                };
                Ok((values, trials))
            }
        }
    }
}

pub(super) fn expand_binomial_size(values: &[f64], rows: usize) -> BuiltinResult<Vec<f64>> {
    if values.len() == 1 {
        return Ok(vec![values[0]; rows]);
    }
    if values.len() != rows {
        return Err(invalid(
            "lassoglm: BinomialSize must be scalar or contain one value per observation",
        ));
    }
    Ok(values.to_vec())
}
