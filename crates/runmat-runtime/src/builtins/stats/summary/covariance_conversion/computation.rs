use super::*;

pub(super) fn covariance_to_correlation(
    name: &'static str,
    covariance: Tensor,
) -> BuiltinResult<(Value, Value)> {
    validate_covariance_matrix(name, &covariance)?;
    let (n, _) = super::matrix_validation::covariance_matrix_extents(covariance.shape());
    let covariance_values = tensor::tensor_values_f64_cow(&covariance);
    let mut sigma = Vec::with_capacity(n);
    for idx in 0..n {
        sigma.push(covariance_values[idx + idx * n].sqrt());
    }

    let mut r = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let denom = sigma[row] * sigma[col];
            let idx = row + col * n;
            r[idx] = if denom == 0.0 {
                f64::NAN
            } else {
                covariance_values[idx] / denom
            };
        }
    }

    let dtype = covariance.numeric_dtype();
    let r = Tensor::new_with_dtype(r, vec![n, n], dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(name, format!("{name}: {err}")))?;
    let sigma = Tensor::new_with_dtype(sigma, vec![n, 1], dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(name, format!("{name}: {err}")))?;
    Ok((r, sigma))
}
