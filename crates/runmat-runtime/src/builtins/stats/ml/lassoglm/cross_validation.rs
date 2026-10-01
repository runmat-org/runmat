use super::*;

pub(super) fn cross_validate(
    x: &Tensor,
    data: &PreparedData,
    distribution: Distribution,
    lambdas: &[f64],
    options: &Options,
    k: usize,
) -> BuiltinResult<CvInfo> {
    if k > data.rows {
        return Err(invalid(
            "lassoglm: CV fold count cannot exceed the number of observations",
        ));
    }
    let mut fold_errors = vec![vec![0.0; lambdas.len()]; k];
    for fold in 0..k {
        let train_rows = (0..x.rows())
            .filter(|row| row % k != fold)
            .collect::<Vec<_>>();
        let test_rows = (0..x.rows())
            .filter(|row| row % k == fold)
            .collect::<Vec<_>>();
        let train_x = subset_rows_tensor(x, &train_rows)?;
        let train_response = (
            train_rows
                .iter()
                .map(|row| data.y[*row])
                .collect::<Vec<_>>(),
            train_rows
                .iter()
                .map(|row| data.trials[*row])
                .collect::<Vec<_>>(),
        );
        let mut fold_options = options.clone();
        fold_options.cv = CvSpec::Resubstitution;
        fold_options.lambda = Some(lambdas.to_vec());
        if let Some(weights) = &options.weights {
            fold_options.weights = Some(train_rows.iter().map(|row| weights[*row]).collect());
        }
        if let Some(offset) = &options.offset {
            fold_options.offset = Some(train_rows.iter().map(|row| offset[*row]).collect());
        }
        if let Some(binomial_size) = &options.binomial_size {
            fold_options.binomial_size =
                Some(train_rows.iter().map(|row| binomial_size[*row]).collect());
        }
        let prepared = prepare_data(&train_x, train_response, &fold_options)?;
        let path = fit_path(&prepared, distribution, lambdas, &fold_options)?;
        for (lambda_idx, beta) in path.coefficients.iter().enumerate() {
            let intercept = path.intercepts[lambda_idx];
            fold_errors[fold][lambda_idx] =
                validation_deviance(x, data, distribution, beta, intercept, &test_rows)?;
        }
    }
    let mut deviance = vec![0.0; lambdas.len()];
    let mut se = vec![0.0; lambdas.len()];
    for lambda_idx in 0..lambdas.len() {
        let values = (0..k)
            .map(|fold| fold_errors[fold][lambda_idx])
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / k as f64;
        let variance = if k > 1 {
            values
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (k - 1) as f64
        } else {
            0.0
        };
        deviance[lambda_idx] = mean;
        se[lambda_idx] = variance.sqrt() / (k as f64).sqrt();
    }
    let index_min_deviance = deviance
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let threshold = deviance[index_min_deviance] + se[index_min_deviance];
    let index_1se = deviance
        .iter()
        .enumerate()
        .filter(|(_, value)| **value <= threshold)
        .max_by(|(left, _), (right, _)| lambdas[*left].total_cmp(&lambdas[*right]))
        .map(|(idx, _)| idx)
        .unwrap_or(index_min_deviance);
    Ok(CvInfo {
        deviance,
        se,
        lambda_min_deviance: lambdas[index_min_deviance],
        lambda_1se: lambdas[index_1se],
        index_min_deviance: index_min_deviance + 1,
        index_1se: index_1se + 1,
    })
}

pub(super) fn subset_rows_tensor(x: &Tensor, rows: &[usize]) -> BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(rows.len() * x.cols());
    for col in 0..x.cols() {
        for row in rows {
            data.push(
                x.get2(*row, col)
                    .map_err(|err| invalid(format!("lassoglm: {err}")))?,
            );
        }
    }
    Tensor::new(data, vec![rows.len(), x.cols()]).map_err(|err| invalid(format!("lassoglm: {err}")))
}

pub(super) fn validation_deviance(
    x: &Tensor,
    data: &PreparedData,
    distribution: Distribution,
    beta: &[f64],
    intercept: f64,
    rows: &[usize],
) -> BuiltinResult<f64> {
    if rows.is_empty() {
        return Ok(f64::NAN);
    }
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for row in rows {
        let eta = data.offset[*row]
            + intercept
            + (0..x.cols())
                .map(|col| x.get2(*row, col).unwrap_or(0.0) * beta[col])
                .sum::<f64>();
        sum += data.weights[*row] * deviance_contribution(distribution, data.y[*row], eta)?;
        weight_sum += data.weights[*row];
    }
    if weight_sum <= EPS {
        return Ok(f64::NAN);
    }
    Ok(sum / weight_sum)
}
