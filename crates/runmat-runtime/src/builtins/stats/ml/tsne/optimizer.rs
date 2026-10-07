use super::geometry::{
    apply_pca_if_requested, guard_dense_feature_work, needs_pca, prepare_data_cooperative,
    standardize_rows, tsne_matrix_dimensions,
};
use super::metric::pairwise_squared_distances_cooperative;
use super::{
    internal, invalid, Algorithm, CacheSize, DistanceMetric, EmbeddingResult, Options,
    PreparedData, MAX_WORK_CELLS, MIN_PROB,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_value::Tensor;

pub(super) async fn compute_tsne_cooperative(
    x: Tensor,
    options: Options,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<EmbeddingResult> {
    if x.shape().len() > 2 {
        return Err(invalid("tsne: X must be a 2-D numeric matrix"));
    }
    let PreparedData {
        mut rows,
        good_rows,
        original_rows,
    } = prepare_data_cooperative(&x, cooperation).await?;
    if rows.is_empty() {
        return Err(invalid(
            "tsne: at least one complete observation is required",
        ));
    }
    if rows.len() < 2 {
        return single_row_embedding(rows.len(), options.num_dimensions);
    }
    if rows.len().saturating_mul(rows.len()) > MAX_WORK_CELLS {
        return Err(invalid(format!(
            "tsne: exact CPU implementation supports at most {MAX_WORK_CELLS} pairwise cells"
        )));
    }
    let complete_rows = rows.len();
    let feature_count = rows[0].len();
    if matches!(options.distance, DistanceMetric::Mahalanobis) || needs_pca(&options, feature_count)
    {
        guard_dense_feature_work(feature_count)?;
    }
    let _reporting_options = (options.num_print, options.verbose);
    let _barnes_hut_theta = options.theta;
    match options.algorithm {
        Algorithm::Exact | Algorithm::BarnesHut => {}
    }
    match &options.cache_size {
        Some(CacheSize::Megabytes(value)) if !value.is_finite() || *value <= 0.0 => {
            return Err(invalid("tsne: CacheSize must be positive"));
        }
        Some(CacheSize::Megabytes(_)) | Some(CacheSize::Maximal) | None => {}
    }
    if options.standardize {
        standardize_rows(&mut rows, cooperation).await?;
    }
    rows = apply_pca_if_requested(rows, options.num_pca_components, cooperation).await?;
    let distances =
        pairwise_squared_distances_cooperative(&rows, options.distance, cooperation).await?;
    let probabilities = joint_probabilities(&distances, options.perplexity, cooperation).await?;
    let mut embedding = initial_embedding(
        &options,
        complete_rows,
        original_rows,
        &good_rows,
        cooperation,
    )
    .await?;
    center_embedding(&mut embedding, options.num_dimensions, cooperation).await?;
    let mut velocity = filled_f64(embedding.len(), 0.0, cooperation).await?;
    let mut gains = filled_f64(embedding.len(), 1.0, cooperation).await?;
    let iterations = options.max_iter.max(1);
    for iter in 0..iterations {
        let exaggeration = if iter < 99 { options.exaggeration } else { 1.0 };
        let (gradient, _, grad_norm) = gradient_and_loss(
            &embedding,
            options.num_dimensions,
            &probabilities,
            exaggeration,
            cooperation,
        )
        .await?;
        let momentum = if iter < 250 { 0.5 } else { 0.8 };
        for idx in 0..embedding.len() {
            gains[idx] = if (gradient[idx] > 0.0) != (velocity[idx] > 0.0) {
                gains[idx] + 0.2
            } else {
                (gains[idx] * 0.8).max(0.01)
            };
            velocity[idx] =
                momentum * velocity[idx] - options.learn_rate * gains[idx] * gradient[idx];
            embedding[idx] += velocity[idx];
            cooperation.completed(1).await?;
        }
        center_embedding(&mut embedding, options.num_dimensions, cooperation).await?;
        if grad_norm < options.tol_fun {
            break;
        }
    }
    let (_, loss, _) = gradient_and_loss(
        &embedding,
        options.num_dimensions,
        &probabilities,
        1.0,
        cooperation,
    )
    .await?;
    let output = Tensor::new(embedding, vec![complete_rows, options.num_dimensions])
        .map_err(|err| internal(format!("tsne: {err}")))?;
    Ok(EmbeddingResult {
        embedding: output,
        loss,
    })
}

fn single_row_embedding(rows: usize, dims: usize) -> BuiltinResult<EmbeddingResult> {
    let tensor = Tensor::new(vec![0.0; rows * dims], vec![rows, dims])
        .map_err(|err| internal(format!("tsne: {err}")))?;
    Ok(EmbeddingResult {
        embedding: tensor,
        loss: 0.0,
    })
}

async fn joint_probabilities(
    distances: &[f64],
    perplexity: f64,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<f64>> {
    let n = integer_sqrt(distances.len())
        .ok_or_else(|| internal("tsne: distance matrix has invalid shape"))?;
    let effective_perplexity = perplexity.min((n - 1) as f64).max(1.0);
    let target_entropy = effective_perplexity.ln();
    let mut conditional = filled_f64(distances.len(), 0.0, cooperation).await?;
    for i in 0..n {
        let mut beta = 1.0;
        let mut beta_min = f64::NEG_INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut row_prob = filled_f64(n, 0.0, cooperation).await?;
        for _ in 0..80 {
            let mut sum = 0.0;
            for j in 0..n {
                if i == j {
                    row_prob[j] = 0.0;
                } else {
                    let value = (-distances[i + j * n] * beta).exp();
                    row_prob[j] = value;
                    sum += value;
                }
                cooperation.completed(1).await?;
            }
            if sum <= 0.0 || !sum.is_finite() {
                for (j, prob) in row_prob.iter_mut().enumerate() {
                    *prob = if i == j { 0.0 } else { 1.0 / (n - 1) as f64 };
                    cooperation.completed(1).await?;
                }
                break;
            }
            let mut entropy = 0.0;
            for prob in &mut row_prob {
                *prob /= sum;
                if *prob > 0.0 {
                    entropy -= *prob * prob.ln();
                }
                cooperation.completed(1).await?;
            }
            let diff = entropy - target_entropy;
            if diff.abs() < 1.0e-5 {
                break;
            }
            if diff > 0.0 {
                beta_min = beta;
                beta = if beta_max.is_finite() {
                    (beta + beta_max) / 2.0
                } else {
                    beta * 2.0
                };
            } else {
                beta_max = beta;
                beta = if beta_min.is_finite() {
                    (beta + beta_min) / 2.0
                } else {
                    beta / 2.0
                };
            }
        }
        for j in 0..n {
            conditional[i + j * n] = row_prob[j];
            cooperation.completed(1).await?;
        }
    }
    let mut joint = filled_f64(distances.len(), 0.0, cooperation).await?;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                joint[i + j * n] = ((conditional[i + j * n] + conditional[j + i * n])
                    / (2.0 * n as f64))
                    .max(MIN_PROB);
            }
            cooperation.completed(1).await?;
        }
    }
    let mut sum = 0.0;
    for value in &joint {
        sum += value;
        cooperation.completed(1).await?;
    }
    if sum > 0.0 {
        for value in &mut joint {
            *value /= sum;
            cooperation.completed(1).await?;
        }
    }
    Ok(joint)
}

async fn initial_embedding(
    options: &Options,
    rows: usize,
    original_rows: usize,
    good_rows: &[usize],
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<f64>> {
    let dims = options.num_dimensions;
    if let Some(initial) = &options.initial_y {
        let (initial_rows, initial_cols) = tsne_matrix_dimensions(initial);
        if initial.shape().len() > 2 || initial_cols != dims {
            return Err(invalid(format!(
                "tsne: InitialY must have {} columns",
                dims
            )));
        }
        let count = rows
            .checked_mul(dims)
            .ok_or_else(|| invalid("tsne: embedding shape overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(count).map_err(|source| {
            super::internal_with_source("tsne: embedding exceeds available memory", source)
        })?;
        if initial_rows == original_rows {
            for col in 0..dims {
                for &row in good_rows {
                    let value = tensor::tensor_value_f64(initial, row + col * initial_rows);
                    if !value.is_finite() {
                        return Err(invalid("tsne: InitialY values must be finite"));
                    }
                    data.push(value);
                    cooperation.completed(1).await?;
                }
            }
        } else if initial_rows == rows {
            for col in 0..dims {
                for row in 0..rows {
                    let value = tensor::tensor_value_f64(initial, row + col * initial_rows);
                    if !value.is_finite() {
                        return Err(invalid("tsne: InitialY values must be finite"));
                    }
                    data.push(value);
                    cooperation.completed(1).await?;
                }
            }
        } else {
            return Err(invalid(
                "tsne: InitialY row count must match X or complete X rows",
            ));
        }
        return Ok(data);
    }
    let count = rows
        .checked_mul(dims)
        .ok_or_else(|| invalid("tsne: embedding shape overflow"))?;
    let normals = cooperation.normals(count).await?;
    let mut scaled = Vec::new();
    scaled.try_reserve_exact(normals.len()).map_err(|source| {
        super::internal_with_source("tsne: embedding exceeds available memory", source)
    })?;
    for value in normals {
        scaled.push(value * 1.0e-4);
        cooperation.completed(1).await?;
    }
    Ok(scaled)
}

async fn gradient_and_loss(
    embedding: &[f64],
    dims: usize,
    probabilities: &[f64],
    exaggeration: f64,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<(Vec<f64>, f64, f64)> {
    let n = integer_sqrt(probabilities.len()).unwrap_or(0);
    let mut q_num = filled_f64(probabilities.len(), 0.0, cooperation).await?;
    let mut sum_q = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist = 0.0;
            for dim in 0..dims {
                let diff = embedding[i + dim * n] - embedding[j + dim * n];
                dist += diff * diff;
                cooperation.completed(1).await?;
            }
            let value = 1.0 / (1.0 + dist);
            q_num[i + j * n] = value;
            q_num[j + i * n] = value;
            sum_q += 2.0 * value;
            cooperation.completed(1).await?;
        }
    }
    let sum_q = sum_q.max(MIN_PROB);
    let mut gradient = filled_f64(embedding.len(), 0.0, cooperation).await?;
    let mut loss = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let p = probabilities[i + j * n].max(MIN_PROB);
            let q = (q_num[i + j * n] / sum_q).max(MIN_PROB);
            loss += p * (p / q).ln();
            let coeff = 4.0 * (exaggeration * p - q) * q_num[i + j * n];
            for dim in 0..dims {
                gradient[i + dim * n] += coeff * (embedding[i + dim * n] - embedding[j + dim * n]);
                cooperation.completed(1).await?;
            }
            cooperation.completed(1).await?;
        }
    }
    let mut grad_square_sum = 0.0;
    for value in &gradient {
        grad_square_sum += value * value;
        cooperation.completed(1).await?;
    }
    Ok((gradient, loss, grad_square_sum.sqrt()))
}

async fn center_embedding(
    embedding: &mut [f64],
    dims: usize,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<()> {
    let rows = embedding.len() / dims.max(1);
    for dim in 0..dims {
        let start = dim * rows;
        let mut sum = 0.0;
        for value in &embedding[start..start + rows] {
            sum += value;
            cooperation.completed(1).await?;
        }
        let mean = sum / rows.max(1) as f64;
        for row in 0..rows {
            embedding[start + row] -= mean;
            cooperation.completed(1).await?;
        }
    }
    Ok(())
}

fn integer_sqrt(value: usize) -> Option<usize> {
    let root = (value as f64).sqrt() as usize;
    if root.saturating_mul(root) == value {
        Some(root)
    } else {
        None
    }
}

async fn filled_f64(
    len: usize,
    value: f64,
    cooperation: &mut super::cooperation::TsneCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        super::internal_with_source("tsne: numerical workspace exceeds available memory", source)
    })?;
    for _ in 0..len {
        values.push(value);
        cooperation.completed(1).await?;
    }
    Ok(values)
}
