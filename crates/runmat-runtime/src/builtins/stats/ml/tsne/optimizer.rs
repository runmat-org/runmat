use super::geometry::{
    apply_pca_if_requested, guard_dense_feature_work, matrix_value, needs_pca, prepare_data,
    standardize_rows,
};
use super::metric::pairwise_squared_distances;
use super::{
    internal, invalid, Algorithm, CacheSize, DistanceMetric, EmbeddingResult, Options,
    PreparedData, MAX_WORK_CELLS, MIN_PROB, NAME,
};
use crate::builtins::common::{random, tensor};
use crate::BuiltinResult;
use runmat_value::Tensor;

pub(super) fn compute_tsne(x: Tensor, options: Options) -> BuiltinResult<EmbeddingResult> {
    if x.shape.len() > 2 {
        return Err(invalid("tsne: X must be a 2-D numeric matrix"));
    }
    let prepared = prepare_data(&x)?;
    if prepared.rows.is_empty() {
        return Err(invalid(
            "tsne: at least one complete observation is required",
        ));
    }
    if prepared.rows.len() < 2 {
        return single_row_embedding(prepared.rows.len(), options.num_dimensions);
    }
    if prepared.rows.len().saturating_mul(prepared.rows.len()) > MAX_WORK_CELLS {
        return Err(invalid(format!(
            "tsne: exact CPU implementation supports at most {MAX_WORK_CELLS} pairwise cells"
        )));
    }
    let feature_count = prepared.rows[0].len();
    if matches!(options.distance, DistanceMetric::Mahalanobis) || needs_pca(&options, feature_count)
    {
        guard_dense_feature_work(feature_count)?;
    }
    let mut rows = prepared.rows.clone();
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
        standardize_rows(&mut rows);
    }
    rows = apply_pca_if_requested(rows, options.num_pca_components)?;
    let distances = pairwise_squared_distances(&rows, options.distance)?;
    let probabilities = joint_probabilities(&distances, options.perplexity)?;
    let mut embedding = initial_embedding(&options, &prepared)?;
    center_embedding(&mut embedding, options.num_dimensions);
    let mut velocity = vec![0.0; embedding.len()];
    let mut gains = vec![1.0_f64; embedding.len()];
    let iterations = options.max_iter.max(1);
    for iter in 0..iterations {
        let exaggeration = if iter < 99 { options.exaggeration } else { 1.0 };
        let (gradient, _, grad_norm) = gradient_and_loss(
            &embedding,
            options.num_dimensions,
            &probabilities,
            exaggeration,
        );
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
        }
        center_embedding(&mut embedding, options.num_dimensions);
        if grad_norm < options.tol_fun {
            break;
        }
    }
    let (_, loss, _) = gradient_and_loss(&embedding, options.num_dimensions, &probabilities, 1.0);
    let output = Tensor::new(embedding, vec![prepared.rows.len(), options.num_dimensions])
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

fn joint_probabilities(distances: &[f64], perplexity: f64) -> BuiltinResult<Vec<f64>> {
    let n = integer_sqrt(distances.len())
        .ok_or_else(|| internal("tsne: distance matrix has invalid shape"))?;
    let effective_perplexity = perplexity.min((n - 1) as f64).max(1.0);
    let target_entropy = effective_perplexity.ln();
    let mut conditional = vec![0.0; n * n];
    for i in 0..n {
        let mut beta = 1.0;
        let mut beta_min = f64::NEG_INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut row_prob = vec![0.0; n];
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
            }
            if sum <= 0.0 || !sum.is_finite() {
                for (j, prob) in row_prob.iter_mut().enumerate() {
                    *prob = if i == j { 0.0 } else { 1.0 / (n - 1) as f64 };
                }
                break;
            }
            let mut entropy = 0.0;
            for prob in &mut row_prob {
                *prob /= sum;
                if *prob > 0.0 {
                    entropy -= *prob * prob.ln();
                }
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
        }
    }
    let mut joint = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                joint[i + j * n] = ((conditional[i + j * n] + conditional[j + i * n])
                    / (2.0 * n as f64))
                    .max(MIN_PROB);
            }
        }
    }
    let sum = joint.iter().sum::<f64>();
    if sum > 0.0 {
        for value in &mut joint {
            *value /= sum;
        }
    }
    Ok(joint)
}

fn initial_embedding(options: &Options, prepared: &PreparedData) -> BuiltinResult<Vec<f64>> {
    let rows = prepared.rows.len();
    let dims = options.num_dimensions;
    if let Some(initial) = &options.initial_y {
        let values = tensor::tensor_values_f64_cow(initial);
        if initial.shape.len() > 2 || initial.cols != dims {
            return Err(invalid(format!(
                "tsne: InitialY must have {} columns",
                dims
            )));
        }
        let mut data = Vec::with_capacity(rows * dims);
        if initial.rows == prepared.original_rows {
            for col in 0..dims {
                for &row in &prepared.good_rows {
                    let value = matrix_value(values.as_ref(), initial.rows, row, col);
                    if !value.is_finite() {
                        return Err(invalid("tsne: InitialY values must be finite"));
                    }
                    data.push(value);
                }
            }
        } else if initial.rows == rows {
            for col in 0..dims {
                for row in 0..rows {
                    let value = matrix_value(values.as_ref(), initial.rows, row, col);
                    if !value.is_finite() {
                        return Err(invalid("tsne: InitialY values must be finite"));
                    }
                    data.push(value);
                }
            }
        } else {
            return Err(invalid(
                "tsne: InitialY row count must match X or complete X rows",
            ));
        }
        return Ok(data);
    }
    let normals = random::generate_normal(rows * dims, NAME)?;
    Ok(normals.into_iter().map(|value| value * 1.0e-4).collect())
}

fn gradient_and_loss(
    embedding: &[f64],
    dims: usize,
    probabilities: &[f64],
    exaggeration: f64,
) -> (Vec<f64>, f64, f64) {
    let n = integer_sqrt(probabilities.len()).unwrap_or(0);
    let mut q_num = vec![0.0; n * n];
    let mut sum_q = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist = 0.0;
            for dim in 0..dims {
                let diff = embedding[i + dim * n] - embedding[j + dim * n];
                dist += diff * diff;
            }
            let value = 1.0 / (1.0 + dist);
            q_num[i + j * n] = value;
            q_num[j + i * n] = value;
            sum_q += 2.0 * value;
        }
    }
    let sum_q = sum_q.max(MIN_PROB);
    let mut gradient = vec![0.0; embedding.len()];
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
            }
        }
    }
    let grad_norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    (gradient, loss, grad_norm)
}

fn center_embedding(embedding: &mut [f64], dims: usize) {
    let rows = embedding.len() / dims.max(1);
    for dim in 0..dims {
        let start = dim * rows;
        let mean = embedding[start..start + rows].iter().sum::<f64>() / rows.max(1) as f64;
        for row in 0..rows {
            embedding[start + row] -= mean;
        }
    }
}

fn integer_sqrt(value: usize) -> Option<usize> {
    let root = (value as f64).sqrt() as usize;
    if root.saturating_mul(root) == value {
        Some(root)
    } else {
        None
    }
}
