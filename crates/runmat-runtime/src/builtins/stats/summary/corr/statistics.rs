use super::*;
#[derive(Clone, Copy)]
pub(super) struct CorrPair {
    pub(super) rho: f64,
    pub(super) n: usize,
}

pub(super) fn corr_pair_masked(
    x: &CorrColumn,
    y: &CorrColumn,
    mask: &[bool],
    corr_type: CorrType,
    rows: RowsMode,
    weights: Option<&[f64]>,
) -> CorrPair {
    let mut indices = Vec::new();
    for idx in 0..mask.len() {
        if !mask[idx] {
            continue;
        }
        match rows {
            RowsMode::All => indices.push(idx),
            RowsMode::Complete | RowsMode::Pairwise => {
                if !x.is_nan(idx) && !y.is_nan(idx) {
                    indices.push(idx);
                }
            }
        }
    }
    if matches!(rows, RowsMode::All)
        && indices
            .iter()
            .any(|index| x.is_nan(*index) || y.is_nan(*index))
    {
        return CorrPair {
            rho: f64::NAN,
            n: indices.len(),
        };
    }
    if indices.len() < 2 {
        return CorrPair {
            rho: f64::NAN,
            n: indices.len(),
        };
    }
    let rho = match corr_type {
        CorrType::Pearson => pearson_columns(x, y, &indices, weights),
        CorrType::Spearman => {
            let xr = x.ranks(&indices, weights);
            let yr = y.ranks(&indices, weights);
            let selected_weights = weights.map(|weights| {
                indices
                    .iter()
                    .map(|index| weights[*index])
                    .collect::<Vec<_>>()
            });
            pearson(&xr, &yr, selected_weights.as_deref())
        }
        CorrType::Kendall => kendall_tau_b(x, y, &indices, weights),
    };
    CorrPair {
        rho,
        n: indices.len(),
    }
}

pub(super) fn pearson_columns(
    x: &CorrColumn,
    y: &CorrColumn,
    indices: &[usize],
    weights: Option<&[f64]>,
) -> f64 {
    let xs = x.centered_values(indices);
    let ys = y.centered_values(indices);
    let selected_weights = weights.map(|weights| {
        indices
            .iter()
            .map(|index| weights[*index])
            .collect::<Vec<_>>()
    });
    pearson(&xs, &ys, selected_weights.as_deref())
}

pub(super) fn pearson(x: &[f64], y: &[f64], weights: Option<&[f64]>) -> f64 {
    let weight_sum = weights
        .map(|weights| weights.iter().sum::<f64>())
        .unwrap_or(x.len() as f64);
    if weight_sum <= 0.0 {
        return f64::NAN;
    }
    let mean_x = x
        .iter()
        .enumerate()
        .map(|(index, value)| weights.map_or(*value, |weights| weights[index] * *value))
        .sum::<f64>()
        / weight_sum;
    let mean_y = y
        .iter()
        .enumerate()
        .map(|(index, value)| weights.map_or(*value, |weights| weights[index] * *value))
        .sum::<f64>()
        / weight_sum;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (index, (a, b)) in x.iter().zip(y.iter()).enumerate() {
        let weight = weights.map_or(1.0, |weights| weights[index]);
        let dx = *a - mean_x;
        let dy = *b - mean_y;
        sxx += weight * dx * dx;
        syy += weight * dy * dy;
        sxy += weight * dx * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        f64::NAN
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

pub(super) fn kendall_tau_b(
    x: &CorrColumn,
    y: &CorrColumn,
    indices: &[usize],
    weights: Option<&[f64]>,
) -> f64 {
    if indices.len() < 2 {
        return f64::NAN;
    }
    let mut concordant = 0.0;
    let mut discordant = 0.0;
    let mut ties_x = 0.0;
    let mut ties_y = 0.0;
    for i in 0..indices.len() {
        for j in i + 1..indices.len() {
            let left = indices[i];
            let right = indices[j];
            let pair_weight = weights.map_or(1.0, |weights| weights[left] * weights[right]);
            let dx = x.compare(left, right);
            let dy = y.compare(left, right);
            if dx == Ordering::Equal && dy == Ordering::Equal {
                continue;
            }
            if dx == Ordering::Equal {
                ties_x += pair_weight;
            } else if dy == Ordering::Equal {
                ties_y += pair_weight;
            } else if dx == dy {
                concordant += pair_weight;
            } else {
                discordant += pair_weight;
            }
        }
    }
    let denominator =
        f64::sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y));
    if denominator == 0.0 {
        f64::NAN
    } else {
        (concordant - discordant) / denominator
    }
}
