use super::*;

pub(in crate::builtins::stats::ml::perfcurve) fn prepare_observations(
    labels: &LabelVector,
    scores: &[f64],
    posclass: &Label,
    neg_classes: &[Label],
    options: &Options,
) -> BuiltinResult<Vec<Observation>> {
    let weights = if let Some(weights) = &options.weights {
        if weights.len() != labels.labels.len() {
            return Err(invalid(
                "perfcurve: Weights must have one value per observation",
            ));
        }
        weights.clone()
    } else {
        vec![1.0; labels.labels.len()]
    };
    if weights
        .iter()
        .any(|value| value.is_nan() || *value < 0.0 || value.is_infinite())
    {
        return Err(invalid(
            "perfcurve: Weights must be finite nonnegative values",
        ));
    }
    let mut observations = Vec::with_capacity(labels.labels.len());
    for ((label, score), weight) in labels.labels.iter().zip(scores).zip(weights.iter()) {
        if is_missing_label(label) || *weight == 0.0 {
            continue;
        }
        if score.is_nan() && options.process_nan == ProcessNaN::Ignore {
            continue;
        }
        if score.is_infinite() {
            return Err(invalid("perfcurve: scores must be finite or NaN"));
        }
        let is_positive = same_label(label, posclass);
        let neg_index = if is_positive {
            None
        } else {
            neg_classes
                .iter()
                .position(|candidate| same_label(candidate, label))
        };
        if !is_positive && neg_index.is_none() {
            continue;
        }
        observations.push(Observation {
            score: *score,
            weight: *weight,
            is_positive,
            neg_index,
        });
    }
    apply_prior_weights(observations, &options.prior)
}

fn apply_prior_weights(
    mut observations: Vec<Observation>,
    prior: &PriorOption,
) -> BuiltinResult<Vec<Observation>> {
    let (pos_prior, neg_prior) = match prior {
        PriorOption::Empirical => {
            let mut total = 0.0_f64;
            for observation in &observations {
                total += observation.weight;
                if !total.is_finite() {
                    return Err(invalid(
                        "perfcurve: total observation weight exceeds finite double range",
                    ));
                }
            }
            return Ok(observations);
        }
        PriorOption::Uniform => (0.5, 0.5),
        PriorOption::Values(pos, neg) => (*pos, *neg),
    };
    let pos_max = observations
        .iter()
        .filter(|obs| obs.is_positive)
        .map(|obs| obs.weight)
        .fold(0.0_f64, f64::max);
    let neg_max = observations
        .iter()
        .filter(|obs| !obs.is_positive)
        .map(|obs| obs.weight)
        .fold(0.0_f64, f64::max);
    if pos_max == 0.0 || neg_max == 0.0 {
        return Ok(observations);
    }
    if pos_prior <= 0.0 || neg_prior <= 0.0 {
        return Err(invalid(
            "perfcurve: positive and negative prior probabilities must be nonzero",
        ));
    }
    let pos_scaled_total: f64 = observations
        .iter()
        .filter(|obs| obs.is_positive)
        .map(|obs| obs.weight / pos_max)
        .sum();
    let neg_scaled_total: f64 = observations
        .iter()
        .filter(|obs| !obs.is_positive)
        .map(|obs| obs.weight / neg_max)
        .sum();
    for obs in &mut observations {
        obs.weight = if obs.is_positive {
            (obs.weight / pos_max) * (pos_prior / pos_scaled_total)
        } else {
            (obs.weight / neg_max) * (neg_prior / neg_scaled_total)
        };
    }
    Ok(observations)
}
