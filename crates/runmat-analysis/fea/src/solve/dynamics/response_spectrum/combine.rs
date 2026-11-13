use runmat_analysis_core::{DirectionalCombination, ModalCombination};

use crate::contracts::FeaRunError;

pub(super) fn combine_modal_responses(
    modal_vectors: &[Vec<f64>],
    frequencies_hz: &[f64],
    damping_ratio: f64,
    policy: ModalCombination,
) -> Result<Vec<f64>, FeaRunError> {
    let width = validate_vectors(modal_vectors)?;
    if modal_vectors.len() != frequencies_hz.len()
        || frequencies_hz
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(FeaRunError::Solve(
            "modal response inventory is incomplete or invalid".into(),
        ));
    }
    let mut combined = vec![0.0; width];
    match policy {
        ModalCombination::AbsoluteSum => {
            for vector in modal_vectors {
                for (combined, response) in combined.iter_mut().zip(vector) {
                    *combined += response.abs();
                }
            }
        }
        ModalCombination::Srss => {
            for vector in modal_vectors {
                for (combined, response) in combined.iter_mut().zip(vector) {
                    *combined += response * response;
                }
            }
            combined.iter_mut().for_each(|value| *value = value.sqrt());
        }
        ModalCombination::TenPercent | ModalCombination::CompleteQuadraticCombination => {
            for (left_index, left) in modal_vectors.iter().enumerate() {
                for (right_index, right) in modal_vectors.iter().enumerate() {
                    let coefficient = match policy {
                        ModalCombination::TenPercent => {
                            let lower = frequencies_hz[left_index].min(frequencies_hz[right_index]);
                            let upper = frequencies_hz[left_index].max(frequencies_hz[right_index]);
                            if upper <= 1.1 * lower {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ModalCombination::CompleteQuadraticCombination => cqc_coefficient(
                            frequencies_hz[left_index],
                            frequencies_hz[right_index],
                            damping_ratio,
                        ),
                        ModalCombination::Srss | ModalCombination::AbsoluteSum => unreachable!(),
                    };
                    for ((combined, left), right) in combined.iter_mut().zip(left).zip(right) {
                        *combined += coefficient * left * right;
                    }
                }
            }
            for value in &mut combined {
                let tolerance = 256.0 * f64::EPSILON * value.abs().max(1.0);
                if *value < -tolerance {
                    return Err(FeaRunError::Solve(
                        "modal correlation produced a negative response variance".into(),
                    ));
                }
                *value = value.max(0.0).sqrt();
            }
        }
    }
    Ok(combined)
}

pub(super) fn combine_periodic_and_rigid_responses(
    modal_vectors: &[Vec<f64>],
    frequencies_hz: &[f64],
    rigid_coefficients: &[f64],
    damping_ratio: f64,
    policy: ModalCombination,
) -> Result<(Vec<f64>, Vec<f64>), FeaRunError> {
    if modal_vectors.len() != rigid_coefficients.len()
        || rigid_coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite() || !(0.0..=1.0).contains(coefficient))
    {
        return Err(FeaRunError::Solve(
            "rigid-response coefficients do not match the modal inventory".into(),
        ));
    }
    let periodic_vectors = modal_vectors
        .iter()
        .zip(rigid_coefficients)
        .map(|(vector, coefficient)| {
            let scale = (1.0 - coefficient * coefficient).max(0.0).sqrt();
            vector.iter().map(|value| scale * value).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let periodic =
        combine_modal_responses(&periodic_vectors, frequencies_hz, damping_ratio, policy)?;
    let width = validate_vectors(modal_vectors)?;
    let mut rigid = vec![0.0; width];
    for (vector, coefficient) in modal_vectors.iter().zip(rigid_coefficients) {
        for (rigid, value) in rigid.iter_mut().zip(vector) {
            *rigid += coefficient * value;
        }
    }
    Ok((periodic, rigid))
}

pub(super) fn combine_directional_responses(
    responses: &[&[f64]],
    policy: DirectionalCombination,
) -> Result<Vec<f64>, FeaRunError> {
    let width = validate_slices(responses)?;
    let mut combined = vec![0.0; width];
    match policy {
        DirectionalCombination::Srss => {
            for response in responses {
                for (combined, response) in combined.iter_mut().zip(*response) {
                    *combined += response * response;
                }
            }
            combined.iter_mut().for_each(|value| *value = value.sqrt());
        }
        DirectionalCombination::AbsoluteSum => {
            for response in responses {
                for (combined, response) in combined.iter_mut().zip(*response) {
                    *combined += response.abs();
                }
            }
        }
        DirectionalCombination::PercentageRule { primary, secondary } => {
            for component in 0..width {
                combined[component] = responses
                    .iter()
                    .enumerate()
                    .map(|(primary_index, _)| {
                        responses
                            .iter()
                            .enumerate()
                            .map(|(index, response)| {
                                (if index == primary_index {
                                    primary
                                } else {
                                    secondary
                                }) * response[component].abs()
                            })
                            .sum::<f64>()
                    })
                    .fold(0.0_f64, f64::max);
            }
        }
    }
    Ok(combined)
}

fn cqc_coefficient(left_hz: f64, right_hz: f64, damping: f64) -> f64 {
    if left_hz == right_hz {
        return 1.0;
    }
    let ratio = right_hz / left_hz;
    let numerator = 8.0 * damping * damping * (1.0 + ratio) * ratio.powf(1.5);
    let denominator =
        (1.0 - ratio * ratio).powi(2) + 4.0 * damping * damping * ratio * (1.0 + ratio).powi(2);
    numerator / denominator
}

fn validate_vectors(vectors: &[Vec<f64>]) -> Result<usize, FeaRunError> {
    validate_slices(&vectors.iter().map(Vec::as_slice).collect::<Vec<_>>())
}

fn validate_slices(vectors: &[&[f64]]) -> Result<usize, FeaRunError> {
    let width = vectors.first().map(|vector| vector.len()).unwrap_or(0);
    if width == 0
        || vectors
            .iter()
            .any(|vector| vector.len() != width || vector.iter().any(|value| !value.is_finite()))
    {
        return Err(FeaRunError::Solve(
            "response vectors must be finite, nonempty, and shape-compatible".into(),
        ));
    }
    Ok(width)
}
