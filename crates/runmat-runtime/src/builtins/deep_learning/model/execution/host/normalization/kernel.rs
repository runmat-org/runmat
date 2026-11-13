use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use crate::builtins::deep_learning::model::plan::normalization::{
    Geometry, NormalizationError, NormalizationOptions,
};

pub(in crate::builtins::deep_learning) enum NormalizationStatsData {
    F32 {
        scale: Vec<f32>,
        mean_scaled: Vec<f32>,
        inv_std_scaled: Vec<f32>,
        inv_std_input: Vec<f32>,
    },
    F64 {
        scale: Vec<f64>,
        mean_scaled: Vec<f64>,
        inv_std_scaled: Vec<f64>,
        inv_std_input: Vec<f64>,
    },
}

pub(in crate::builtins::deep_learning) struct NormalizationStats {
    pub(super) axes: AxisPlan,
    pub(super) reduce: Vec<bool>,
    pub(super) epsilon: f64,
    pub(super) data: NormalizationStatsData,
}

pub(in crate::builtins::deep_learning) struct NormalizationOutput {
    pub tensor: Tensor,
    pub axes: AxisPlan,
    pub stats: NormalizationStats,
}

struct ForwardValues<T> {
    output: Vec<T>,
    scale: Vec<T>,
    mean_scaled: Vec<T>,
    inv_std_scaled: Vec<T>,
    inv_std_input: Vec<T>,
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    scale: &Tensor,
    offset: &Tensor,
    axes: &AxisPlan,
    options: NormalizationOptions,
) -> Result<NormalizationOutput, NormalizationError> {
    let geometry = Geometry::new(axes, options)?;
    geometry.validate_payload(input, scale, offset)?;
    let (tensor, data) = if let (Some(x), Some(gamma), Some(beta)) = (
        input.as_f32_slice(),
        scale.as_f32_slice(),
        offset.as_f32_slice(),
    ) {
        let ForwardValues {
            output,
            scale,
            mean_scaled,
            inv_std_scaled,
            inv_std_input,
        } = forward_values(x, gamma, beta, &geometry);
        (
            Tensor::from_numeric_storage(NumericStorage::F32(output), input.shape().to_vec()),
            NormalizationStatsData::F32 {
                scale,
                mean_scaled,
                inv_std_scaled,
                inv_std_input,
            },
        )
    } else if let (Some(x), Some(gamma), Some(beta)) = (
        input.as_f64_slice(),
        scale.as_f64_slice(),
        offset.as_f64_slice(),
    ) {
        let ForwardValues {
            output,
            scale,
            mean_scaled,
            inv_std_scaled,
            inv_std_input,
        } = forward_values(x, gamma, beta, &geometry);
        (
            Tensor::from_numeric_storage(NumericStorage::F64(output), input.shape().to_vec()),
            NormalizationStatsData::F64 {
                scale,
                mean_scaled,
                inv_std_scaled,
                inv_std_input,
            },
        )
    } else {
        return Err(NormalizationError::UnsupportedPrecision);
    };
    Ok(NormalizationOutput {
        tensor: tensor.map_err(NormalizationError::Storage)?,
        axes: axes.clone(),
        stats: NormalizationStats {
            axes: axes.clone(),
            reduce: geometry.reduce,
            epsilon: options.epsilon,
            data,
        },
    })
}

fn forward_values<T: Real>(x: &[T], gamma: &[T], beta: &[T], g: &Geometry) -> ForwardValues<T> {
    // Scale before accumulating. For finite near-limit inputs, the mean and
    // normalized output can be finite even when their unscaled sum or variance
    // would overflow in the input precision.
    let sqrt_epsilon = T::from_f64(g.epsilon).sqrt();
    let mut scale = vec![sqrt_epsilon; g.groups];
    for (index, &value) in x.iter().enumerate() {
        let (group, _) = g.group_and_channel(index);
        if value.abs() > scale[group] {
            scale[group] = value.abs();
        }
    }
    let mut mean_scaled = vec![T::zero(); g.groups];
    let mut variance = vec![T::zero(); g.groups];
    let mut inv_std_scaled = vec![T::zero(); g.groups];
    let mut inv_std_input = vec![T::zero(); g.groups];
    let count = T::from_f64(g.group_size as f64);
    for (index, &value) in x.iter().enumerate() {
        let (group, _) = g.group_and_channel(index);
        mean_scaled[group] += value / scale[group];
    }
    for value in &mut mean_scaled {
        *value = *value / count;
    }
    for (index, &value) in x.iter().enumerate() {
        let (group, _) = g.group_and_channel(index);
        let delta = value / scale[group] - mean_scaled[group];
        variance[group] += delta * delta;
    }
    for index in 0..g.groups {
        let epsilon_ratio = sqrt_epsilon / scale[index];
        let denominator = (variance[index] / count + epsilon_ratio * epsilon_ratio).sqrt();
        if denominator == T::zero() {
            // Epsilon can disappear only after scaling by an extreme finite
            // input. Zero scaled variance means every represented xhat is zero.
            inv_std_input[index] = T::one() / sqrt_epsilon;
        } else {
            inv_std_scaled[index] = T::one() / denominator;
            inv_std_input[index] = inv_std_scaled[index] / scale[index];
        }
    }
    let mut output = vec![T::zero(); x.len()];
    for (index, &value) in x.iter().enumerate() {
        let (group, channel) = g.group_and_channel(index);
        let normalized = (value / scale[group] - mean_scaled[group]) * inv_std_scaled[group];
        output[index] = normalized * gamma[channel] + beta[channel];
    }
    ForwardValues {
        output,
        scale,
        mean_scaled,
        inv_std_scaled,
        inv_std_input,
    }
}
