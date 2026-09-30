use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{Geometry, NormalizationError, NormalizationOptions};

pub(in crate::builtins::deep_learning) enum NormalizationStatsData {
    F32 { mean: Vec<f32>, inv_std: Vec<f32> },
    F64 { mean: Vec<f64>, inv_std: Vec<f64> },
}

pub(in crate::builtins::deep_learning) struct NormalizationStats {
    pub(super) shape: Vec<usize>,
    pub(super) labels: String,
    pub(super) reduce: Vec<bool>,
    pub(super) epsilon: f64,
    pub(super) data: NormalizationStatsData,
}

pub(in crate::builtins::deep_learning) struct NormalizationOutput {
    pub tensor: Tensor,
    pub axes: AxisPlan,
    pub stats: NormalizationStats,
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
        let (values, mean, inv_std) = forward_values(x, gamma, beta, &geometry);
        (
            Tensor::from_numeric_storage(NumericStorage::F32(values), input.shape.clone()),
            NormalizationStatsData::F32 { mean, inv_std },
        )
    } else if let (Some(x), Some(gamma), Some(beta)) = (
        input.as_f64_slice(),
        scale.as_f64_slice(),
        offset.as_f64_slice(),
    ) {
        let (values, mean, inv_std) = forward_values(x, gamma, beta, &geometry);
        (
            Tensor::from_numeric_storage(NumericStorage::F64(values), input.shape.clone()),
            NormalizationStatsData::F64 { mean, inv_std },
        )
    } else {
        return Err(NormalizationError::UnsupportedPrecision);
    };
    Ok(NormalizationOutput {
        tensor: tensor.map_err(NormalizationError::Storage)?,
        axes: axes.clone(),
        stats: NormalizationStats {
            shape: input.shape.clone(),
            labels: axes.layout().labels(),
            reduce: geometry.reduce,
            epsilon: options.epsilon,
            data,
        },
    })
}

fn forward_values<T: Real>(
    x: &[T],
    gamma: &[T],
    beta: &[T],
    g: &Geometry,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut mean = vec![T::zero(); g.groups];
    let mut variance = vec![T::zero(); g.groups];
    let mut inv_std = vec![T::zero(); g.groups];
    let count = T::from_f64(g.group_size as f64);
    for (index, &value) in x.iter().enumerate() {
        let (group, _) = g.group_and_channel(index);
        mean[group] += value;
    }
    for value in &mut mean {
        *value = *value / count;
    }
    for (index, &value) in x.iter().enumerate() {
        let (group, _) = g.group_and_channel(index);
        let delta = value - mean[group];
        variance[group] += delta * delta;
    }
    for (index, value) in inv_std.iter_mut().enumerate() {
        *value = T::one() / (variance[index] / count + T::from_f64(g.epsilon)).sqrt();
    }
    let mut output = vec![T::zero(); x.len()];
    for (index, &value) in x.iter().enumerate() {
        let (group, channel) = g.group_and_channel(index);
        output[index] = (value - mean[group]) * inv_std[group] * gamma[channel] + beta[channel];
    }
    (output, mean, inv_std)
}
