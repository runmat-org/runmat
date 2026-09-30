use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::geometry::{Geometry, NormalizationError, NormalizationOptions};
use super::kernel::{NormalizationStats, NormalizationStatsData};

pub(in crate::builtins::deep_learning) struct NormalizationGradients {
    pub input: Tensor,
    pub scale: Tensor,
    pub offset: Tensor,
}

pub(in crate::builtins::deep_learning) fn backward(
    input: &Tensor,
    scale: &Tensor,
    offset: &Tensor,
    output_gradient: &Tensor,
    axes: &AxisPlan,
    options: NormalizationOptions,
    stats: &NormalizationStats,
) -> Result<NormalizationGradients, NormalizationError> {
    let geometry = Geometry::new(axes, options)?;
    geometry.validate_payload(input, scale, offset)?;
    if output_gradient.shape != input.shape
        || output_gradient.numeric_dtype() != input.numeric_dtype()
    {
        return Err(NormalizationError::ShapeMismatch);
    }
    if stats.shape != input.shape
        || stats.labels != axes.layout().labels()
        || stats.reduce != geometry.reduce
        || stats.epsilon != options.epsilon
    {
        return Err(NormalizationError::StatisticsMismatch);
    }
    let (dx, dgamma, dbeta) = match (
        &stats.data,
        input.as_f32_slice(),
        scale.as_f32_slice(),
        output_gradient.as_f32_slice(),
    ) {
        (NormalizationStatsData::F32 { mean, inv_std }, Some(x), Some(gamma), Some(dy))
            if mean.len() == geometry.groups && inv_std.len() == geometry.groups =>
        {
            let (dx, dgamma, dbeta) = backward_values(x, gamma, dy, mean, inv_std, &geometry);
            (
                NumericStorage::F32(dx),
                NumericStorage::F32(dgamma),
                NumericStorage::F32(dbeta),
            )
        }
        (NormalizationStatsData::F64 { mean, inv_std }, _, _, _)
            if mean.len() == geometry.groups && inv_std.len() == geometry.groups =>
        {
            let (Some(x), Some(gamma), Some(dy)) = (
                input.as_f64_slice(),
                scale.as_f64_slice(),
                output_gradient.as_f64_slice(),
            ) else {
                return Err(NormalizationError::UnsupportedPrecision);
            };
            let (dx, dgamma, dbeta) = backward_values(x, gamma, dy, mean, inv_std, &geometry);
            (
                NumericStorage::F64(dx),
                NumericStorage::F64(dgamma),
                NumericStorage::F64(dbeta),
            )
        }
        _ => return Err(NormalizationError::StatisticsMismatch),
    };
    Ok(NormalizationGradients {
        input: Tensor::from_numeric_storage(dx, input.shape.clone())
            .map_err(NormalizationError::Storage)?,
        scale: Tensor::from_numeric_storage(dgamma, scale.shape.clone())
            .map_err(NormalizationError::Storage)?,
        offset: Tensor::from_numeric_storage(dbeta, offset.shape.clone())
            .map_err(NormalizationError::Storage)?,
    })
}

fn backward_values<T: Real>(
    x: &[T],
    gamma: &[T],
    dy: &[T],
    mean: &[T],
    inv_std: &[T],
    g: &Geometry,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut dgamma = vec![T::zero(); g.channels];
    let mut dbeta = vec![T::zero(); g.channels];
    let mut sum_dxhat = vec![T::zero(); g.groups];
    let mut sum_dxhat_xhat = vec![T::zero(); g.groups];
    for (index, &value) in x.iter().enumerate() {
        let (group, channel) = g.group_and_channel(index);
        let xhat = (value - mean[group]) * inv_std[group];
        let dxhat = dy[index] * gamma[channel];
        dgamma[channel] += dy[index] * xhat;
        dbeta[channel] += dy[index];
        sum_dxhat[group] += dxhat;
        sum_dxhat_xhat[group] += dxhat * xhat;
    }
    let count = T::from_f64(g.group_size as f64);
    let mut dx = vec![T::zero(); x.len()];
    for (index, &value) in x.iter().enumerate() {
        let (group, channel) = g.group_and_channel(index);
        let xhat = (value - mean[group]) * inv_std[group];
        let dxhat = dy[index] * gamma[channel];
        dx[index] = inv_std[group] / count
            * (count * dxhat - sum_dxhat[group] - xhat * sum_dxhat_xhat[group]);
    }
    (dx, dgamma, dbeta)
}
