//! Native deep-learning activations and adjoints over authenticated payloads.

use super::*;
use runmat_accelerate_api::ProviderActivationAdjoint;
use std::ops::{Add, Div, Mul, Sub};

#[cfg(test)]
mod tests;

trait FloatLane:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_INFINITY: Self;
    fn from_f64(value: f64) -> Self;
    fn exp(self) -> Self;
    fn is_nan(self) -> bool;
}

impl FloatLane for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    fn from_f64(value: f64) -> Self {
        value as Self
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl FloatLane for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    fn from_f64(value: f64) -> Self {
        value
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

fn elu_typed<T: FloatLane>(input: &[T], alpha: T) -> Vec<T> {
    input
        .iter()
        .copied()
        .map(|value| {
            if value >= T::ZERO {
                value
            } else {
                alpha * (value.exp() - T::ONE)
            }
        })
        .collect()
}

fn relu_typed<T: FloatLane>(input: &[T]) -> Vec<T> {
    input
        .iter()
        .copied()
        .map(|value| {
            if value.is_nan() || value > T::ZERO {
                value
            } else {
                T::ZERO
            }
        })
        .collect()
}

fn softmax_typed<T: FloatLane>(input: &[T], rows: usize, cols: usize) -> Result<Vec<T>> {
    let mut output = vec![T::ZERO; input.len()];
    for row in 0..rows {
        let mut maximum = T::NEG_INFINITY;
        for col in 0..cols {
            let value = input[row + col * rows];
            if value > maximum || value.is_nan() {
                maximum = value;
            }
        }
        let mut sum = T::ZERO;
        for col in 0..cols {
            let index = row + col * rows;
            let value = (input[index] - maximum).exp();
            output[index] = value;
            sum = sum + value;
        }
        // An undefined nonfinite row yields NaNs, as on the host path. For
        // finite nonempty rows the maximum term guarantees a positive sum.
        for col in 0..cols {
            output[row + col * rows] = output[row + col * rows] / sum;
        }
    }
    Ok(output)
}

fn activation_typed<T: FloatLane>(
    input: &[T],
    upstream: &[T],
    operation: ProviderActivationAdjoint,
) -> Vec<T> {
    input
        .iter()
        .zip(upstream)
        .map(|(&x, &dy)| {
            let derivative = match operation {
                ProviderActivationAdjoint::Relu if x.is_nan() => x,
                ProviderActivationAdjoint::Relu if x > T::ZERO => T::ONE,
                ProviderActivationAdjoint::Relu => T::ZERO,
                ProviderActivationAdjoint::Elu { .. } if x >= T::ZERO => T::ONE,
                ProviderActivationAdjoint::Elu { alpha } => T::from_f64(alpha) * x.exp(),
            };
            dy * derivative
        })
        .collect()
}

impl InProcessProvider {
    pub(super) fn native_relu(&self, input: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let count = validate_real_input(input, "activation_relu")?;
        let result = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("activation_relu: allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("activation_relu: input retired"))?;
            match data {
                InProcessNumericData::F32(values) => {
                    ensure!(
                        values.len() == count,
                        "activation_relu: payload extent mismatch"
                    );
                    InProcessNumericData::F32(relu_typed(values))
                }
                InProcessNumericData::F64(values) => {
                    ensure!(
                        values.len() == count,
                        "activation_relu: payload extent mismatch"
                    );
                    InProcessNumericData::F64(relu_typed(values))
                }
                InProcessNumericData::Integer(_) => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "activation_relu: native floating storage is required",
                    ))
                }
            }
        };
        self.allocate_numeric_data(result, input.shape().to_vec(), GpuTensorStorage::Real)
    }

    pub(super) fn native_elu(
        &self,
        input: &GpuTensorHandle,
        alpha: f64,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            alpha.is_finite() && alpha >= 0.0,
            "activation_elu: invalid alpha"
        );
        let count = validate_real_input(input, "activation_elu")?;
        let result = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("activation_elu: allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("activation_elu: input retired"))?;
            match data {
                InProcessNumericData::F32(values) => {
                    ensure!(
                        (alpha as f32).is_finite(),
                        "activation_elu: alpha exceeds single precision"
                    );
                    ensure!(
                        values.len() == count,
                        "activation_elu: payload extent mismatch"
                    );
                    InProcessNumericData::F32(elu_typed(values, alpha as f32))
                }
                InProcessNumericData::F64(values) => {
                    ensure!(
                        values.len() == count,
                        "activation_elu: payload extent mismatch"
                    );
                    InProcessNumericData::F64(elu_typed(values, alpha))
                }
                InProcessNumericData::Integer(_) => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "activation_elu: native floating storage is required",
                    ))
                }
            }
        };
        self.allocate_numeric_data(result, input.shape().to_vec(), GpuTensorStorage::Real)
    }

    pub(super) fn native_softmax_rows(&self, input: &GpuTensorHandle) -> Result<GpuTensorHandle> {
        let count = validate_real_input(input, "activation_softmax_rows")?;
        ensure!(
            input.shape().len() == 2,
            "activation_softmax_rows: input must be 2-D"
        );
        let rows = input.shape()[0];
        let cols = input.shape()[1];
        let result = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("activation_softmax_rows: allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("activation_softmax_rows: input retired"))?;
            match data {
                InProcessNumericData::F32(values) => {
                    ensure!(
                        values.len() == count,
                        "activation_softmax_rows: payload extent mismatch"
                    );
                    InProcessNumericData::F32(softmax_typed(values, rows, cols)?)
                }
                InProcessNumericData::F64(values) => {
                    ensure!(
                        values.len() == count,
                        "activation_softmax_rows: payload extent mismatch"
                    );
                    InProcessNumericData::F64(softmax_typed(values, rows, cols)?)
                }
                InProcessNumericData::Integer(_) => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "activation_softmax_rows: native floating storage is required",
                    ))
                }
            }
        };
        self.allocate_numeric_data(result, input.shape().to_vec(), GpuTensorStorage::Real)
    }

    pub(super) fn native_activation_adjoint(
        &self,
        input: &GpuTensorHandle,
        upstream: &GpuTensorHandle,
        operation: ProviderActivationAdjoint,
    ) -> Result<GpuTensorHandle> {
        ensure!(
            input.shape() == upstream.shape(),
            "activation_adjoint: shape mismatch"
        );
        if let ProviderActivationAdjoint::Elu { alpha } = operation {
            ensure!(
                alpha.is_finite() && alpha >= 0.0,
                "activation_adjoint: invalid ELU alpha"
            );
            if input.physical().element_type() == NumericElementType::F32 {
                ensure!(
                    (alpha as f32).is_finite(),
                    "activation_adjoint: ELU alpha exceeds single precision"
                );
            }
        }
        let expected = validate_real_input(input, "activation_adjoint")?;
        validate_real_input(upstream, "activation_adjoint")?;
        let result = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("activation_adjoint: allocation owner is poisoned"))?;
            let x = guard
                .get(input)
                .ok_or_else(|| anyhow!("activation_adjoint: input retired"))?;
            let dy = guard
                .get(upstream)
                .ok_or_else(|| anyhow!("activation_adjoint: upstream retired"))?;
            match (x, dy) {
                (InProcessNumericData::F32(x), InProcessNumericData::F32(dy)) => {
                    ensure!(
                        x.len() == expected && dy.len() == expected,
                        "activation_adjoint: payload extent mismatch"
                    );
                    InProcessNumericData::F32(activation_typed(x, dy, operation))
                }
                (InProcessNumericData::F64(x), InProcessNumericData::F64(dy)) => {
                    ensure!(
                        x.len() == expected && dy.len() == expected,
                        "activation_adjoint: payload extent mismatch"
                    );
                    InProcessNumericData::F64(activation_typed(x, dy, operation))
                }
                _ => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "activation_adjoint: matching native floating classes are required",
                    ))
                }
            }
        };
        self.allocate_numeric_data(result, input.shape().to_vec(), GpuTensorStorage::Real)
    }
}

fn validate_real_input(input: &GpuTensorHandle, operation: &'static str) -> Result<usize> {
    ensure!(
        input.physical().storage() == GpuTensorStorage::Real
            && !runmat_accelerate_api::handle_is_logical(input)
            && input.view().layout() == runmat_accelerate_api::GpuTensorLayout::Contiguous,
        "{operation}: real contiguous floating storage is required"
    );
    logical_len_for_shape(operation, input.shape())
}
