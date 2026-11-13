//! Native typed reference kernels over the existing authenticated allocation map.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum Unary {
    Sin,
    Cos,
    Abs,
    Exp,
    Expm1,
    Log,
    Sqrt,
    Conj,
}

impl InProcessProvider {
    pub(super) fn conjugate_optional(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result = (|| -> Result<Option<GpuTensorHandle>> {
            {
                let guard = numeric_registry()
                    .lock_for(self.device_id)
                    .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
                let data = guard
                    .get(input)
                    .ok_or_else(|| anyhow!("unary input is not a live owned allocation"))?;
                if runmat_accelerate_api::handle_is_logical(input)
                    || matches!(data, InProcessNumericData::Integer(_))
                {
                    return Ok(None);
                }
            }
            self.native_unary(input, Unary::Conj).map(Some)
        })();
        result.map_err(|error| {
            runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
        })
    }

    pub(super) fn native_unary(
        &self,
        input: &GpuTensorHandle,
        operation: Unary,
    ) -> Result<GpuTensorHandle> {
        let storage = input.physical().storage();
        let output_storage = if matches!(operation, Unary::Abs) {
            GpuTensorStorage::Real
        } else {
            storage
        };
        let output = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("unary input is not a live owned allocation"))?;
            if runmat_accelerate_api::handle_is_logical(input)
                || matches!(data, InProcessNumericData::Integer(_))
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation("floating unary requires numeric F32 or F64 input; class promotion belongs to the domain"));
            }
            if storage == GpuTensorStorage::ComplexInterleaved
                && matches!(
                    operation,
                    Unary::Exp | Unary::Expm1 | Unary::Log | Unary::Sqrt
                )
            {
                // Canonical domains retain their complex host implementation.
                // No direct complex kernel is advertised by this reference hook.
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "direct complex exponential/logarithm/root kernel is not implemented",
                ));
            }
            let count = logical_len_for_shape("unary", input.shape())?;
            let words = count
                .checked_mul(if storage == GpuTensorStorage::ComplexInterleaved {
                    2
                } else {
                    1
                })
                .ok_or_else(|| anyhow!("unary storage extent overflow"))?;
            ensure!(
                data.len() == words,
                "unary payload does not match admitted view extent"
            );
            match data {
                InProcessNumericData::F32(values) => {
                    InProcessNumericData::F32(map_f32(values, storage, operation))
                }
                InProcessNumericData::F64(values) => {
                    InProcessNumericData::F64(map_f64(values, storage, operation))
                }
                InProcessNumericData::Integer(_) => unreachable!("integer class rejected above"),
            }
        };
        // Never reenter allocation publication while holding the payload map.
        self.allocate_numeric_data(output, input.shape().to_vec(), output_storage)
    }

    pub(super) fn native_extremum(
        &self,
        input: &GpuTensorHandle,
        minimum: bool,
    ) -> Result<GpuTensorHandle> {
        let output = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("extremum input is not a live owned allocation"))?;
            if input.physical().storage() != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_is_logical(input)
                || matches!(data, InProcessNumericData::Integer(_))
            {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "real floating extremum requires F32 or F64 input",
                ));
            }
            ensure!(
                data.len() == logical_len_for_shape("extremum", input.shape())?,
                "extremum payload does not match admitted view extent"
            );
            // Preserve the existing omit-NaN fold and empty +/-Infinity identity.
            match data {
                InProcessNumericData::F32(values) => InProcessNumericData::F32(vec![if minimum {
                    values.iter().copied().fold(f32::INFINITY, f32::min)
                } else {
                    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                }]),
                InProcessNumericData::F64(values) => InProcessNumericData::F64(vec![if minimum {
                    values.iter().copied().fold(f64::INFINITY, f64::min)
                } else {
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                }]),
                InProcessNumericData::Integer(_) => unreachable!("integer class rejected above"),
            }
        };
        self.allocate_numeric_data(output, vec![1, 1], GpuTensorStorage::Real)
    }
}

fn map_f32(values: &[f32], storage: GpuTensorStorage, operation: Unary) -> Vec<f32> {
    if storage == GpuTensorStorage::Real {
        return values
            .iter()
            .map(|&x| match operation {
                Unary::Sin => x.sin(),
                Unary::Cos => x.cos(),
                Unary::Abs => x.abs(),
                Unary::Exp => x.exp(),
                Unary::Expm1 => x.exp_m1(),
                Unary::Log => x.ln(),
                Unary::Conj => x,
                Unary::Sqrt => {
                    if x == 0.0 {
                        0.0
                    } else {
                        x.sqrt()
                    }
                }
            })
            .collect();
    }
    if matches!(operation, Unary::Abs) {
        return values.chunks_exact(2).map(|z| z[0].hypot(z[1])).collect();
    }
    if matches!(operation, Unary::Conj) {
        // Negate native Single lanes directly: no widening or NaN canonicalization.
        return values.chunks_exact(2).flat_map(|z| [z[0], -z[1]]).collect();
    }
    values
        .chunks_exact(2)
        .flat_map(|z| {
            // The owning trig host policy evaluates complex Single components in
            // f64, then rounds each component. This is not an F64 payload mirror.
            let pair = match operation {
                Unary::Sin => sin_complex_host(f64::from(z[0]), f64::from(z[1])),
                Unary::Cos => cos_complex_host(f64::from(z[0]), f64::from(z[1])),
                _ => unreachable!("unsupported complex operation rejected before mapping"),
            };
            [pair.0 as f32, pair.1 as f32]
        })
        .collect()
}

fn map_f64(values: &[f64], storage: GpuTensorStorage, operation: Unary) -> Vec<f64> {
    if storage == GpuTensorStorage::Real {
        return values
            .iter()
            .map(|&x| match operation {
                Unary::Sin => x.sin(),
                Unary::Cos => x.cos(),
                Unary::Abs => x.abs(),
                Unary::Exp => x.exp(),
                Unary::Expm1 => x.exp_m1(),
                Unary::Log => x.ln(),
                Unary::Conj => x,
                Unary::Sqrt => {
                    if x == 0.0 {
                        0.0
                    } else {
                        x.sqrt()
                    }
                }
            })
            .collect();
    }
    if matches!(operation, Unary::Abs) {
        return values.chunks_exact(2).map(|z| z[0].hypot(z[1])).collect();
    }
    values
        .chunks_exact(2)
        .flat_map(|z| {
            let pair = match operation {
                Unary::Sin => sin_complex_host(z[0], z[1]),
                Unary::Cos => cos_complex_host(z[0], z[1]),
                Unary::Conj => (z[0], -z[1]),
                _ => unreachable!("unsupported complex operation rejected before mapping"),
            };
            [pair.0, pair.1]
        })
        .collect()
}

#[cfg(test)]
#[path = "unary/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "unary/conjugation_tests.rs"]
mod conjugation_tests;
