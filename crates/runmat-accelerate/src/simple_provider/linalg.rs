//! One authenticated real-floating boundary for in-process matrix algorithms.
//! Only a typed pre-effect Unsupported admission can authorize host replay.

use super::*;
use anyhow::Context;
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};
use runmat_value::{NumericDType, NumericStorage};

pub(super) fn host_error(source: RuntimeError) -> ProviderExecutionError {
    ProviderExecutionError::from_source(source).with_unproven_effects()
}

impl InProcessProvider {
    pub(super) fn real_floating_linalg_input(
        &self,
        handle: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Tensor> {
        self.read_real_floating_linalg_input(handle)
            .map_err(|source| {
                let error = ProviderExecutionError::from(source);
                if error.is_unsupported() {
                    error
                } else {
                    error.with_unproven_effects()
                }
            })
    }

    fn read_real_floating_linalg_input(&self, handle: &GpuTensorHandle) -> Result<Tensor> {
        ensure!(
            handle.device_id() == self.device_id,
            "real floating linear algebra: input belongs to another device"
        );
        handle
            .allocation_lease()
            .validate_live()
            .context("real floating linear algebra: input lease is not live")?;
        if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(handle)
            || runmat_accelerate_api::handle_integer_type(handle).is_some()
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "real floating linear algebra requires real F32 or F64 storage",
            ));
        }
        let data = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            let producer_logical = guard.producer_is_logical(handle).ok_or_else(|| {
                anyhow!("real floating linear algebra: input is not a live owned allocation")
            })?;
            ensure!(
                !producer_logical,
                "real floating linear algebra has contradictory logical metadata"
            );
            let source = guard.get(handle).ok_or_else(|| {
                anyhow!(
                    "real floating linear algebra: unknown buffer {}",
                    handle.buffer_id()
                )
            })?;
            source.clone()
        };
        match (runmat_accelerate_api::handle_precision(handle), data) {
            (Some(ProviderPrecision::F32), InProcessNumericData::F32(values)) => {
                Tensor::from_f32(values, handle.shape().to_vec()).map_err(anyhow::Error::msg)
            }
            (Some(ProviderPrecision::F64), InProcessNumericData::F64(values)) => {
                Tensor::new(values, handle.shape().to_vec()).map_err(anyhow::Error::msg)
            }
            _ => Err(anyhow!(
                "real floating linear algebra has contradictory physical metadata"
            )),
        }
    }

    pub(super) fn publish_real_floating_linalg_tensor(
        &self,
        tensor: Tensor,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let shape = tensor.shape().to_vec();
        let data = match tensor
            .into_numeric_storage()
            .map_err(|error| ProviderExecutionError::from(anyhow!(error)).with_unproven_effects())?
        {
            NumericStorage::F32(values) => InProcessNumericData::F32(values),
            NumericStorage::F64(values) => InProcessNumericData::F64(values),
            _ => {
                return Err(ProviderExecutionError::from(anyhow!(
                    "real floating linear algebra produced malformed output storage"
                ))
                .with_unproven_effects())
            }
        };
        self.allocate_numeric_data(data, shape, GpuTensorStorage::Real)
            .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
    }

    pub(super) fn publish_real_floating_linalg_scalar(
        &self,
        value: f64,
        class: NumericDType,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let data = match class {
            NumericDType::F32 => InProcessNumericData::F32(vec![value as f32]),
            NumericDType::F64 => InProcessNumericData::F64(vec![value]),
            _ => {
                return Err(ProviderExecutionError::from(anyhow!(
                    "real floating linear algebra produced malformed scalar class"
                ))
                .with_unproven_effects())
            }
        };
        self.allocate_numeric_data(data, vec![1, 1], GpuTensorStorage::Real)
            .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())
    }
}
