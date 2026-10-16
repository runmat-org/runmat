//! Exact real-floating host boundary for WGPU linear algebra fallbacks.
//!
//! The provider's preferred precision is not the class of an existing input
//! allocation. These transfers preserve the physical F32/F64 element type;
//! only an algorithm that explicitly computes in double may widen internally.

use super::*;
use runmat_accelerate_api::{
    unsupported_provider_operation, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostNumericTensorView,
};

impl WgpuProvider {
    pub(crate) fn real_floating_input_precision_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<ProviderPrecision> {
        let entry = self.get_entry_raw(handle)?;
        if entry.element_type != handle.physical().element_type()
            || entry.storage != runmat_accelerate_api::handle_storage(handle)
        {
            return Err(anyhow!(
                "real floating linear algebra has contradictory physical and handle metadata"
            ));
        }
        if entry.storage != GpuTensorStorage::Real
            || entry.logical
            || entry.integer_type().is_some()
        {
            return Err(unsupported_provider_operation(
                "real floating linear algebra requires real F32 or F64 storage",
            ));
        }
        let precision = match entry.element_type {
            runmat_accelerate_api::NumericElementType::F32 => ProviderPrecision::F32,
            runmat_accelerate_api::NumericElementType::F64 => ProviderPrecision::F64,
            _ => {
                return Err(unsupported_provider_operation(
                    "real floating linear algebra requires F32 or F64 storage",
                ))
            }
        };
        if runmat_accelerate_api::handle_precision(handle) != Some(precision) {
            return Err(anyhow!(
                "real floating linear algebra has contradictory physical and handle metadata"
            ));
        }
        Ok(precision)
    }

    pub(crate) async fn download_real_floating_tensor_exec(
        &self,
        handle: &GpuTensorHandle,
    ) -> Result<Tensor> {
        let precision = self.real_floating_input_precision_exec(handle)?;
        let HostNumericTensorOwned {
            data,
            shape,
            storage,
        } = self.download_numeric_exec(handle).await?;
        if storage != GpuTensorStorage::Real {
            return Err(anyhow!("real floating readback changed physical storage"));
        }
        match (precision, data) {
            (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                Tensor::from_f32(values, shape)
                    .map_err(|error| anyhow!("real floating readback: {error}"))
            }
            (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => {
                Tensor::new(values, shape)
                    .map_err(|error| anyhow!("real floating readback: {error}"))
            }
            _ => Err(anyhow!("real floating readback changed physical precision")),
        }
    }

    pub(crate) fn upload_real_floating_tensor_exec(
        &self,
        tensor: &Tensor,
    ) -> Result<GpuTensorHandle> {
        let data = if let Some(values) = tensor.as_f32_slice() {
            HostNumericDataView::F32(values)
        } else if let Some(values) = tensor.as_f64_slice() {
            HostNumericDataView::F64(values)
        } else {
            return Err(unsupported_provider_operation(
                "real floating linear algebra output must be F32 or F64",
            ));
        };
        self.upload_numeric_exec(&HostNumericTensorView {
            data,
            shape: tensor.shape(),
            storage: GpuTensorStorage::Real,
        })
    }

    pub(crate) fn upload_real_floating_values_exec(
        &self,
        values: &[f64],
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> Result<GpuTensorHandle> {
        match precision {
            ProviderPrecision::F32 => {
                let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
                self.upload_numeric_exec(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&values),
                    shape,
                    storage: GpuTensorStorage::Real,
                })
            }
            ProviderPrecision::F64 => self.upload_numeric_exec(&HostNumericTensorView {
                data: HostNumericDataView::F64(values),
                shape,
                storage: GpuTensorStorage::Real,
            }),
        }
    }
}
