//! Resident selection using the shared fused broadcast dispatch.
use super::*;
use runmat_accelerate_api::{NumericElementType, ResidentSelectPlan};

impl WgpuProvider {
    pub(crate) fn select_exec(
        &self,
        mask: &GpuTensorHandle,
        then_value: &GpuTensorHandle,
        else_value: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        let plan = ResidentSelectPlan::new(mask, then_value, else_value)?;
        let expected = match self.precision {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        // A foreign or released allocation is an ownership failure, never an
        // unsupported-operation signal that could trigger a fallback.
        let entries = [mask, then_value, else_value]
            .into_iter()
            .map(|handle| self.get_entry_raw(handle))
            .collect::<Result<Vec<_>>>()?;
        if plan.element_type() != expected {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "select requires the provider's native floating precision",
            ));
        }
        if plan.shape().len() > crate::backend::wgpu::params::BCAST_MAX_RANK
            || plan
                .shape()
                .iter()
                .any(|&extent| extent > u32::MAX as usize)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "select output exceeds GPU broadcast rank or extent limits",
            ));
        }
        if plan.len() > u32::MAX as usize {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "select output exceeds GPU dispatch length limit",
            ));
        }
        let uniform_bytes = crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
            3,
            crate::backend::wgpu::params::BCAST_MAX_RANK,
        )
        .ok_or_else(|| anyhow!("select broadcast uniform size overflow"))?;
        if uniform_bytes > u64::from(self.adapter_limits.max_uniform_buffer_binding_size) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "select broadcast uniform exceeds adapter limit",
            ));
        }
        let output_bytes = u64::try_from(plan.len().max(1))
            .ok()
            .and_then(|len| len.checked_mul(expected.element_size() as u64))
            .ok_or_else(|| anyhow!("select output byte count overflow"))?;
        if output_bytes > self.adapter_limits.max_buffer_size
            || output_bytes > u64::from(self.adapter_limits.max_storage_buffer_binding_size)
        {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "select output exceeds adapter storage limits",
            ));
        }
        // Authenticate all three physical records before allocation or submission.
        // The fused dispatcher performs the same check when it binds the records.
        for (handle, entry) in [mask, then_value, else_value].into_iter().zip(&entries) {
            ensure!(
                entry.element_type == expected
                    && entry.storage == GpuTensorStorage::Real
                    && entry.len
                        == handle
                            .shape()
                            .iter()
                            .try_fold(1usize, |n, &extent| n.checked_mul(extent))
                            .ok_or_else(|| anyhow!("select input size overflow"))?,
                "select input allocation and view disagree"
            );
            super::backend_shared::fusion_operand_layout(handle, entry, plan.shape())?;
        }
        let output = self.fused_elementwise_exec(
            &crate::backend::wgpu::shaders::select::shader(self.precision),
            &[mask.clone(), then_value.clone(), else_value.clone()],
            plan.shape(),
            plan.len(),
            GpuTensorStorage::Real,
        )?;
        ensure!(
            ![mask, then_value, else_value]
                .iter()
                .any(|input| output.shares_allocation_with(input)),
            "select provider returned an aliased output"
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        AccelProvider, HostLogicalTensorView, HostNumericDataOwned, HostNumericDataView,
        HostNumericTensorView,
    };

    #[test]
    #[ignore = "requires provisioned WGPU adapter; run explicitly with --ignored"]
    fn actual_wgpu_selects_selected_bits_with_broadcast_and_no_input_alias() {
        let provider = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("resident select qualification requires a WGPU adapter");
        let mask = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[1, 0],
                shape: &[2, 1],
            })
            .unwrap();
        let (then_value, else_value) = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => (
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&[f32::NAN, f32::INFINITY, -0.0]),
                        shape: &[1, 3],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap(),
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F32(&[f32::NEG_INFINITY, 7.0, 0.0]),
                        shape: &[1, 3],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap(),
            ),
            runmat_accelerate_api::ProviderPrecision::F64 => (
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F64(&[f64::NAN, f64::INFINITY, -0.0]),
                        shape: &[1, 3],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap(),
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F64(&[f64::NEG_INFINITY, 7.0, 0.0]),
                        shape: &[1, 3],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap(),
            ),
        };
        let output = block_on(provider.select(&mask, &then_value, &else_value)).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        assert!(!output.shares_allocation_with(&mask));
        assert!(!output.shares_allocation_with(&then_value));
        assert!(!output.shares_allocation_with(&else_value));
        match block_on(provider.download_numeric(&output)).unwrap().data {
            HostNumericDataOwned::F32(values) => {
                assert!(values[0].is_nan());
                assert_eq!(values[1], f32::NEG_INFINITY);
                assert_eq!(values[2], f32::INFINITY);
                assert_eq!(values[3], 7.0);
                assert_eq!(values[4].to_bits(), (-0.0f32).to_bits());
                assert_eq!(values[5].to_bits(), 0.0f32.to_bits());
            }
            HostNumericDataOwned::F64(values) => {
                assert!(values[0].is_nan());
                assert_eq!(values[1], f64::NEG_INFINITY);
                assert_eq!(values[2], f64::INFINITY);
                assert_eq!(values[3], 7.0);
                assert_eq!(values[4].to_bits(), (-0.0f64).to_bits());
                assert_eq!(values[5].to_bits(), 0.0f64.to_bits());
            }
            _ => panic!("selection changed output class"),
        }
        let reshaped_then = provider.reshape(&then_value, &[1, 1, 3]).unwrap();
        let reshaped_else = provider.reshape(&else_value, &[1, 1, 3]).unwrap();
        let reshaped = block_on(provider.select(&mask, &reshaped_then, &reshaped_else)).unwrap();
        assert_eq!(reshaped.shape(), &[2, 1, 3]);
        assert!(!reshaped.shares_allocation_with(&reshaped_then));
        drop((mask, then_value, else_value, reshaped_then, reshaped_else));
        assert!(block_on(provider.download_numeric(&output)).is_ok());
        assert!(block_on(provider.download_numeric(&reshaped)).is_ok());
    }

    #[test]
    #[ignore = "requires provisioned WGPU adapter; run explicitly with --ignored"]
    fn actual_wgpu_empty_select_preserves_shape() {
        let provider = crate::backend::wgpu::provider::register_wgpu_provider(
            crate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("resident select qualification requires a WGPU adapter");
        let mask = provider
            .upload_logical(&HostLogicalTensorView {
                data: &[],
                shape: &[0, 2],
            })
            .unwrap();
        let value = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[]),
                    shape: &[0, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
            runmat_accelerate_api::ProviderPrecision::F64 => provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&[]),
                    shape: &[0, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
        };
        let output = block_on(provider.select(&mask, &value, &value)).unwrap();
        assert_eq!(output.shape(), &[0, 2]);
        assert!(!output.shares_allocation_with(&value));
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().shape,
            vec![0, 2]
        );
    }
}
