//! Pure physical view-chain admission against the owning WGPU allocation.

use super::*;
use runmat_accelerate_api::{
    NumericElementType, ProviderViewCapability, ProviderViewChain, ProviderViewPlanError,
    ProviderViewSource, ProviderViewStep,
};

impl WgpuProvider {
    pub(crate) fn provider_view_chain_capability_exec(
        &self,
        anchor: &GpuTensorHandle,
        chain: &ProviderViewChain,
    ) -> Result<ProviderViewCapability> {
        let entry = self.get_entry_raw(anchor)?;
        if entry.element_type != anchor.physical().element_type()
            || entry.storage != anchor.physical().storage()
            || entry.logical != anchor.view().is_logical()
        {
            return Err(anyhow!(
                "physical view anchor differs from owning descriptor"
            ));
        }
        if let ProviderViewSource::Planned(view) = &chain.source {
            if view.is_logical() != anchor.view().is_logical() {
                return Err(anyhow!("planned physical view changes logical class"));
            }
        }
        // Generic bitwise view operations support more storage classes, but
        // this route admits only the native real class whose byte geometry is
        // identical to the accelerator's arithmetic element geometry.
        if entry.storage != GpuTensorStorage::Real || entry.logical {
            return Ok(ProviderViewCapability::UnsupportedOperation);
        }
        self.provider_view_chain_capability_for_type(anchor.view(), entry.element_type, chain)
    }

    pub(crate) fn provider_view_chain_capability_for_type(
        &self,
        current: &runmat_accelerate_api::GpuTensorView,
        element_type: NumericElementType,
        chain: &ProviderViewChain,
    ) -> Result<ProviderViewCapability> {
        if let ProviderViewSource::Planned(view) = &chain.source {
            ensure!(
                view.is_logical() == current.is_logical(),
                "planned physical view changes logical class"
            );
        }
        if current.is_logical() || !self.supports_native_real_element_type(element_type) {
            return Ok(ProviderViewCapability::UnsupportedOperation);
        }
        let outcome = match chain.analyze(current, element_type) {
            Ok(outcome) => outcome,
            Err(ProviderViewPlanError::SizeOverflow) => {
                return Ok(ProviderViewCapability::ResourceLimit);
            }
            Err(error) => return Err(anyhow!("invalid physical view transition: {error:?}")),
        };
        let source_view = match &chain.source {
            ProviderViewSource::Current => current,
            ProviderViewSource::Planned(view) => view,
        };
        if source_view
            .shape()
            .iter()
            .chain(chain.steps.iter().flat_map(|step| match step {
                ProviderViewStep::Reshape(shape) => shape.as_slice(),
                _ => &[],
            }))
            .any(|extent| *extent > u32::MAX as usize)
        {
            return Ok(ProviderViewCapability::ResourceLimit);
        }
        if outcome
            .view
            .shape()
            .iter()
            .any(|extent| *extent > u32::MAX as usize)
        {
            return Ok(ProviderViewCapability::ResourceLimit);
        }
        let limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        if outcome.materialization_bytes > limit {
            return Ok(ProviderViewCapability::ResourceLimit);
        }
        Ok(ProviderViewCapability::Ready(outcome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, AccelProvider, GpuTensorLayout, GpuTensorView,
        HostNumericDataView, HostNumericTensorView, ProviderViewStep, TransposeInfo,
    };

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for physical-view admission"]
    fn query_authenticates_current_and_planned_views_without_effects() {
        let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
        let source_f32 = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let source_f64 = source_f32.map(f64::from);
        let handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: if provider.precision() == runmat_accelerate_api::ProviderPrecision::F32 {
                    HostNumericDataView::F32(&source_f32)
                } else {
                    HostNumericDataView::F64(&source_f64)
                },
                shape: &[3, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let lazy = provider.transpose(&handle).unwrap();
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        let request = ProviderViewChain {
            source: ProviderViewSource::Current,
            steps: vec![ProviderViewStep::Reshape(vec![2, 3])],
        };
        let ProviderViewCapability::Ready(outcome) = provider
            .provider_view_chain_capability_exec(&lazy, &request)
            .unwrap()
        else {
            panic!("lazy view materialization should be admitted")
        };
        assert_eq!(outcome.view.layout(), GpuTensorLayout::Contiguous);
        assert_eq!(outcome.view.shape(), [2, 3]);
        assert_eq!(
            outcome.materialization_bytes,
            6 * handle.physical().element_type().element_size() as u64
        );
        let invalid = ProviderViewChain {
            source: ProviderViewSource::Planned(GpuTensorView::contiguous(vec![2, 3]).with_layout(
                GpuTensorLayout::Transpose(TransposeInfo {
                    base_rows: 2,
                    base_cols: 3,
                }),
            )),
            steps: vec![ProviderViewStep::Permute2D],
        };
        assert!(provider
            .provider_view_chain_capability_exec(&handle, &invalid)
            .is_err());
        let overflow = ProviderViewChain {
            source: ProviderViewSource::Planned(GpuTensorView::contiguous(vec![usize::MAX, 2])),
            steps: vec![ProviderViewStep::Reshape(vec![usize::MAX, 2])],
        };
        assert!(matches!(
            provider
                .provider_view_chain_capability_exec(&handle, &overflow)
                .unwrap(),
            ProviderViewCapability::ResourceLimit
        ));
        let foreign = TestAllocationOwner::new()
            .allocate_zeroed(
                vec![3, 2],
                handle.physical().element_type(),
                GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(provider
            .provider_view_chain_capability_exec(&foreign.handle, &request)
            .is_err());
        assert_eq!(provider.telemetry_snapshot(), before);
        assert_eq!(foreign.probe.retirements(), 0);
    }
}
