//! A fusion request publishes no allocation until all output representations
//! have been validated and adapted. The owner is captured before any await.

use anyhow::{anyhow, Result};
use runmat_accelerate_api::{
    clear_handle_metadata, AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType,
    ProviderPrecision,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, free_rejected_provider_output, gpu_class_metadata_matches,
    same_gpu_handle,
};
use runmat_value::Value;

struct Allocation {
    handle: GpuTensorHandle,
    owner: &'static dyn AccelProvider,
}

pub(super) struct FusionAllocations {
    borrowed: Vec<GpuTensorHandle>,
    allocations: Vec<Allocation>,
}

impl FusionAllocations {
    pub fn new(inputs: &[Value]) -> Self {
        Self {
            borrowed: inputs
                .iter()
                .filter_map(|value| match value {
                    Value::GpuTensor(handle) => Some(handle.clone()),
                    _ => None,
                })
                .collect(),
            allocations: Vec::new(),
        }
    }

    /// The central numeric-upload validator must see every borrowed or already
    /// owned allocation before it can reject an alias. Validation after the
    /// raw hook is too late if a convenience adapter has already freed it.
    pub fn upload(
        &mut self,
        provider: &'static dyn AccelProvider,
        data: &[f64],
        shape: &[usize],
    ) -> Result<GpuTensorHandle> {
        use runmat_accelerate_api::{
            upload_numeric_protecting, HostNumericDataView, HostNumericTensorView,
        };
        let protected: Vec<_> = self
            .borrowed
            .iter()
            .chain(self.allocations.iter().map(|allocation| &allocation.handle))
            .collect();
        let single;
        let data = match provider.precision() {
            ProviderPrecision::F64 => HostNumericDataView::F64(data),
            ProviderPrecision::F32 => {
                single = data.iter().map(|value| *value as f32).collect::<Vec<_>>();
                HostNumericDataView::F32(&single)
            }
        };
        let handle = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data,
                shape,
                storage: GpuTensorStorage::Real,
            },
            &protected,
        )?;
        self.admit(handle, provider, shape, provider.precision())
    }

    pub fn admit(
        &mut self,
        handle: GpuTensorHandle,
        provider: &'static dyn AccelProvider,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> Result<GpuTensorHandle> {
        let element = match precision {
            ProviderPrecision::F32 => NumericElementType::F32,
            ProviderPrecision::F64 => NumericElementType::F64,
        };
        let owner = exact_provider_for_handle(&handle);
        let valid = handle.shape == shape
            && handle.device_id == provider.device_id()
            && handle.descriptor.storage == Some(GpuTensorStorage::Real)
            && handle.descriptor.element_type == Some(element)
            && owner.is_some_and(|owner| std::ptr::eq(owner, provider))
            && gpu_class_metadata_matches(&handle, Some(precision), None, false)
            && !self
                .borrowed
                .iter()
                .any(|input| same_gpu_handle(input, &handle));
        if !valid {
            self.reject(&handle, provider);
            return Err(anyhow!(
                "fusion provider returned an invalid floating allocation"
            ));
        }
        if !self
            .allocations
            .iter()
            .any(|allocation| same_gpu_handle(&allocation.handle, &handle))
        {
            self.allocations.push(Allocation {
                handle: handle.clone(),
                owner: provider,
            });
        }
        Ok(handle)
    }

    pub fn reject(&self, handle: &GpuTensorHandle, provider: &'static dyn AccelProvider) {
        let protected: Vec<_> = self
            .borrowed
            .iter()
            .chain(self.allocations.iter().map(|allocation| &allocation.handle))
            .collect();
        free_rejected_provider_output(handle, &protected, provider);
    }

    /// A malformed batch must release even the entries after the first invalid
    /// one, while repeated aliases are released only once.
    pub fn admit_batch(
        &mut self,
        handles: Vec<GpuTensorHandle>,
        provider: &'static dyn AccelProvider,
        count: usize,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> Result<Vec<GpuTensorHandle>> {
        if handles.len() != count {
            self.reject_batch(&handles, provider);
            return Err(anyhow!("fusion provider returned the wrong output count"));
        }
        for handle in &handles {
            // Validate without losing the remaining returned allocations.
            if self
                .admit(handle.clone(), provider, shape, precision)
                .is_err()
            {
                // admit already rejected this handle; exclude its duplicates.
                let remaining: Vec<_> = handles
                    .iter()
                    .filter(|other| !same_gpu_handle(handle, other))
                    .cloned()
                    .collect();
                self.reject_batch(&remaining, provider);
                return Err(anyhow!("fusion provider returned a malformed output batch"));
            }
        }
        Ok(handles)
    }

    fn reject_batch(&self, handles: &[GpuTensorHandle], provider: &'static dyn AccelProvider) {
        for (index, handle) in handles.iter().enumerate() {
            if !handles[..index]
                .iter()
                .any(|previous| same_gpu_handle(previous, handle))
            {
                self.reject(handle, provider);
            }
        }
    }

    /// Consume the transaction only after all results are ready. Non-published
    /// uploads/intermediates are released by Drop; aliases are adopted once.
    pub fn publish(mut self, outputs: &[&GpuTensorHandle]) {
        self.allocations.retain(|allocation| {
            !outputs
                .iter()
                .any(|output| same_gpu_handle(&allocation.handle, output))
        });
    }
}

impl Drop for FusionAllocations {
    fn drop(&mut self) {
        for allocation in &self.allocations {
            if allocation.owner.free(&allocation.handle).is_ok() {
                clear_handle_metadata(&allocation.handle);
            }
        }
    }
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
