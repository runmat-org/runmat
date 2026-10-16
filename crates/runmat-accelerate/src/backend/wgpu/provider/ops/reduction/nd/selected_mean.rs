use super::{geometry::NdReductionGeometry, GpuTensorHandle, WgpuProvider};
use anyhow::{anyhow, ensure, Result};

/// Shared admission at preparation and consumption. A public route description
/// is untrusted until its source, geometry and current device limits validate.
struct ColdNdResources {
    output_bytes: u64,
    pair_bytes: u64,
    uniform_bytes: u64,
}

impl ColdNdResources {
    fn new(
        cols: usize,
        element_bytes: usize,
        input_bytes: u64,
        limits: &wgpu::Limits,
    ) -> Option<Self> {
        let output_bytes = u64::try_from(cols)
            .ok()?
            .checked_mul(u64::try_from(element_bytes).ok()?)?;
        let pair_bytes = output_bytes.checked_mul(2)?;
        let uniform_bytes =
            std::mem::size_of::<crate::backend::wgpu::params::ReduceNdParams>() as u64;
        let storage_limit = u64::from(limits.max_storage_buffer_binding_size);
        if cols > limits.max_compute_workgroups_per_dimension as usize
            || input_bytes > limits.max_buffer_size
            || input_bytes > storage_limit
            || output_bytes > limits.max_buffer_size
            || output_bytes > storage_limit
            || uniform_bytes > limits.max_buffer_size
            || uniform_bytes > u64::from(limits.max_uniform_buffer_binding_size)
        {
            return None;
        }
        Some(Self {
            output_bytes,
            pair_bytes,
            uniform_bytes,
        })
    }
}

impl WgpuProvider {
    pub(in crate::backend::wgpu::provider::backend) fn prepare_cold_mean_nd_exec(
        &self,
        source: &GpuTensorHandle,
        dims: &[usize],
    ) -> Result<Option<runmat_accelerate_api::ProviderColdNdMean>> {
        let entry = self.real_floating_entry(source)?;
        if entry.storage != runmat_accelerate_api::GpuTensorStorage::Real
            || runmat_accelerate_api::handle_is_logical(source)
        {
            return Ok(None);
        }
        // Lazy materialization is a distinct kernel/submission that needs its
        // own selected receipt; this route owns canonical input buffers only.
        if runmat_accelerate_api::handle_is_transposed(source) {
            return Ok(None);
        }
        let geometry = NdReductionGeometry::new(&entry.shape, dims, "reduce_moments_nd")?;
        if geometry.reduce.as_slice() != dims {
            return Ok(None);
        }
        let Some(resources) = ColdNdResources::new(
            geometry.cols,
            entry.element_type.element_size(),
            entry.buffer.size(),
            &self.adapter_limits,
        ) else {
            return Ok(None);
        };
        let Some(cache_payload) = Self::cold_key_payload_bytes(geometry.reduce.len()) else {
            return Ok(None);
        };
        Ok(Some(runmat_accelerate_api::ProviderColdNdMean::new(
            source.clone(),
            geometry.reduce,
            geometry.out_shape,
            resources.output_bytes,
            resources.uniform_bytes,
            cache_payload,
        )))
    }

    pub(in crate::backend::wgpu::provider::backend) fn execute_cold_mean_nd_exec(
        &self,
        mut selected: runmat_accelerate_api::ProviderColdNdMean,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<runmat_accelerate_api::ProviderExecutedColdNdMean> {
        let source = selected.source().clone();
        let entry = self.real_floating_entry(&source)?;
        ensure!(
            entry.storage == runmat_accelerate_api::GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(&source),
            "selected cold N-D mean source class changed"
        );
        let geometry =
            NdReductionGeometry::new(&entry.shape, selected.dims(), "reduce_moments_nd")?;
        let resources = ColdNdResources::new(
            geometry.cols,
            entry.element_type.element_size(),
            entry.buffer.size(),
            &self.adapter_limits,
        )
        .ok_or_else(|| anyhow!("selected cold ND mean exceeds device binding limits"))?;
        ensure!(
            !runmat_accelerate_api::handle_is_transposed(&source)
                && geometry.reduce.as_slice() == selected.dims()
                && geometry.out_shape.as_slice() == selected.output_shape()
                && selected.requested_uniform_bytes() == resources.uniform_bytes
                && selected.requested_pair_bytes() == Some(resources.pair_bytes)
                && Self::cold_key_payload_bytes(geometry.reduce.len())
                    == Some(selected.requested_cache_metadata_payload_bytes()),
            "selected cold ND mean geometry changed"
        );
        // Selection promised this two-output kernel. A newly warmed cache
        // cannot change the selected work or arithmetic at consumption.
        let moments = self.reduce_moments_nd_exec_admitted(&source, selected.dims(), admission)?;
        // The selected cold pair enters the provider reuse cache below. Give
        // those two cache-held aliases their own move-only host-view claims;
        // the unclaimed registration views remain request-local temporaries.
        let moments = if let Some([mean_grant, ex2_grant]) = selected.take_cache_view_grants() {
            runmat_accelerate_api::ProviderMoments2 {
                mean: moments
                    .mean
                    .with_view_retained_grant(moments.mean.view().clone(), mean_grant)?,
                ex2: moments
                    .ex2
                    .with_view_retained_grant(moments.ex2.view().clone(), ex2_grant)?,
            }
        } else {
            moments
        };
        let mean_bytes = self.get_entry_raw(&moments.mean)?.buffer.size();
        let ex2_bytes = self.get_entry_raw(&moments.ex2)?.buffer.size();
        let observed_pair_bytes = mean_bytes
            .checked_add(ex2_bytes)
            .ok_or_else(|| anyhow!("cold ND moments observed byte overflow"))?;
        let cache_metadata_claim = selected.take_cache_metadata_claim();
        self.remember_moments(
            &source,
            selected.dims(),
            &moments.mean,
            &moments.ex2,
            cache_metadata_claim,
        )?;
        Ok(runmat_accelerate_api::ProviderExecutedColdNdMean {
            mean: moments.mean,
            ex2: moments.ex2,
            requested_pair_bytes: resources.pair_bytes,
            observed_pair_bytes,
            requested_uniform_bytes: resources.uniform_bytes,
            selected_submissions: 1,
        })
    }
}

#[cfg(test)]
#[path = "selected_mean/tests.rs"]
mod tests;
