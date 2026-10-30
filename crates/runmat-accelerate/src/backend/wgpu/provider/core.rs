use anyhow::{anyhow, Result};
use bytemuck::{bytes_of, Pod};
#[cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use runmat_accelerate_api::{
    unsupported_provider_operation, GpuTensorHandle, GpuTensorStorage, IntegerElementType,
    NumericElementType, ProviderPrecision, WgpuEmission,
};
use std::sync::atomic::Ordering as AtomicOrdering;
use tracing::info_span;

use super::backend_shared::NEXT_SUBMISSION_ID;
use super::backend_types::{BufferEntry, NumericBufferRegistration, WgpuProvider};
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use crate::backend::wgpu::residency::{BufferResidency, BufferUsageClass};
use crate::backend::wgpu::types::NumericPrecision;
use crate::fusion::active_fusion;

#[path = "core/readback.rs"]
mod readback;
pub(super) use readback::map_completion_host_upper_bytes;
pub(super) use readback::ReadbackResourceOwner;

impl WgpuProvider {
    pub(crate) fn supports_native_real_element_type(
        &self,
        element_type: NumericElementType,
    ) -> bool {
        matches!(
            (self.precision, element_type),
            (NumericPrecision::F32, NumericElementType::F32)
                | (NumericPrecision::F64, NumericElementType::F64)
        )
    }

    /// Operation entry only. Nested helpers borrow these same two arguments;
    /// they never acquire another read lock behind a pending scoped writer.
    pub(super) fn with_emission<T>(
        &self,
        action: impl for<'p> FnOnce(&WgpuEmission<'p>, &mut PendingSubmissions) -> T,
    ) -> Result<T> {
        // Declared outside the gate so unwind cleanup also runs after unlock.
        let mut pending = PendingSubmissions::new(self.access.clone()).with_profiling(
            self.timestamp_capability,
            self.observation_budget.clone(),
            #[cfg(not(target_arch = "wasm32"))]
            self.observation_progress
                .clone()
                .map(|progress| progress as _),
        );
        let result = self
            .access
            .with_profiled_shared(&self.emission_authority, |emission| {
                action(emission, &mut pending)
            });
        pending.commit();
        #[cfg(target_arch = "wasm32")]
        self.access.poll(wgpu::Maintain::Poll);
        Ok(result?)
    }
    pub(super) fn inherit_elementwise_layout(
        &self,
        output: GpuTensorHandle,
        source: &GpuTensorHandle,
    ) -> Result<GpuTensorHandle> {
        if runmat_accelerate_api::handle_is_transposed(source) {
            Ok(output.with_view(output.view().clone().with_layout(source.view().layout()))?)
        } else {
            Ok(output)
        }
    }
    #[cfg(any(not(target_arch = "wasm32"), test))]
    pub(super) fn map_readback_bytes<'a>(
        &'a self,
        staging: runmat_accelerate_api::WgpuBufferOwner,
        size_bytes: u64,
        context: &'a str,
    ) -> impl std::future::Future<Output = Result<Vec<u8>>> + 'a {
        let staging = readback::track(&self.allocations, staging, None);
        async move {
            readback::read(
                self.access.clone(),
                staging?,
                size_bytes,
                context,
                #[cfg(not(target_arch = "wasm32"))]
                self.observation_progress.clone(),
            )
            .await
        }
    }

    pub(super) fn map_readback_with<'a, T: 'a>(
        &'a self,
        staging: impl ReadbackResourceOwner + 'a,
        context: &'a str,
        decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    ) -> impl std::future::Future<Output = Result<T>> + 'a {
        let staging = readback::track(&self.allocations, staging, None);
        async move {
            readback::read_with(
                self.access.clone(),
                staging?,
                context,
                decode,
                #[cfg(not(target_arch = "wasm32"))]
                self.observation_progress.clone(),
            )
            .await
        }
    }

    pub(super) fn map_readback_with_admitted<'a, T: 'a>(
        &'a self,
        staging: impl ReadbackResourceOwner + 'a,
        context: &'a str,
        decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
        claim: std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
    ) -> impl std::future::Future<Output = Result<T>> + 'a {
        let staging = readback::track(&self.allocations, staging, Some(claim.clone()));
        async move {
            readback::read_with_admitted(
                self.access.clone(),
                staging?,
                context,
                decode,
                claim,
                #[cfg(not(target_arch = "wasm32"))]
                self.observation_progress.clone(),
            )
            .await
        }
    }

    pub(super) fn map_readback_with_admitted_stage<'a, T: 'a>(
        &'a self,
        staging: impl ReadbackResourceOwner + 'a,
        context: &'a str,
        decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
        claim: std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        callback_stage: runmat_accelerate_api::PreparedNumericDownloadStage,
    ) -> impl std::future::Future<Output = Result<T>> + 'a {
        let staging = readback::track(&self.allocations, staging, Some(claim.clone()));
        async move {
            readback::read_with_admitted_stage(
                self.access.clone(),
                staging?,
                context,
                decode,
                claim,
                callback_stage,
                #[cfg(not(target_arch = "wasm32"))]
                self.observation_progress.clone(),
            )
            .await
        }
    }

    pub(super) fn map_readback_bytes_sync(
        &self,
        staging: runmat_accelerate_api::WgpuBufferOwner,
        size_bytes: u64,
        context: &str,
    ) -> Result<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = size_bytes;
            // Ordinary Buffer Drop does not emit into WGPU error scopes; only
            // explicit unmap (not used on this unsupported path) needs access.
            drop(staging);
            Err(anyhow!("{context}: readback requires async path on wasm"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            block_on(self.map_readback_bytes(staging, size_bytes, context))
        }
    }
    pub(super) const BUFFER_RESIDENCY_MAX_PER_KEY: usize = 8;
    // Native-single variance now uses a centered second scan. Old tuning
    // entries describe the former one-scan shader and cannot be reused.
    pub(super) const IMAGE_NORMALIZE_AUTOTUNE_VERSION: u8 = 2;
    pub(super) const IMAGE_NORMALIZE_STREAM_COLD_CAP: u32 = 8;
    pub(super) const IMAGE_NORMALIZE_TARGET_SAMPLES_PER_LANE: f64 = 256.0;
    pub(super) const IMAGE_NORMALIZE_TARGET_LOOP_ITERS_PER_LANE: f64 = 16.0;

    pub(super) fn register_existing_buffer(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
    ) -> Result<GpuTensorHandle> {
        self.register_existing_buffer_with_usage(buffer, shape, len, BufferUsageClass::Generic)
    }

    pub(super) fn register_existing_buffer_with_storage(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        storage: GpuTensorStorage,
    ) -> Result<GpuTensorHandle> {
        self.register_existing_buffer_with_usage_and_storage(
            buffer,
            shape,
            len,
            storage,
            BufferUsageClass::Generic,
        )
    }

    pub(super) fn register_existing_buffer_with_usage(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        usage: BufferUsageClass,
    ) -> Result<GpuTensorHandle> {
        self.register_existing_buffer_with_usage_and_storage(
            buffer,
            shape,
            len,
            GpuTensorStorage::Real,
            usage,
        )
    }

    pub(super) fn register_existing_buffer_with_usage_and_storage(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        storage: GpuTensorStorage,
        usage: BufferUsageClass,
    ) -> Result<GpuTensorHandle> {
        let element_type = match self.precision {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        self.register_numeric_buffer(
            buffer,
            NumericBufferRegistration {
                shape,
                len,
                physical_element_type: element_type,
                storage,
                allocated_bytes: (len as u64).saturating_mul(element_type.element_size() as u64),
                usage,
            },
        )
    }

    pub(super) fn register_numeric_buffer(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        registration: NumericBufferRegistration,
    ) -> Result<GpuTensorHandle> {
        self.register_numeric_buffer_produced(buffer, registration, false)
    }

    pub(super) fn register_logical_buffer(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        usage: BufferUsageClass,
    ) -> Result<GpuTensorHandle> {
        let physical_element_type = match self.precision {
            NumericPrecision::F32 => NumericElementType::F32,
            NumericPrecision::F64 => NumericElementType::F64,
        };
        let allocated_bytes = buffer.size();
        self.register_numeric_buffer_produced(
            buffer,
            NumericBufferRegistration {
                shape,
                len,
                physical_element_type,
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage,
            },
            true,
        )
    }

    pub(super) fn register_numeric_buffer_produced(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        registration: NumericBufferRegistration,
        logical: bool,
    ) -> Result<GpuTensorHandle> {
        self.register_numeric_buffer_produced_admitted(buffer, registration, logical, None)
    }

    pub(super) fn register_numeric_buffer_produced_admitted(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        registration: NumericBufferRegistration,
        logical: bool,
        admission: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> Result<GpuTensorHandle> {
        let NumericBufferRegistration {
            shape,
            len,
            physical_element_type: element_type,
            storage,
            allocated_bytes,
            usage,
        } = registration;

        let entry = BufferEntry {
            logical,
            buffer,
            len,
            shape,
            storage,
            element_type,
            precision: match element_type.precision() {
                Some(ProviderPrecision::F32) => NumericPrecision::F32,
                Some(ProviderPrecision::F64) => NumericPrecision::F64,
                None => self.precision,
            },
            allocated_bytes,
            usage,
            last_submission_id: None,
        };
        self.allocations
            .register_produced_admitted(entry, logical, admission)
    }

    pub(super) fn register_integer_buffer(
        &self,
        buffer: runmat_accelerate_api::WgpuBufferOwner,
        shape: Vec<usize>,
        len: usize,
        element_type: IntegerElementType,
        allocated_bytes: u64,
    ) -> Result<GpuTensorHandle> {
        self.register_numeric_buffer(
            buffer,
            NumericBufferRegistration {
                shape,
                len,
                physical_element_type: element_type.into(),
                storage: GpuTensorStorage::Real,
                allocated_bytes,
                usage: BufferUsageClass::Generic,
            },
        )
    }

    pub(super) fn remember_matmul_sources(
        &self,
        product: &GpuTensorHandle,
        lhs: &GpuTensorHandle,
        rhs: &GpuTensorHandle,
    ) {
        if self.get_entry_raw(product).is_err()
            || self.get_entry_raw(lhs).is_err()
            || self.get_entry_raw(rhs).is_err()
        {
            return;
        }
        log::debug!(
            "remember_matmul_sources: product={} lhs={} rhs={} active_fusion={:?}",
            product.buffer_id(),
            lhs.buffer_id(),
            rhs.buffer_id(),
            active_fusion()
        );
        self.kernel_resources
            .remember_matmul_sources(product, lhs, rhs);
    }

    pub(super) fn mark_buffer_usage(&self, handle: &GpuTensorHandle, usage: BufferUsageClass) {
        // Usage is an optional pooling hint, never a live-operation admission.
        if handle.device_id() == self.runtime_device_id {
            let _ = self.allocations.mark_usage(handle.buffer_id(), usage);
        }
    }

    pub(super) fn record_buffer_submission(&self, buffer_id: u64, submission_id: u32) {
        let _ = self.allocations.record_submission(buffer_id, submission_id);
    }

    pub(super) fn create_storage_buffer_for_usage(
        &self,
        emission: &WgpuEmission<'_>,
        usage: BufferUsageClass,
        len: usize,
        label: &str,
    ) -> Result<(runmat_accelerate_api::WgpuBufferOwner, bool)> {
        let size_bytes = BufferResidency::requested_bytes(len, self.element_size)
            .ok_or_else(|| anyhow!("{label}: allocation byte extent overflow"))?;
        if size_bytes > self.adapter_limits.max_buffer_size {
            return Err(super::backend_shared::gpu_per_buffer_limit_error(
                label,
                size_bytes,
                self.adapter_limits.max_buffer_size,
            ));
        }
        emission.ensure_device(&self.access)?;
        self.allocations
            .acquire(emission, usage, len, self.element_size, label)
    }

    pub(super) fn create_storage_buffer(
        &self,
        emission: &WgpuEmission<'_>,
        len: usize,
        label: &str,
    ) -> Result<runmat_accelerate_api::WgpuBufferOwner> {
        Ok(self
            .create_storage_buffer_for_usage(emission, BufferUsageClass::Generic, len, label)?
            .0)
    }

    /// Allocate a raw-word buffer used by exact integer kernels. Integer
    /// storage is packed as u32 words, independently of float precision.
    pub(super) fn create_integer_word_buffer(
        &self,
        emission: &WgpuEmission<'_>,
        words: usize,
        label: &str,
    ) -> Result<runmat_accelerate_api::WgpuBufferOwner> {
        let bytes = crate::backend::wgpu_storage::raw_word_requested_bytes(words as u64)
            .ok_or_else(|| anyhow!("{label}: integer buffer size overflow"))?;
        if bytes > self.adapter_limits.max_buffer_size {
            return Err(anyhow!("{label}: integer buffer exceeds GPU limits"));
        }
        emission.ensure_device(&self.access)?;
        Ok(runmat_accelerate_api::WgpuBufferOwner::new(
            emission.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ))
    }

    pub(super) fn uniform_buffer<T: Pod>(
        &self,
        emission: &WgpuEmission<'_>,
        data: &T,
        label: &str,
    ) -> Result<wgpu::Buffer> {
        emission.ensure_device(&self.access)?;
        Ok(
            emission.create_initialized_buffer(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
        )
    }

    pub(super) fn prepare_matmul_pipeline(
        &self,
        emission: &WgpuEmission<'_>,
        pending: &mut PendingSubmissions,
    ) -> Result<()> {
        emission.ensure_device(&self.access)?;
        let mut enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-warmup"),
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul.get(emission)?.pipeline);
        }
        self.submit(emission, pending, enc)?;

        let enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-flush-gap"),
            },
        );
        self.submit(emission, pending, enc)?;
        Ok(())
    }

    pub(super) fn prepare_matmul_vec4_pipeline(
        &self,
        emission: &WgpuEmission<'_>,
        pending: &mut PendingSubmissions,
    ) -> Result<()> {
        emission.ensure_device(&self.access)?;
        let mut enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-vec4-warmup"),
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runmat-matmul-vec4-warmup-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.matmul_vec4.get(emission)?.pipeline);
        }
        self.submit(emission, pending, enc)?;

        let enc = pending.profiled_encoder(
            emission,
            &wgpu::CommandEncoderDescriptor {
                label: Some("runmat-matmul-vec4-flush-gap"),
            },
        );
        self.submit(emission, pending, enc)?;
        Ok(())
    }

    pub(super) fn submit(
        &self,
        emission: &WgpuEmission<'_>,
        pending: &mut PendingSubmissions,
        encoder: impl crate::backend::wgpu::dispatch::observation::SubmissionEncoder,
    ) -> Result<u32> {
        emission.ensure_device(&self.access)?;
        let submission_id = NEXT_SUBMISSION_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let _span = info_span!(
            "gpu.dispatch",
            label = "runmat-wgpu-submit",
            submission_id = submission_id
        )
        .entered();
        log::trace!("wgpu submit {}: begin", submission_id);
        let submission_index = pending.submit(emission, encoder)?;
        log::trace!(
            "wgpu submit {}: submitted (index={:?}); progress after emission",
            submission_id,
            submission_index
        );
        Ok(submission_id)
    }
    pub(super) fn submit_numeric_readback(
        &self,
        source: &runmat_accelerate_api::WgpuNumericReadbackSource,
        stage: &runmat_accelerate_api::WgpuNumericReadbackBuffer,
    ) -> Result<()> {
        let mut pending = PendingSubmissions::new(self.access.clone()).with_profiling(
            self.timestamp_capability,
            self.observation_budget.clone(),
            #[cfg(not(target_arch = "wasm32"))]
            self.observation_progress
                .clone()
                .map(|progress| progress as _),
        );
        let submission_id = NEXT_SUBMISSION_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let _span =
            info_span!("gpu.dispatch", label = "runmat-wgpu-submit", submission_id).entered();
        let result = pending.submit_numeric_readback(source, stage);
        pending.commit();
        #[cfg(target_arch = "wasm32")]
        self.access.poll(wgpu::Maintain::Poll);
        result.map(|_| ())
    }
    pub(super) fn get_entry(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        let entry = self.get_entry_raw(handle)?;
        if let Some(element_type) = entry.integer_type() {
            return Err(unsupported_provider_operation(format!(
                "native {:?} gpuArray buffers require an integer provider kernel; floating-point dispatch is not permitted",
                element_type
            )));
        }
        if entry.precision != self.precision {
            return Err(unsupported_provider_operation(format!(
                "native {:?} gpuArray buffer requires a precision-aware provider kernel; {:?} dispatch is not permitted",
                entry.element_type,
                self.precision
            )));
        }
        Ok(entry)
    }

    pub(super) fn get_entry_raw(&self, handle: &GpuTensorHandle) -> Result<BufferEntry> {
        self.allocations.entry_for_handle(handle)
    }

    pub(super) fn get_entry_for_storage_move(
        &self,
        handle: &GpuTensorHandle,
        operation: &str,
    ) -> Result<BufferEntry> {
        let entry = self.get_entry_raw(handle)?;
        if entry.integer_type().is_none() && entry.precision != self.precision {
            return Err(anyhow!(
                "{operation}: native {:?} gpuArray buffer requires a precision-aware provider kernel; {:?} dispatch is not permitted",
                entry.element_type,
                self.precision
            ));
        }
        Ok(entry)
    }
}
