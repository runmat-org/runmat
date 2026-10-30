//! Staging ownership across asynchronous mapping. No emission permit survives
//! a poll/await; cancellation performs only a fresh restricted cleanup phase.
#[cfg(not(target_arch = "wasm32"))]
use super::super::observation_progress::{
    Completion as ProgressCompletion, NativeObservationProgress,
};
use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::WgpuObservationProgressError;
use runmat_accelerate_api::{WgpuAccessError, WgpuDeviceAccess};
use runmat_execution::resource::{ExactShared, RetainedHostAllocationClaim};
use std::{
    borrow::Borrow,
    error::Error,
    fmt,
    future::poll_fn,
    sync::{Arc, Mutex},
    task::{Poll, Waker},
};

/// Exact request-owned completion control allocation. Scheduler-owned Wakers
/// are not charged as a second copy of this request's completion state.
pub(crate) fn map_completion_host_upper_bytes() -> Option<u64> {
    ExactShared::<MapCompletion>::requested_bytes()
        .try_into()
        .ok()
        .and_then(|bytes: u64| {
            let bytes =
                bytes.checked_add(super::super::allocation_state::mapped_stage_host_bytes()?)?;
            // The only request-local callback Box captures the exact shared
            // state. Its claim stays inline in the preallocated progress Job;
            // moving that claim creates no additional request heap backing.
            #[cfg(not(target_arch = "wasm32"))]
            let bytes = bytes.checked_add(
                u64::try_from(std::mem::size_of::<ExactShared<MapCompletion>>()).ok()?,
            )?;
            Some(bytes)
        })
}

struct MapCompletion {
    state: Mutex<MapCompletionState>,
}

/// Every clone keeps the claim outside the exact shared allocation. Field
/// drop order releases the control before its claim, including cancellation.
#[derive(Clone)]
struct FundedMapCompletion {
    state: ExactShared<MapCompletion>,
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FundedMapCompletion {
    fn progress_callback(self) -> ProgressCompletion {
        let Self { state, _claim } = self;
        ProgressCompletion::funded(
            Box::new(move |status| state.get().complete_progress(status)),
            _claim,
        )
    }
}

#[derive(Default)]
struct MapCompletionState {
    result: Option<Result<()>>,
    waker: Option<Waker>,
    #[cfg(not(target_arch = "wasm32"))]
    progress: MapProgress,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
enum MapProgress {
    Pending,
    Complete(std::result::Result<(), WgpuObservationProgressError>),
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for MapProgress {
    fn default() -> Self {
        Self::Complete(Ok(()))
    }
}

impl MapCompletion {
    #[cfg(not(target_arch = "wasm32"))]
    fn complete_progress(&self, status: std::result::Result<(), WgpuObservationProgressError>) {
        let wake = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.progress = MapProgress::Complete(status);
            state.waker.take()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    fn complete(&self, result: Result<()>) {
        let wake = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    async fn wait(&self) -> Result<()> {
        poll_fn(|cx| {
            // Scheduler-owned clone/drop callbacks may re-enter completion.
            // Neither may execute while its state mutex is held.
            let replacement = cx.waker().clone();
            let (result, retired) = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                #[cfg(not(target_arch = "wasm32"))]
                let result = match state.progress {
                    MapProgress::Pending => None,
                    MapProgress::Complete(Ok(())) => state.result.take(),
                    MapProgress::Complete(Err(error)) => Some(match state.result.take() {
                        Some(Err(primary)) => Err(primary),
                        _ => Err(error.into()),
                    }),
                };
                #[cfg(target_arch = "wasm32")]
                let result = state.result.take();
                if let Some(result) = result {
                    (Poll::Ready(result), state.waker.take())
                } else {
                    (Poll::Pending, state.waker.replace(replacement))
                }
            };
            drop(retired);
            result
        })
        .await
    }
}

struct MapCallback {
    completion: FundedMapCompletion,
    numeric_stage: Option<runmat_accelerate_api::PreparedNumericDownloadStage>,
    numeric_owner: Option<super::super::allocation_state::NumericDownloadStageHandle>,
    invoked: bool,
}

impl MapCallback {
    fn complete(&mut self, result: std::result::Result<(), wgpu::BufferAsyncError>) {
        self.invoked = true;
        if let Some(owner) = self.numeric_owner.take() {
            if result.is_err() {
                owner.map_failed();
            } else {
                owner.map_complete();
            }
        }
        // The waiter may run as soon as complete() wakes it. Retire this
        // callback's physical staging lease before exposing completion; the
        // awaiting path retires its own staging lease during cleanup.
        drop(self.numeric_stage.take());
        self.completion
            .state
            .get()
            .complete(result.map_err(anyhow::Error::from));
    }
}

impl Drop for MapCallback {
    fn drop(&mut self) {
        if !self.invoked {
            if let Some(owner) = self.numeric_owner.take() {
                owner.map_failed();
            }
            self.completion
                .state
                .get()
                .complete(Err(anyhow!("map_async callback dropped")));
        }
    }
}

pub(crate) trait ReadbackResourceOwner: Borrow<wgpu::Buffer> {
    fn readback_buffer(&self) -> Option<&runmat_accelerate_api::WgpuNumericReadbackBuffer> {
        None
    }
    fn map_async(
        &self,
        access: &WgpuDeviceAccess,
        callback: impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>)
            + wgpu::WasmNotSend
            + 'static,
    ) -> Result<()> {
        if let Some(buffer) = self.readback_buffer() {
            buffer.map_async(callback)?;
            return Ok(());
        }
        access.with_shared(|_| {
            self.borrow()
                .slice(..)
                .map_async(wgpu::MapMode::Read, callback);
        })?;
        Ok(())
    }
    fn read_mapped<T>(
        &self,
        access: &WgpuDeviceAccess,
        decode: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        if let Some(buffer) = self.readback_buffer() {
            self.validate_scope()?;
            let bytes = buffer.mapped_range()?;
            // Decoding is ordinary host work, outside fixed transfer authority.
            return decode(&bytes);
        }
        access.with_shared(|_| {
            self.validate_scope()?;
            let bytes = self.borrow().slice(..).get_mapped_range();
            decode(&bytes)
        })?
    }
    fn cleanup_failed(&self) {}
    fn validate_scope(&self) -> Result<()> {
        Ok(())
    }
    fn buffer_owner(&self) -> Option<&runmat_accelerate_api::WgpuBufferOwner> {
        None
    }
    fn resource_ticket(
        &self,
    ) -> Option<runmat_accelerate_api::observation::ProviderResourceTicket> {
        None
    }
    fn numeric_callback_owner(
        &self,
    ) -> Option<super::super::allocation_state::NumericDownloadStageHandle> {
        None
    }
}

impl ReadbackResourceOwner for wgpu::Buffer {}

impl ReadbackResourceOwner for std::sync::Arc<wgpu::Buffer> {}
impl ReadbackResourceOwner for runmat_accelerate_api::WgpuBufferOwner {
    fn buffer_owner(&self) -> Option<&runmat_accelerate_api::WgpuBufferOwner> {
        Some(self)
    }
    fn validate_scope(&self) -> Result<()> {
        self.validate_creation_scope()?;
        Ok(())
    }
}

impl Borrow<wgpu::Buffer> for super::super::allocation_state::NumericDownloadStageHandle {
    fn borrow(&self) -> &wgpu::Buffer {
        self.staging_buffer()
            .expect("mapped stage owns physical buffer")
    }
}
impl ReadbackResourceOwner for super::super::allocation_state::NumericDownloadStageHandle {
    fn readback_buffer(&self) -> Option<&runmat_accelerate_api::WgpuNumericReadbackBuffer> {
        super::super::allocation_state::NumericDownloadStageHandle::readback_buffer(self)
    }
    fn validate_scope(&self) -> Result<()> {
        super::super::allocation_state::NumericDownloadStageHandle::validate_scope(self)
    }
    fn numeric_callback_owner(
        &self,
    ) -> Option<super::super::allocation_state::NumericDownloadStageHandle> {
        Some(self.callback_owner())
    }
}

pub(super) struct TrackedReadback<B> {
    // Retire the original resource/ticket before its staging fact/claim.
    buffer: B,
    stage: Option<super::super::allocation_state::NumericDownloadStageHandle>,
}
impl<B: ReadbackResourceOwner> Borrow<wgpu::Buffer> for TrackedReadback<B> {
    fn borrow(&self) -> &wgpu::Buffer {
        self.buffer.borrow()
    }
}
impl<B: ReadbackResourceOwner> ReadbackResourceOwner for TrackedReadback<B> {
    fn map_async(
        &self,
        access: &WgpuDeviceAccess,
        callback: impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>)
            + wgpu::WasmNotSend
            + 'static,
    ) -> Result<()> {
        self.buffer.map_async(access, callback)
    }
    fn read_mapped<T>(
        &self,
        access: &WgpuDeviceAccess,
        decode: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<T> {
        self.buffer.read_mapped(access, decode)
    }
    fn cleanup_failed(&self) {
        self.buffer.cleanup_failed();
    }
    fn validate_scope(&self) -> Result<()> {
        self.buffer.validate_scope()?;
        if let Some(stage) = &self.stage {
            stage.validate_scope()?;
        }
        Ok(())
    }
    fn numeric_callback_owner(
        &self,
    ) -> Option<super::super::allocation_state::NumericDownloadStageHandle> {
        self.stage
            .as_ref()
            .map(|stage| stage.callback_owner())
            .or_else(|| self.buffer.numeric_callback_owner())
    }
}
pub(super) fn track<B: ReadbackResourceOwner>(
    allocations: &super::super::owner_shared::OwnerShared<
        super::super::allocation_state::AllocationState,
    >,
    buffer: B,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
) -> Result<TrackedReadback<B>> {
    buffer.validate_scope()?;
    let stage = if let Some(existing) = buffer.numeric_callback_owner() {
        // Selected/ordinary numeric download already registered its original
        // owner. Do not allocate or charge a second staging record.
        drop(existing);
        None
    } else if let Some(owner) = buffer.buffer_owner() {
        Some(
            super::super::allocation_state::AllocationState::own_numeric_download_stage(
                allocations,
                Some(owner.clone().into()),
                owner.size(),
                1,
                claim,
                buffer.resource_ticket(),
                None,
            )?,
        )
    } else {
        // Deliberate raw SDK unit tests may have no Runtime at all. A scoped
        // production map must carry its actual creation stamp, never annotate
        // an arbitrary raw buffer with the registering task's newer epoch.
        runmat_accelerate_api::validate_device_allocation_scope(None)?;
        None
    };
    Ok(TrackedReadback { buffer, stage })
}

#[cfg(target_arch = "wasm32")]
impl ReadbackResourceOwner for std::rc::Rc<wgpu::Buffer> {}

#[cfg(test)]
#[path = "readback/tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "readback/browser_tests.rs"]
mod browser_tests;

#[derive(Debug)]
enum CleanupError {
    Access(WgpuAccessError),
    UnmapPanicked,
}
impl fmt::Display for CleanupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Access(error) => write!(f, "WGPU staging cleanup access failed: {error}"),
            Self::UnmapPanicked => f.write_str(
                "WGPU staging unmap panicked; completion is uncertain; no retry attempted",
            ),
        }
    }
}
impl Error for CleanupError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Access(error) => Some(error),
            Self::UnmapPanicked => None,
        }
    }
}

/// Preserve both independent failures. The primary remains the ordinary error
/// source; the typed cleanup failure remains inspectable on this local carrier.
#[derive(Debug)]
struct CompletionError {
    primary: anyhow::Error,
    cleanup: CleanupError,
}
impl fmt::Display for CompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}; additionally, {}", self.primary, self.cleanup)
    }
}
impl Error for CompletionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.primary.as_ref())
    }
}

fn complete<T>(result: Result<T>, cleanup: std::result::Result<(), CleanupError>) -> Result<T> {
    match (result, cleanup) {
        (result, Ok(())) => result,
        (Ok(_), Err(error)) => Err(error.into()),
        (Err(primary), Err(cleanup)) => Err(CompletionError { primary, cleanup }.into()),
    }
}

struct Readback<B: ReadbackResourceOwner> {
    access: WgpuDeviceAccess,
    buffer: Option<B>,
    map_registered: bool,
}

impl<B: ReadbackResourceOwner> Readback<B> {
    fn cleanup(&mut self) -> std::result::Result<(), CleanupError> {
        let Some(buffer) = self.buffer.take() else {
            return Ok(());
        };
        if !std::mem::replace(&mut self.map_registered, false) {
            // Unpolled futures own an ordinary buffer, not an active mapping.
            // Raw Buffer Drop cannot feed the pinned backends' error scopes.
            drop(buffer);
            return Ok(());
        }
        // Disarm before driver access, including denied access or unwind. The
        // ordinary resource drop below is safe outside an emission and is not
        // reported as successful unmap when the explicit teardown failed.
        let result = catch_allocation_cleanup_unwind(|| {
            self.access
                .with_cleanup(|cleanup| cleanup.unmap_buffer(buffer.borrow()))
        })
        .map_err(|_| CleanupError::UnmapPanicked)
        .and_then(|result| result.map_err(CleanupError::Access));
        if result.is_err() {
            buffer.cleanup_failed();
            if let Some(recorder) =
                runmat_accelerate_api::observation::current_provider_observation()
            {
                recorder.fail(
                    runmat_accelerate_api::observation::ProviderObservationFailure::ProviderFailed,
                );
            }
        }
        drop(buffer);
        result
    }
}

impl<B: ReadbackResourceOwner> Drop for Readback<B> {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            log::error!("WGPU staging cancellation/unwind cleanup failed: {error}");
        }
    }
}

#[cfg(any(not(target_arch = "wasm32"), test))]
pub(super) fn read<'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    size_bytes: u64,
    context: &'a str,
    #[cfg(not(target_arch = "wasm32"))] progress: Option<Arc<NativeObservationProgress>>,
) -> impl std::future::Future<Output = Result<Vec<u8>>> + 'a {
    read_with(
        access,
        buffer,
        context,
        move |data| {
            let size = usize::try_from(size_bytes)
                .map_err(|_| anyhow!("{context}: readback size overflow"))?;
            if data.len() != size {
                return Err(anyhow!("{context}: mapped readback size mismatch"));
            }
            Ok(data.to_vec())
        },
        #[cfg(not(target_arch = "wasm32"))]
        progress,
    )
}

pub(super) fn read_with<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    #[cfg(not(target_arch = "wasm32"))] progress: Option<Arc<NativeObservationProgress>>,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    read_with_optional_claim(
        access,
        buffer,
        context,
        decode,
        None,
        None,
        #[cfg(not(target_arch = "wasm32"))]
        progress,
    )
}

/// The selected finite route supplies the retained control claim before its
/// first provider effect. A dropped future cannot refund the callback's Arc.
pub(super) fn read_with_admitted<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    claim: Arc<dyn RetainedHostAllocationClaim>,
    #[cfg(not(target_arch = "wasm32"))] progress: Option<Arc<NativeObservationProgress>>,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    read_with_optional_claim(
        access,
        buffer,
        context,
        decode,
        Some(claim),
        None,
        #[cfg(not(target_arch = "wasm32"))]
        progress,
    )
}

pub(super) fn read_with_admitted_stage<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    claim: Arc<dyn RetainedHostAllocationClaim>,
    callback_stage: runmat_accelerate_api::PreparedNumericDownloadStage,
    #[cfg(not(target_arch = "wasm32"))] progress: Option<Arc<NativeObservationProgress>>,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    read_with_optional_claim(
        access,
        buffer,
        context,
        decode,
        Some(claim),
        Some(callback_stage),
        #[cfg(not(target_arch = "wasm32"))]
        progress,
    )
}

fn read_with_optional_claim<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    callback_stage: Option<runmat_accelerate_api::PreparedNumericDownloadStage>,
    #[cfg(not(target_arch = "wasm32"))] progress: Option<Arc<NativeObservationProgress>>,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    // Establish staging ownership before construction, including unpolled
    // teardown. The one callback state persists independently after polling.
    let mut staging = Readback {
        access,
        buffer: Some(buffer),
        map_registered: false,
    };
    async move {
        let result = async {
            // Reserve before registering a map callback. No emission permit
            // survives commit or the asynchronous wait on the device worker.
            #[cfg(not(target_arch = "wasm32"))]
            let reservation = progress
                .as_ref()
                .ok_or(WgpuObservationProgressError::Stopped)?
                .reserve_slot()?;
            let completion = FundedMapCompletion {
                state: ExactShared::try_new(MapCompletion {
                    state: Mutex::new(MapCompletionState {
                        #[cfg(not(target_arch = "wasm32"))]
                        progress: MapProgress::Pending,
                        ..MapCompletionState::default()
                    }),
                })
                .map_err(|_| anyhow!("map completion control allocation failed"))?,
                _claim: claim,
            };
            let mut callback = MapCallback {
                completion: completion.clone(),
                numeric_stage: callback_stage,
                numeric_owner: staging
                    .buffer
                    .as_ref()
                    .and_then(ReadbackResourceOwner::numeric_callback_owner),
                invoked: false,
            };
            staging
                .buffer
                .as_ref()
                .expect("readback owns staging")
                .validate_scope()?;
            if let Some(owner) = &callback.numeric_owner {
                owner.map_pending();
            }
            staging.map_registered = true;
            staging
                .buffer
                .as_ref()
                .expect("readback owns staging")
                .map_async(&staging.access, move |result| callback.complete(result))?;
            #[cfg(not(target_arch = "wasm32"))]
            {
                reservation.complete_queue_tail_owner(completion.clone().progress_callback())?;
            }
            completion
                .state
                .get()
                .wait()
                .await
                .map_err(|source| source.context(context.to_owned()))?;
            let output = staging
                .buffer
                .as_ref()
                .expect("readback owns staging")
                .read_mapped(&staging.access, decode)?;
            Ok(output)
        }
        .await;
        complete(result, staging.cleanup())
    }
}
