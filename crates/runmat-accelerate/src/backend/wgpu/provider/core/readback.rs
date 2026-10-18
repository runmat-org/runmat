//! Staging ownership across asynchronous mapping. No emission permit survives
//! a poll/await; cancellation performs only a fresh restricted cleanup phase.
use anyhow::{anyhow, Result};
use runmat_accelerate_api::allocation::catch_allocation_cleanup_unwind;
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

#[derive(Default)]
struct MapCompletionState {
    result: Option<Result<()>>,
    waker: Option<Waker>,
}

impl MapCompletion {
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
                if let Some(result) = state.result.take() {
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
    invoked: bool,
}

impl MapCallback {
    fn complete(&mut self, result: std::result::Result<(), wgpu::BufferAsyncError>) {
        self.invoked = true;
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
            self.completion
                .state
                .get()
                .complete(Err(anyhow!("map_async callback dropped")));
        }
    }
}

pub(crate) trait ReadbackResourceOwner: Borrow<wgpu::Buffer> {
    fn cleanup_failed(&self) {}
}

impl ReadbackResourceOwner for wgpu::Buffer {}

impl ReadbackResourceOwner for std::sync::Arc<wgpu::Buffer> {}

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
pub(super) fn read(
    access: WgpuDeviceAccess,
    buffer: wgpu::Buffer,
    size_bytes: u64,
    context: &str,
) -> impl std::future::Future<Output = Result<Vec<u8>>> + '_ {
    read_with(access, buffer, context, move |data| {
        let size = usize::try_from(size_bytes)
            .map_err(|_| anyhow!("{context}: readback size overflow"))?;
        if data.len() != size {
            return Err(anyhow!("{context}: mapped readback size mismatch"));
        }
        Ok(data.to_vec())
    })
}

pub(super) fn read_with<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    read_with_optional_claim(access, buffer, context, decode, None, None)
}

/// The selected finite route supplies the retained control claim before its
/// first provider effect. A dropped future cannot refund the callback's Arc.
pub(super) fn read_with_admitted<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    claim: Arc<dyn RetainedHostAllocationClaim>,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    read_with_optional_claim(access, buffer, context, decode, Some(claim), None)
}

pub(super) fn read_with_admitted_stage<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    claim: Arc<dyn RetainedHostAllocationClaim>,
    callback_stage: runmat_accelerate_api::PreparedNumericDownloadStage,
) -> impl std::future::Future<Output = Result<T>> + 'a {
    read_with_optional_claim(
        access,
        buffer,
        context,
        decode,
        Some(claim),
        Some(callback_stage),
    )
}

fn read_with_optional_claim<'a, T: 'a>(
    access: WgpuDeviceAccess,
    buffer: impl ReadbackResourceOwner + 'a,
    context: &'a str,
    decode: impl FnOnce(&[u8]) -> Result<T> + 'a,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    callback_stage: Option<runmat_accelerate_api::PreparedNumericDownloadStage>,
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
            let completion = FundedMapCompletion {
                state: ExactShared::try_new(MapCompletion {
                    state: Mutex::new(MapCompletionState::default()),
                })
                .map_err(|_| anyhow!("map completion control allocation failed"))?,
                _claim: claim,
            };
            let mut callback = MapCallback {
                completion: completion.clone(),
                numeric_stage: callback_stage,
                invoked: false,
            };
            staging.access.with_shared(|_| {
                staging.map_registered = true;
                staging
                    .buffer
                    .as_ref()
                    .expect("readback owns staging")
                    .borrow()
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        callback.complete(result);
                    });
            })?;
            #[cfg(not(target_arch = "wasm32"))]
            staging.access.poll(wgpu::Maintain::Wait);
            completion
                .state
                .get()
                .wait()
                .await
                .map_err(|source| anyhow!("{context}: {source}"))?;
            let output = staging.access.with_shared(|_| {
                let data = staging
                    .buffer
                    .as_ref()
                    .expect("readback owns staging")
                    .borrow()
                    .slice(..)
                    .get_mapped_range();
                let output = decode(&data);
                drop(data);
                output
            })??;
            Ok(output)
        }
        .await;
        complete(result, staging.cleanup())
    }
}
