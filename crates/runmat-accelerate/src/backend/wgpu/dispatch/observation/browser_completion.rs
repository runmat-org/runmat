//! Browser physical completion uses the SAME callback owner as native.
//! RunMat owns the payload and callback Boxes and funds them through explicit
//! destruction. The existing wasm task only waits for map delivery; its
//! dependency-private scheduler/future storage is a separate documented bound.
use super::{
    completion::{DirectCompletion, FenceCompletion},
    completion_signal,
};
use crate::wgsl::host_buffers::Claim;
use runmat_accelerate_api::observation::{
    ProviderCompletedSubmission, WgpuObservationCompletion, WgpuObservationProgressError,
};

struct Pending<T> {
    // Cancellation/disposal frees the actual payload before its own funding.
    payload: Box<T>,
    claims: [Claim; 2],
}

fn direct_callback(
    payload: Option<Box<DirectCompletion>>,
    mapped: bool,
) -> impl FnOnce(Result<(), WgpuObservationProgressError>) {
    move |status| {
        if let Some(payload) = payload {
            // Move out and deallocate our Box before any outer funding drops.
            let completion = *payload;
            completion.retire(status.is_ok() && mapped, mapped);
        }
    }
}
fn observed_callback(
    payload: Option<Box<FenceCompletion>>,
    completed: Option<ProviderCompletedSubmission>,
    timing_mapped: bool,
) -> impl FnOnce(Result<(), WgpuObservationProgressError>) {
    move |status| {
        if let Some(payload) = payload {
            let completion = *payload;
            completion.cleanup(completed, status.is_ok(), timing_mapped);
        }
    }
}

pub(super) fn direct_host_upper_bytes() -> Option<u64> {
    let callback = direct_callback(None, false);
    std::mem::size_of::<DirectCompletion>()
        .checked_add(std::mem::size_of_val(&callback))?
        .checked_add(usize::try_from(completion_signal::channel_host_upper_bytes::<bool>()?).ok()?)?
        .try_into()
        .ok()
}
pub(super) fn observed_host_upper_bytes() -> Option<u64> {
    let callback = observed_callback(None, None, false);
    std::mem::size_of::<FenceCompletion>()
        .checked_add(std::mem::size_of_val(&callback))?
        .checked_add(
            usize::try_from(completion_signal::channel_host_upper_bytes::<
                Option<ProviderCompletedSubmission>,
            >()?)
            .ok()?,
        )?
        .checked_add(usize::try_from(completion_signal::channel_host_upper_bytes::<bool>()?).ok()?)?
        .try_into()
        .ok()
}

pub(super) fn direct(completion: DirectCompletion, claims: [Claim; 2]) {
    let mut pending = Pending {
        payload: Box::new(completion),
        claims,
    };
    wasm_bindgen_futures::spawn_local(async move {
        let mapped = match pending.payload.result.as_mut() {
            Some(result) => result.receive().await.unwrap_or(false),
            None => false,
        };
        let Pending { payload, claims } = pending;
        WgpuObservationCompletion::new(Box::new(direct_callback(Some(payload), mapped)), claims)
            .finish(Ok(()));
    });
}
pub(super) fn observed(completion: FenceCompletion, claims: [Claim; 2]) {
    let mut pending = Pending {
        payload: Box::new(completion),
        claims,
    };
    wasm_bindgen_futures::spawn_local(async move {
        let completed = pending.payload.result.receive().await.flatten();
        let timing_mapped = match pending.payload.readback_result.as_mut() {
            Some(result) => result.receive().await.unwrap_or(false),
            None => false,
        };
        let Pending { payload, claims } = pending;
        WgpuObservationCompletion::new(
            Box::new(observed_callback(Some(payload), completed, timing_mapped)),
            claims,
        )
        .finish(Ok(()));
    });
}
