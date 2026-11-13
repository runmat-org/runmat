//! Known-layout controls of canonical WGPU completion owners.
#[cfg(not(target_arch = "wasm32"))]
use super::{
    completion::{DirectCompletion, FenceCompletion},
    completion_signal,
};

pub(crate) fn typed_submission_dispatch_host_control_upper_bytes(observed: bool) -> Option<u64> {
    #[cfg(target_arch = "wasm32")]
    {
        if observed {
            super::browser_completion::observed_host_upper_bytes()
        } else {
            super::browser_completion::direct_host_upper_bytes()
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let payload = if observed {
            std::mem::size_of::<FenceCompletion>()
        } else {
            std::mem::size_of::<DirectCompletion>()
        };
        let bytes = u64::try_from(payload)
            .ok()?
            .checked_add(completion_signal::channel_host_upper_bytes::<bool>()?)?;
        if observed {
            bytes.checked_add(completion_signal::channel_host_upper_bytes::<
                Option<runmat_accelerate_api::observation::ProviderCompletedSubmission>,
            >()?)
        } else {
            Some(bytes)
        }
    }
}

pub(crate) fn observed_submission_dispatch_host_control_upper_bytes() -> Option<u64> {
    let bytes = typed_submission_dispatch_host_control_upper_bytes(true)?;
    #[cfg(not(target_arch = "wasm32"))]
    let bytes =
        bytes.checked_add(super::PendingSubmissions::scoped_retirement_host_upper_bytes()?)?;
    Some(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn observed_completion_callback_fits_bound<T>(callback: &T) -> bool {
    std::mem::size_of_val(callback) <= std::mem::size_of::<FenceCompletion>()
        && std::mem::align_of_val(callback) <= std::mem::align_of::<FenceCompletion>()
}
