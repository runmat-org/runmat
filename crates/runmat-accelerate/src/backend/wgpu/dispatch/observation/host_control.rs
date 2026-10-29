//! Exact heap-control bounds for one observed WGPU submission.

use super::{completion::FenceCompletion, completion_signal};

pub(crate) fn observed_submission_dispatch_host_control_upper_bytes() -> Option<u64> {
    fn boxed_upper<T>() -> Option<u64> {
        std::mem::size_of::<T>()
            .checked_add(std::mem::align_of::<T>())?
            .try_into()
            .ok()
    }

    let bytes = completion_signal::channel_host_upper_bytes::<
        Option<runmat_accelerate_api::observation::ProviderCompletedSubmission>,
    >()?
    .checked_add(completion_signal::channel_host_upper_bytes::<bool>()?)?
    .checked_add(boxed_upper::<FenceCompletion>()?)?;
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
