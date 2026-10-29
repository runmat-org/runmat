//! One encoder's provider and command admission, captured before optional GPU
//! resources are allocated. Later poll-local selection cannot replace it.
use runmat_accelerate_api::observation::{
    ProviderObservationFailure, ProviderOperationRecorder, ProviderSubmission,
};
#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::observation::{
    WgpuObservationProgressError, WgpuObservationProgressReservation,
};
use runmat_execution::profiling::CommandProfilingReceipt;

pub(super) struct PreparedObservation {
    pub(super) token: ProviderSubmission,
    pub(super) recorder: ProviderOperationRecorder,
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) progress: Option<Box<dyn WgpuObservationProgressReservation>>,
}

pub(super) enum ProviderAdmission {
    Absent,
    Prepared(PreparedObservation),
    Failed(ProviderOperationRecorder),
}

impl ProviderAdmission {
    pub(super) fn recorder(&self) -> Option<&ProviderOperationRecorder> {
        match self {
            Self::Absent => None,
            Self::Prepared(prepared) => Some(&prepared.recorder),
            Self::Failed(recorder) => Some(recorder),
        }
    }
}

pub(super) fn prepare(recorder: ProviderOperationRecorder) -> ProviderAdmission {
    #[cfg(not(target_arch = "wasm32"))]
    let progress = match recorder
        .wgpu_progress()
        .ok_or(WgpuObservationProgressError::Stopped)
        .and_then(|driver| driver.reserve())
    {
        Ok(reservation) => reservation,
        Err(reason) => {
            log::debug!("WGPU provider observation progress reservation failed: {reason:?}");
            recorder.fail(match reason {
                WgpuObservationProgressError::Capacity => ProviderObservationFailure::ResourceLimit,
                WgpuObservationProgressError::Stopped => ProviderObservationFailure::ProviderFailed,
            });
            return ProviderAdmission::Failed(recorder);
        }
    };
    let token = match recorder.submission() {
        Ok(token) => token,
        Err(_) => {
            log::debug!("WGPU provider observation submission admission failed");
            return ProviderAdmission::Failed(recorder);
        }
    };
    ProviderAdmission::Prepared(PreparedObservation {
        token,
        recorder,
        #[cfg(not(target_arch = "wasm32"))]
        progress: Some(progress),
    })
}

/// Native diagnostic scopes already own one bounded physical progress slot.
/// Their observation token shares that retirement proof, never another index.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn prepare_scoped(recorder: ProviderOperationRecorder) -> ProviderAdmission {
    match recorder.submission() {
        Ok(token) => ProviderAdmission::Prepared(PreparedObservation {
            token,
            recorder,
            progress: None,
        }),
        Err(_) => ProviderAdmission::Failed(recorder),
    }
}

fn same_owner<T>(left: Option<&T>, right: Option<&T>, same: impl Fn(&T, &T) -> bool) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => same(left, right),
        _ => false,
    }
}

pub(super) fn fail_mismatched_admission(
    admission: &ProviderAdmission,
    current_recorder: Option<&ProviderOperationRecorder>,
    captured_command: Option<&CommandProfilingReceipt>,
    current_command: Option<&CommandProfilingReceipt>,
) -> bool {
    let matched = same_owner(admission.recorder(), current_recorder, |a, b| {
        a.same_observation(b)
    }) && same_owner(captured_command, current_command, |a, b| a.same_command(b));
    if !matched {
        if let Some(recorder) = admission.recorder() {
            recorder.fail(ProviderObservationFailure::ExecutionUnproven);
        }
        if let Some(recorder) = current_recorder {
            recorder.fail(ProviderObservationFailure::ExecutionUnproven);
        }
        if let Some(command) = captured_command {
            command.mark_unproven_device_route();
        }
        if let Some(command) = current_command {
            command.mark_unproven_device_route();
        }
    }
    matched
}
