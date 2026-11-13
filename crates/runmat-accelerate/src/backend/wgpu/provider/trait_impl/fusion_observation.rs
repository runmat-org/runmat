//! Selected-operation proof belongs to the validated fused kernel hook, not
//! uploads, pipeline preparation, or its caller's queue activity.
use runmat_accelerate_api::observation::{current_provider_observation, ProviderOperationRecorder};
use runmat_accelerate_api::ProviderOperationIdentity;

pub(super) struct ElementwiseKernelObservation {
    recorder: ProviderOperationRecorder,
    checkpoint: u64,
    operation: ProviderOperationIdentity,
}

impl ElementwiseKernelObservation {
    /// Called after validated pipeline preparation and before the actual
    /// kernel. Empty outputs cannot certify a nonempty kernel submission.
    pub(super) fn begin(elements: usize) -> Option<Self> {
        if elements == 0 {
            return None;
        }
        let recorder = current_provider_observation()?;
        let operation = ProviderOperationIdentity::new(crate::placement::fusion_operation_token(
            &crate::fusion::FusionKind::ElementwiseChain,
        ));
        if !recorder.expects_operation(&operation) {
            return None;
        }
        let checkpoint = recorder.submission_checkpoint();
        Some(Self {
            recorder,
            checkpoint,
            operation,
        })
    }

    /// The hook invokes this only after successful validated output creation.
    /// The receipt itself checks the matching device and post-checkpoint work.
    pub(super) fn submitted(self, device_id: u32) {
        self.recorder.confirm_submitted_provider_execution_after(
            device_id,
            &self.operation,
            self.checkpoint,
        );
    }
}
