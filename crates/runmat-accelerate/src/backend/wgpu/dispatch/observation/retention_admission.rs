//! Recorder-independent exact physical resource ownership admission.
use super::*;

impl PendingSubmissions {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_scoped_retirement(mut self) -> Self {
        self.scoped_retirement = true;
        self
    }

    /// Recorder-independent ownership admission, before encoding or submission.
    /// Reuses the canonical observed fence/direct completion; native diagnostic
    /// scopes instead keep this owner in their already-prearmed physical slot.
    pub(crate) fn retain_submission_resources(
        &mut self,
        retention: SubmittedAllocationRetention,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.armed_allocation_retention.is_none()
                && self.prepared_allocation_admission.is_none(),
            "submission resource ownership already armed"
        );
        let admission = match current_provider_observation() {
            Some(recorder) => {
                #[cfg(not(target_arch = "wasm32"))]
                let admission = if self.scoped_retirement {
                    admission::prepare_scoped(recorder)
                } else {
                    prepare(recorder)
                };
                #[cfg(target_arch = "wasm32")]
                let admission = prepare(recorder);
                anyhow::ensure!(
                    matches!(admission, ProviderAdmission::Prepared(_)),
                    "submission resource completion admission unavailable"
                );
                admission
            }
            None => {
                #[cfg(not(target_arch = "wasm32"))]
                if !self.scoped_retirement {
                    self.prepared_direct_progress = Some(
                        self.progress_port
                            .as_ref()
                            .ok_or_else(|| {
                                anyhow::anyhow!("submission resource progress unavailable")
                            })?
                            .reserve()
                            .map_err(anyhow::Error::new)?,
                    );
                }
                #[cfg(target_arch = "wasm32")]
                {
                    self.prepared_direct_fence = Some(
                        self.budget
                            .as_ref()
                            .and_then(|budget| budget.try_reserve_wgpu_tail(FENCE_REQUESTED_BYTES))
                            .ok_or_else(|| {
                                anyhow::anyhow!("submission resource fence budget unavailable")
                            })?,
                    );
                }
                ProviderAdmission::Absent
            }
        };
        self.prepared_allocation_admission = Some(admission);
        self.armed_allocation_retention = Some(retention);
        Ok(())
    }
}
