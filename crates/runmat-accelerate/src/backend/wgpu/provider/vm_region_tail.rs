//! Provider-owned completion of all earlier ordinary work on one exact queue.
//! Child operation receipts remain independent. Native uses the bounded
//! progress port; browser uses one admitted 8-byte map fence. Neither route
//! reattributes a child's measurements.

use super::*;
use runmat_accelerate_api::observation::with_provider_observation;
use runmat_execution::profiling::with_command_profiling;

impl WgpuProvider {
    pub(super) fn region_tail_control_host_bytes(&self) -> Option<u64> {
        let future = u64::try_from(std::mem::size_of_val(
            &self.observe_region_queue_tail_exec(self.runtime_device_id, None),
        ))
        .ok()?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            future
                .checked_add(super::device_control::FundedCompletion::host_bytes()?)?
                .checked_add(super::observation_progress::reservation_host_upper_bytes()?)
        }
        #[cfg(target_arch = "wasm32")]
        {
            future.checked_add(super::core::map_completion_host_upper_bytes()?)
        }
    }
    pub(super) async fn observe_region_queue_tail_exec(
        &self,
        device_id: u32,
        completion_claim: Option<
            std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>,
        >,
    ) -> runmat_accelerate_api::ProviderExecutionResult<()> {
        use runmat_accelerate_api::ProviderExecutionError;
        if device_id != self.runtime_device_id {
            return Err(ProviderExecutionError::from(anyhow!(
                "source region queue-tail owner differs from the installed provider"
            )));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let permit = self
                .observation_budget
                .try_reserve_wgpu_tail(
                    crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
                )
                .ok_or_else(|| {
                    ProviderExecutionError::from(anyhow!("WGPU tail fence unavailable"))
                })?;
            let (buffer, _) = with_command_profiling(None, || {
                with_provider_observation(None, || {
                    self.with_emission(|emission, pending| -> Result<_> {
                        let buffer = runmat_accelerate_api::WgpuBufferOwner::new(
                            emission.device().create_buffer(&wgpu::BufferDescriptor {
                                label: Some("runmat-source-region-completion-fence"),
                                size: crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }),
                        );
                        let mut encoder = pending.profiled_encoder(
                            emission,
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-source-region-completion-tail"),
                            },
                        );
                        encoder.clear_buffer(&buffer, 0, None);
                        let submission = pending.submit(emission, encoder)?;
                        Ok((buffer, submission))
                    })
                })
            })
            .map_err(ProviderExecutionError::from)?
            .map_err(ProviderExecutionError::from)?;
            let stage = super::allocation_state::AllocationState::own_numeric_download_stage(
                &self.allocations,
                Some(buffer.into()),
                crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
                1,
                completion_claim.clone(),
                None,
                Some(permit),
            )
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
            let decode = |_: &[u8]| Ok(());
            match completion_claim {
                Some(claim) => {
                    self.map_readback_with_admitted(
                        stage,
                        "source region queue tail",
                        decode,
                        claim,
                    )
                    .await
                }
                None => {
                    self.map_readback_with(stage, "source region queue tail", decode)
                        .await
                }
            }
            .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Reserve progress before the tail submission. Capacity failure only
            // withholds this optional observation; source operations is never replayed.
            let reservation = {
                use runmat_accelerate_api::observation::WgpuObservationProgress;
                self.observation_progress
                    .as_ref()
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!("WGPU progress is absent"))
                    })?
                    .reserve()
                    .map_err(ProviderExecutionError::from_source)?
            };
            let submission = with_command_profiling(None, || {
                with_provider_observation(None, || {
                    self.with_emission(|emission, pending| -> Result<wgpu::SubmissionIndex> {
                        let encoder = pending.profiled_encoder(
                            emission,
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-source-region-completion-tail"),
                            },
                        );
                        pending.submit(emission, encoder)
                    })
                })
            })
            .map_err(ProviderExecutionError::from)?
            .map_err(ProviderExecutionError::from)?;

            let callback_claim = completion_claim.clone();
            let completion = super::device_control::FundedCompletion::new(completion_claim)
                .map_err(ProviderExecutionError::from)?;
            let callback = completion.clone();
            reservation
                .submit(
                    submission,
                    runmat_accelerate_api::observation::WgpuObservationCompletion::new(
                        Box::new(move |result| {
                            callback.complete(result);
                        }),
                        [callback_claim, None],
                    ),
                )
                .map_err(ProviderExecutionError::from_source)?;
            completion
                .wait()
                .await
                .map_err(ProviderExecutionError::from_source)?;
            Ok(())
        }
    }
}
