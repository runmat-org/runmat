//! Provider-owned completion of all earlier ordinary work on one exact queue.
//! Child operation receipts remain independent. Native uses the bounded
//! progress port; browser uses one admitted 8-byte map fence. Neither route
//! reattributes a child's measurements.

use super::*;
use futures::channel::oneshot;
use runmat_accelerate_api::observation::with_provider_observation;
use runmat_execution::profiling::with_command_profiling;

impl WgpuProvider {
    pub(super) async fn observe_vm_region_queue_tail_exec(
        &self,
        device_id: u32,
    ) -> runmat_accelerate_api::ProviderExecutionResult<()> {
        use runmat_accelerate_api::ProviderExecutionError;
        if device_id != self.runtime_device_id {
            return Err(ProviderExecutionError::from(anyhow!(
                "VM queue-tail owner differs from the installed provider"
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
                                label: Some("runmat-vm-region-completion-fence"),
                                size: crate::backend::wgpu::dispatch::observation::FENCE_REQUESTED_BYTES,
                                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }),
                        );
                        let mut encoder = pending.profiled_encoder(
                            emission,
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-vm-region-completion-tail"),
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
            let (send, receive) = oneshot::channel();
            let callback_buffer = buffer.clone();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |status| {
                    // The callback owns both the map buffer and permit until the
                    // driver reports terminal map status. Dropping the awaiting
                    // future cannot unmap early or recycle the capacity.
                    if status.is_err() {
                        permit.poison_budget();
                    }
                    // Successful callbacks hand the permit to the awaiting
                    // owner, retaining the charge through gated unmap. On
                    // cancellation the failed send retires both after this
                    // terminal callback, never at future drop.
                    let _ = send.send((status.is_ok(), callback_buffer, permit));
                });
            let (mapped, retired_buffer, _permit) = receive
                .await
                .map_err(|_| ProviderExecutionError::from(anyhow!("WGPU tail callback dropped")))?;
            self.access
                .with_cleanup(|cleanup| cleanup.unmap_buffer(&retired_buffer))
                .map_err(ProviderExecutionError::from)?;
            if !mapped {
                return Err(ProviderExecutionError::from(anyhow!(
                    "WGPU tail fence map failed"
                )));
            }
            return Ok(());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Reserve progress before the tail submission. Capacity failure only
            // withholds this optional observation; source bytecode is never replayed.
            let reservation = {
                use runmat_accelerate_api::observation::WgpuObservationProgress;
                self.observation_progress
                    .as_ref()
                    .ok_or_else(|| {
                        ProviderExecutionError::from(anyhow!("WGPU progress is absent"))
                    })?
                    .reserve()
                    .map_err(|error| ProviderExecutionError::from(anyhow!("{error:?}")))?
            };
            let submission = with_command_profiling(None, || {
                with_provider_observation(None, || {
                    self.with_emission(|emission, pending| -> Result<wgpu::SubmissionIndex> {
                        let encoder = pending.profiled_encoder(
                            emission,
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("runmat-vm-region-completion-tail"),
                            },
                        );
                        pending.submit(emission, encoder)
                    })
                })
            })
            .map_err(ProviderExecutionError::from)?
            .map_err(ProviderExecutionError::from)?;

            let (send, receive) = oneshot::channel();
            reservation
                .submit(
                    submission,
                    Box::new(move |result| {
                        let _ = send.send(result);
                    }),
                )
                .map_err(|error| ProviderExecutionError::from(anyhow!("{error:?}")))?;
            receive
                .await
                .map_err(|_| ProviderExecutionError::from(anyhow!("WGPU tail progress dropped")))?
                .map_err(|error| ProviderExecutionError::from(anyhow!("{error:?}")))?;
            Ok(())
        }
    }
}
