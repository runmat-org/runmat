use super::*;
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use runmat_accelerate_api::{
    current_provider_observation, ProviderObservationFailure, WgpuEmission,
};

impl WgpuProvider {
    /// Normal chunks prepare and submit within one shared access interval.
    /// Diagnostic chunks prepare cached resources before the exclusive scope,
    /// then await the already-popped scope receipt without holding the gate.
    pub(super) async fn fft_dispatch_exec<'a>(
        &'a self,
        label: &str,
        prepare: impl for<'p> FnOnce(
            &WgpuEmission<'p>,
        )
            -> Result<(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32)>,
    ) -> Result<()> {
        self.fft_dispatch_with_debug(Self::fft_debug_enabled(), label, prepare)
            .await
    }

    async fn fft_dispatch_with_debug<'a>(
        &'a self,
        debug: bool,
        label: &str,
        prepare: impl for<'p> FnOnce(
            &WgpuEmission<'p>,
        )
            -> Result<(&'a wgpu::ComputePipeline, wgpu::BindGroup, u32)>,
    ) -> Result<()> {
        if !debug {
            return self.with_emission(|emission, pending| {
                let (pipeline, bind_group, workgroups) = prepare(emission)?;
                crate::backend::wgpu::dispatch::fft::run(
                    emission,
                    pending,
                    pipeline,
                    &bind_group,
                    workgroups,
                    label,
                )
            })?;
        }

        let (pipeline, bind_group, workgroups) =
            self.with_emission(|emission, _| prepare(emission))??;
        let mut pending = PendingSubmissions::new(self.access.clone());
        let receipt = self.access.with_scoped(|emission| {
            crate::backend::wgpu::dispatch::fft::run(
                emission,
                &mut pending,
                pipeline,
                &bind_group,
                workgroups,
                label,
            )
        })?;
        // Submission progress, including legacy exact-index waits on native,
        // must not run under the exclusive scope gate.
        pending.commit();
        match receipt.resolve().await {
            Ok(result) => result,
            Err(error) => {
                if let Some(recorder) = current_provider_observation() {
                    recorder.fail(ProviderObservationFailure::ProviderFailed);
                }
                Err(anyhow::Error::new(error).context(format!("FFT dispatch {label}")))
            }
        }
    }
}

#[cfg(test)]
#[path = "emission_tests.rs"]
mod tests;
