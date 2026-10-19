use super::*;
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
use runmat_accelerate_api::observation::{
    current_provider_observation, ProviderObservationFailure,
};
use runmat_accelerate_api::{
    ProviderExecutionError, ProviderExecutionResult, WgpuEmission, WgpuScopeErrors,
};

/// Keep the dispatch identity alongside the original worker-local driver errors.
/// In a browser those errors cannot enter an `anyhow::Error` (Send + Sync).
#[derive(Debug)]
struct FftScopedEmissionError {
    label: String,
    source: WgpuScopeErrors,
}

impl std::fmt::Display for FftScopedEmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "FFT dispatch {}: {}", self.label, self.source)
    }
}

impl std::error::Error for FftScopedEmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

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
    ) -> ProviderExecutionResult<()> {
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
    ) -> ProviderExecutionResult<()> {
        if !debug {
            return Ok(self.with_emission(|emission, pending| {
                let (pipeline, bind_group, workgroups) = prepare(emission)?;
                crate::backend::wgpu::dispatch::fft::run(
                    emission,
                    pending,
                    pipeline,
                    &bind_group,
                    workgroups,
                    label,
                )
            })??);
        }

        let (pipeline, bind_group, workgroups) =
            self.with_emission(|emission, _| prepare(emission))??;
        let mut pending = PendingSubmissions::new(self.access.clone());
        let receipt = self
            .access
            .with_profiled_scoped(&self.emission_authority, |emission| {
                crate::backend::wgpu::dispatch::fft::run(
                    emission,
                    &mut pending,
                    pipeline,
                    &bind_group,
                    workgroups,
                    label,
                )
            })
            .map_err(ProviderExecutionError::from_source)?;
        // Submission progress, including legacy exact-index waits on native,
        // must not run under the exclusive scope gate.
        pending.commit();
        match receipt.resolve().await {
            Ok(result) => result.map_err(Into::into),
            Err(error) => {
                if let Some(recorder) = current_provider_observation() {
                    recorder.fail(ProviderObservationFailure::ProviderFailed);
                }
                Err(ProviderExecutionError::from_source(
                    FftScopedEmissionError {
                        label: label.to_owned(),
                        source: error,
                    },
                ))
            }
        }
    }
}

#[cfg(test)]
#[path = "emission_tests.rs"]
mod tests;
