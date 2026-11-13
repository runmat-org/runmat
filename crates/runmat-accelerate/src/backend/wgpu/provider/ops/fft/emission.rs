use super::*;
use crate::backend::wgpu::dispatch::observation::PendingSubmissions;
#[cfg(not(target_arch = "wasm32"))]
use crate::backend::wgpu::dispatch::observation::{ScopeConfirmation, ScopedRetirement};
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
        prepare: impl for<'p> FnOnce(&WgpuEmission<'p>) -> Result<FftPreparedDispatch<'a>>,
    ) -> ProviderExecutionResult<()> {
        self.fft_dispatch_with_debug(Self::fft_debug_enabled(), label, prepare)
            .await
    }

    async fn fft_dispatch_with_debug<'a>(
        &'a self,
        debug: bool,
        label: &str,
        prepare: impl for<'p> FnOnce(&WgpuEmission<'p>) -> Result<FftPreparedDispatch<'a>>,
    ) -> ProviderExecutionResult<()> {
        if !debug {
            return Ok(self.with_emission(|emission, pending| {
                let FftPreparedDispatch {
                    pipeline,
                    bind_group,
                    workgroups,
                    resources,
                } = prepare(emission)?;
                pending.retain_submission_resources(resources.into_retention())?;
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

        let FftPreparedDispatch {
            pipeline,
            bind_group,
            workgroups,
            resources,
        } = self.with_emission(|emission, _| prepare(emission))??;
        // Pre-reserve before the synchronous emission-risk interval; retirement
        // must no longer depend on polling/dropping the logical scope future.
        #[cfg(not(target_arch = "wasm32"))]
        let retirement = ScopedRetirement::prepare()?;
        #[cfg(not(target_arch = "wasm32"))]
        let confirmation = ScopeConfirmation(retirement.clone());
        let receipt = {
            let pending = PendingSubmissions::new(self.access.clone()).with_profiling(
                self.timestamp_capability,
                self.observation_budget.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                self.observation_progress
                    .clone()
                    .map(|progress| progress as _),
            );
            #[cfg(not(target_arch = "wasm32"))]
            let pending = pending.with_scoped_retirement();
            let mut pending = pending;
            pending.retain_submission_resources(resources.into_retention())?;
            #[cfg(not(target_arch = "wasm32"))]
            let progress = self.observation_progress.as_ref().ok_or_else(|| {
                ProviderExecutionError::from(anyhow::anyhow!(
                    "scoped FFT retirement progress unavailable"
                ))
            })?;
            #[cfg(not(target_arch = "wasm32"))]
            let mut gate = progress
                .scoped_emission_guard(pending, retirement.clone())
                .map_err(ProviderExecutionError::from_source)?;
            let receipt = self
                .access
                .with_profiled_scoped(&self.emission_authority, |emission| {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        gate.with_pending(|pending| {
                            crate::backend::wgpu::dispatch::fft::run(
                                emission,
                                pending,
                                pipeline,
                                &bind_group,
                                workgroups,
                                label,
                            )
                        })
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        crate::backend::wgpu::dispatch::fft::run(
                            emission,
                            &mut pending,
                            pipeline,
                            &bind_group,
                            workgroups,
                            label,
                        )
                    }
                })
                .map_err(ProviderExecutionError::from_source)?;
            // Commit physical ownership BEFORE the first await. Neither scope
            // rejection nor cancellation may wait on an unauthenticated index.
            #[cfg(not(target_arch = "wasm32"))]
            {
                // Drop publishes the already-owned fixed slot and merely wakes
                // the existing worker; channel errors cannot drop physical work.
                drop(gate);
            }
            #[cfg(target_arch = "wasm32")]
            {
                // Browser has no indexed wait. Preserve its existing private
                // map/fence callback ownership, including scope cancellation.
                pending.commit();
            }
            receipt
        };
        match receipt.resolve().await {
            Ok(Ok(())) => {
                #[cfg(not(target_arch = "wasm32"))]
                confirmation.0.confirm();
                Ok(())
            }
            Ok(Err(error)) => {
                #[cfg(not(target_arch = "wasm32"))]
                confirmation.0.reject();
                Err(error.into())
            }
            Err(error) => {
                #[cfg(not(target_arch = "wasm32"))]
                confirmation.0.reject();
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
