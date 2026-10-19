//! Public refresh reuses the existing calibrated templates and cleanup owner.
use super::*;

pub async fn refresh_auto_offload_calibration(
    commit: bool,
) -> ProviderExecutionResult<AutoOffloadCalibrationOutcome> {
    let provider = runmat_accelerate_api::provider()
        .ok_or_else(|| anyhow!("no acceleration provider registered"))?;
    let info = provider.device_info_struct();
    let (mut cfg, _) = load_cached_thresholds_async(&info, provider.precision())
        .await
        .unwrap_or_else(|| (ThresholdConfig::default(), String::new()));
    let before = cfg.clone();
    auto_calibrate(provider, &mut cfg).await?;
    let delta = compute_delta(&before, &cfg);
    let persisted_to = if commit {
        Some(
            persist_thresholds_async(&info, &cfg, provider.precision())
                .await
                .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?,
        )
    } else {
        None
    };
    if commit {
        if let Some(state) = AUTO_STATE.get() {
            if let Ok(mut state) = state.lock() {
                state.previous_thresholds = Some(before.clone());
                state.calibration_delta = Some(delta.clone());
                state.thresholds = cfg.clone();
                state.base_source = ThresholdBase::Calibrated;
                state.cache_path = persisted_to.clone();
            }
        }
    }
    Ok(AutoOffloadCalibrationOutcome {
        schema_version: CALIBRATION_VERSION,
        precision: provider.precision(),
        runs: None, // The existing owner does not export a sample census.
        source: CalibrationSource::AutomaticTemplates,
        before: threshold_snapshot(&before),
        after: threshold_snapshot(&cfg),
        delta: Some(delta),
        persisted_to,
        provider: Some(cached_provider_info(&info)),
        commit,
    })
}
