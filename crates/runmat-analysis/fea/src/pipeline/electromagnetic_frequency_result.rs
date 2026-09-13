use crate::contracts::FeaElectromagneticRunResult;

use super::frequency_plan::FrequencyResponsePlan;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElectromagneticFrequencySummary {
    pub peak_flux_density: Vec<f64>,
    pub solve_quality: Vec<f64>,
    pub resonance_peak_frequency_hz: Option<f64>,
    pub resonance_peak_flux_density: Option<f64>,
    pub resonance_bandwidth_hz: Option<f64>,
    pub resonance_quality_factor: Option<f64>,
    pub resonance_flux_gain: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeaElectromagneticFrequencyResponse {
    pub(super) plan: FrequencyResponsePlan,
    pub(super) runs: Vec<FeaElectromagneticRunResult>,
    pub(super) summary: ElectromagneticFrequencySummary,
}

impl FeaElectromagneticFrequencyResponse {
    pub const fn plan(&self) -> &FrequencyResponsePlan {
        &self.plan
    }

    pub fn runs(&self) -> &[FeaElectromagneticRunResult] {
        &self.runs
    }

    pub const fn summary(&self) -> &ElectromagneticFrequencySummary {
        &self.summary
    }

    pub fn primary(&self) -> &FeaElectromagneticRunResult {
        &self.runs[self.plan.reference_index()]
    }

    pub fn into_primary(self) -> FeaElectromagneticRunResult {
        self.runs
            .into_iter()
            .nth(self.plan.reference_index())
            .expect("validated frequency response contains its reference run")
    }
}
