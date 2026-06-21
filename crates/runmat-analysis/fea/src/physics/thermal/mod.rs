use runmat_analysis_core::AnalysisModel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalConstitutiveStats {
    pub conductivity_mean: f64,
    pub heat_capacity_mean: f64,
    pub density_mean: f64,
    pub conductivity_spread_ratio: f64,
    pub heat_capacity_spread_ratio: f64,
    pub diffusivity_estimate: f64,
    pub response_rate: f64,
}

pub fn constitutive_stats(model: &AnalysisModel) -> ThermalConstitutiveStats {
    let mut conductivities = Vec::new();
    let mut heat_capacities = Vec::new();
    for material in &model.materials {
        conductivities.push(material.thermal.conductivity_w_per_mk.max(1.0e-6));
        heat_capacities.push(material.thermal.specific_heat_j_per_kgk.max(1.0));
    }
    let conductivity_mean = if conductivities.is_empty() {
        1.0
    } else {
        conductivities.iter().sum::<f64>() / conductivities.len() as f64
    };
    let heat_capacity_mean = if heat_capacities.is_empty() {
        1.0
    } else {
        heat_capacities.iter().sum::<f64>() / heat_capacities.len() as f64
    };
    let density_mean = 7_800.0;
    let conductivity_spread_ratio = if conductivities.is_empty() {
        1.0
    } else {
        let min = conductivities
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(1.0);
        let max = conductivities
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(1.0);
        (max / min.max(1.0e-9)).clamp(1.0, 32.0)
    };
    let heat_capacity_spread_ratio = if heat_capacities.is_empty() {
        1.0
    } else {
        let min = heat_capacities
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap_or(1.0);
        let max = heat_capacities
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(1.0);
        (max / min.max(1.0e-9)).clamp(1.0, 32.0)
    };
    let diffusivity_estimate = conductivity_mean / (density_mean * heat_capacity_mean).max(1.0e-9);
    let response_rate = (diffusivity_estimate * 5.0e6).clamp(0.02, 2.0);

    ThermalConstitutiveStats {
        conductivity_mean,
        heat_capacity_mean,
        density_mean,
        conductivity_spread_ratio,
        heat_capacity_spread_ratio,
        diffusivity_estimate,
        response_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    #[test]
    fn constitutive_stats_are_finite_for_fixture_model() {
        let model = fixture_model(FixtureId::ThermoRampSmooth);
        let stats = constitutive_stats(&model);
        assert!(stats.conductivity_mean.is_finite());
        assert!(stats.heat_capacity_mean.is_finite());
        assert!(stats.diffusivity_estimate.is_finite());
        assert!(stats.response_rate >= 0.02 && stats.response_rate <= 2.0);
    }
}
