pub(crate) mod assembly;
pub(crate) mod history_fields;
pub(crate) mod initial_state;
pub mod qualification;
pub mod radiation;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalConstitutiveStats {
    pub conductivity_mean: f64,
    pub heat_capacity_mean: f64,
    pub density_mean: f64,
    pub conductivity_spread_ratio: f64,
    pub heat_capacity_spread_ratio: f64,
}

pub(crate) fn constitutive_stats(
    conductivities: &[f64],
    heat_capacities: &[f64],
    densities: &[f64],
) -> Option<ThermalConstitutiveStats> {
    if conductivities.is_empty()
        || conductivities.len() != heat_capacities.len()
        || conductivities.len() != densities.len()
    {
        return None;
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let spread_ratio = |values: &[f64]| {
        let min = values.iter().copied().reduce(f64::min)?;
        let max = values.iter().copied().reduce(f64::max)?;
        Some((max / min.max(1.0e-12)).max(1.0))
    };
    Some(ThermalConstitutiveStats {
        conductivity_mean: mean(conductivities),
        heat_capacity_mean: mean(heat_capacities),
        density_mean: mean(densities),
        conductivity_spread_ratio: spread_ratio(conductivities)?,
        heat_capacity_spread_ratio: spread_ratio(heat_capacities)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constitutive_stats_use_exact_element_coefficients() {
        let stats = constitutive_stats(&[2.0, 8.0], &[10.0, 30.0], &[4.0, 12.0]).unwrap();
        assert_eq!(stats.conductivity_mean, 5.0);
        assert_eq!(stats.heat_capacity_mean, 20.0);
        assert_eq!(stats.density_mean, 8.0);
        assert_eq!(stats.conductivity_spread_ratio, 4.0);
        assert_eq!(stats.heat_capacity_spread_ratio, 3.0);
        assert!(constitutive_stats(&[], &[], &[]).is_none());
        assert!(constitutive_stats(&[1.0], &[], &[1.0]).is_none());
    }
}
