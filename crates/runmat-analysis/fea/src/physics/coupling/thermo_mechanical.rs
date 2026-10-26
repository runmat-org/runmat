use crate::assembly::ThermoMechanicalAssemblySummary;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientPolicy {
    pub effective_residual_target: f64,
    pub growth_limit: f64,
    pub nonconverged_shrink: f64,
}

pub fn severity(context: Option<&ThermoMechanicalAssemblySummary>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    let thermal_strain = (context.thermal_expansion_coefficient.canonical_value()
        * context.applied_temperature_delta_k.canonical_value().abs())
    .clamp(0.0, 0.05);
    (thermal_strain / 0.05).clamp(0.0, 1.0)
}

pub fn transient_policy(base_residual_target: f64, severity: f64) -> TransientPolicy {
    let thermo_residual_relaxation = 1.0 + 1.5 * severity;
    TransientPolicy {
        effective_residual_target: base_residual_target * thermo_residual_relaxation,
        growth_limit: (1.0 - 0.12 * severity).clamp(0.75, 1.0),
        nonconverged_shrink: (1.0 - 0.20 * severity).clamp(0.65, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermo_policies_remain_bounded() {
        let transient = transient_policy(1.0e-6, 0.8);
        assert!(transient.effective_residual_target >= 1.0e-6);
        assert!((0.75..=1.0).contains(&transient.growth_limit));
        assert!((0.65..=1.0).contains(&transient.nonconverged_shrink));
    }
}
