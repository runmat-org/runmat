use crate::{FeaContactInterfaceContext, FeaPlasticityConstitutiveContext};

pub fn plasticity_severity(context: Option<&FeaPlasticityConstitutiveContext>) -> f64 {
    let Some(ctx) = context else {
        return 0.0;
    };
    if !ctx.enabled {
        return 0.0;
    }
    let yield_component = (0.012 / ctx.yield_strain.max(1.0e-6)).clamp(0.0, 1.5) / 1.5;
    let hardening_component = (ctx.hardening_modulus_ratio / 0.2).clamp(0.0, 1.5) / 1.5;
    let saturation_component = (ctx.saturation_exponent / 4.0).clamp(0.0, 1.5) / 1.5;
    (0.55 * yield_component + 0.35 * hardening_component + 0.10 * saturation_component)
        .clamp(0.0, 1.0)
}

pub fn contact_severity(context: Option<&FeaContactInterfaceContext>) -> f64 {
    let Some(ctx) = context else {
        return 0.0;
    };
    if !ctx.enabled {
        return 0.0;
    }
    let penetration_component = (ctx.max_penetration_ratio / 0.02).clamp(0.0, 1.5) / 1.5;
    let penalty_component = (1.0 / ctx.penalty_stiffness_scale.max(1.0e-6)).clamp(0.0, 1.5) / 1.5;
    let friction_component = (ctx.friction_coefficient / 0.8).clamp(0.0, 1.5) / 1.5;
    (0.5 * penetration_component + 0.3 * penalty_component + 0.2 * friction_component)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonlinear_coupling_severity_is_bounded() {
        let plastic = FeaPlasticityConstitutiveContext {
            enabled: true,
            yield_strain: 2.0e-4,
            hardening_modulus_ratio: 0.2,
            saturation_exponent: 4.0,
        };
        let contact = FeaContactInterfaceContext {
            enabled: true,
            penalty_stiffness_scale: 0.2,
            max_penetration_ratio: 0.03,
            friction_coefficient: 0.8,
        };
        let p = plasticity_severity(Some(&plastic));
        let c = contact_severity(Some(&contact));
        assert!((0.0..=1.0).contains(&p));
        assert!((0.0..=1.0).contains(&c));
    }
}
