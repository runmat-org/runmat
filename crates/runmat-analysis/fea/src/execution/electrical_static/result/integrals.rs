//! Checks retained density/total consistency using canonical volumes, not field recovery.
use super::*;

pub(super) fn validate_for(
    response: &CanonicalElectricalResponse,
    projection: &AnalysisProjection,
) -> AnalysisExecutionResult<()> {
    let (densities, total) = match &response.constitutive {
        ElectricalConstitutiveResponse::Electrostatics {
            energy_density_j_per_m3,
            total_energy_j,
            ..
        } => (energy_density_j_per_m3, *total_energy_j),
        ElectricalConstitutiveResponse::SteadyConduction {
            joule_density_w_per_m3,
            total_joule_power_w,
            ..
        } => (joule_density_w_per_m3, *total_joule_power_w),
    };
    let mut integrated = 0.0;
    for (element, density) in projection.elements.iter().zip(densities) {
        let mut volume = 0.0;
        projection
            .visit_element_quadrature(element, |point| volume += point.volume_weight_m3)
            .map_err(invalid)?;
        if !volume.is_finite() || volume <= 0.0 {
            return Err(invalid("invalid canonical electrical element volume"));
        }
        integrated += density * volume;
    }
    let bound = 128.0
        * f64::EPSILON
        * (densities.len() as f64).max(1.0)
        * total.abs().max(integrated.abs()).max(f64::MIN_POSITIVE);
    if !integrated.is_finite() || !bound.is_finite() || (integrated - total).abs() > bound {
        return Err(invalid(
            "electrical integrated total disagrees with retained element densities",
        ));
    }
    Ok(())
}
