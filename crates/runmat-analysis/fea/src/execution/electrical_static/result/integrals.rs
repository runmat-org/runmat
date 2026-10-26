//! Checks retained density/total consistency using canonical volumes, not field recovery.
use super::*;

pub(super) fn validate_for(
    response: &CanonicalElectricalResponse,
    projection: &AnalysisProjection,
) -> AnalysisExecutionResult<()> {
    match &response.constitutive {
        ElectricalConstitutiveResponse::Electrostatics {
            energy_density_j_per_m3,
            total_energy_j,
            ..
        } => validate_integral(energy_density_j_per_m3, *total_energy_j, projection),
        ElectricalConstitutiveResponse::SteadyConduction {
            joule_density_w_per_m3,
            total_joule_power_w,
            ..
        } => validate_integral(joule_density_w_per_m3, *total_joule_power_w, projection),
    }
}

fn validate_integral<
    D: runmat_types::physical::QuantityMarker,
    T: runmat_types::physical::QuantityMarker,
>(
    densities: &runmat_types::physical::Quantity<D, Vec<f64>>,
    total: runmat_types::physical::Quantity<T>,
    projection: &AnalysisProjection,
) -> AnalysisExecutionResult<()> {
    let expected = densities
        .dimension()
        .multiply(runmat_types::PhysicalDimension::VOLUME);
    if expected != Some(total.dimension()) || densities.angle_power() != total.angle_power() {
        return Err(invalid(
            "electrical density integral has incompatible dimensions",
        ));
    }
    // Numerical quadrature retains its density and total types through this owner.
    let densities = densities.canonical_values();
    let total = total.canonical_value();
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
