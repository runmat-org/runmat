//! Surface-to-ambient gray radiation forms; not an admitted thermal
//! procedure. No view factors, enclosure exchange or participating medium.
use runmat_analysis_core::SurfaceEmissivity;
use thiserror::Error;

use crate::topology::{
    quadrature::{
        GAUSS_LEGENDRE_EIGHT_UNIT, GAUSS_LEGENDRE_SIXTEEN_UNIT, GAUSS_LEGENDRE_TWELVE_UNIT,
    },
    AnalysisProjection, ProjectedBoundaryFace,
};

/// CODATA SI value rounded to ten significant digits, W m^-2 K^-4.
pub const STEFAN_BOLTZMANN_W_PER_M2_K4: f64 = 5.670_374_419e-8;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum RadiationError {
    #[error(
        "radiation requires finite nonnegative absolute temperatures and emissivity in [0, 1]"
    )]
    InvalidState,
    #[error("radiation face geometry is invalid: {0}")]
    InvalidGeometry(String),
    #[error("radiation face residual or tangent integration did not converge")]
    IntegrationNotConverged,
}

/// Positive outward heat loss and its exact temperature derivative. Factoring
/// the fourth-power difference preserves small transfers near equilibrium.
pub fn outward_flux_and_tangent(
    temperature_k: f64,
    ambient_k: f64,
    emissivity: SurfaceEmissivity,
) -> Result<(f64, f64), RadiationError> {
    if !temperature_k.is_finite()
        || temperature_k < 0.0
        || !ambient_k.is_finite()
        || ambient_k < 0.0
        || !emissivity.canonical_value().is_finite()
        || !(0.0..=1.0).contains(&emissivity.canonical_value())
    {
        return Err(RadiationError::InvalidState);
    }
    if emissivity.canonical_value() == 0.0 {
        return Ok((0.0, 0.0));
    }
    let coefficient = emissivity.canonical_value() * STEFAN_BOLTZMANN_W_PER_M2_K4;
    let flux = coefficient
        * (temperature_k - ambient_k)
        * (temperature_k + ambient_k)
        * (temperature_k * temperature_k + ambient_k * ambient_k);
    let tangent = 4.0 * coefficient * temperature_k.powi(3);
    if !flux.is_finite() || !tangent.is_finite() {
        return Err(RadiationError::InvalidState);
    }
    Ok((flux, tangent))
}

/// Consistent face residual and Jacobian in face-local node order. These fixed
/// arrays bound scratch independently of mesh size; inactive entries are zero.
#[derive(Debug, Clone)]
pub struct RadiationFaceForm {
    pub outward_nodal_power_w: [f64; 6],
    pub tangent_w_per_k: [[f64; 6]; 6],
    pub node_count: usize,
    pub outward_power_w: f64,
    pub absolute_transfer_w: f64,
    pub integrated_tangent_w_per_k: f64,
}

/// Integrates the actual nonlinear form at 8x8/12x12, refining to 16x16
/// when needed. Flat Tri6 degree-ten radiation forms are polynomial-exact;
/// curved forms must satisfy a 1e-10 physical-transfer/tangent comparison.
/// Positive nodal Kelvin values alone do not guarantee positive Tri6 samples.
pub fn integrate_face(
    projection: &AnalysisProjection,
    face: &ProjectedBoundaryFace,
    nodal_temperature_k: &[f64],
    ambient_k: f64,
    emissivity: SurfaceEmissivity,
) -> Result<RadiationFaceForm, RadiationError> {
    if nodal_temperature_k.len() != face.node_indices.len()
        || nodal_temperature_k
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(RadiationError::InvalidState);
    }
    outward_flux_and_tangent(0.0, ambient_k, emissivity)?;
    let coarse = integrate_rule(
        projection,
        face,
        nodal_temperature_k,
        ambient_k,
        emissivity,
        &GAUSS_LEGENDRE_EIGHT_UNIT,
    )?;
    let refined = integrate_rule(
        projection,
        face,
        nodal_temperature_k,
        ambient_k,
        emissivity,
        &GAUSS_LEGENDRE_TWELVE_UNIT,
    )?;
    if converged(&coarse, &refined) {
        return Ok(refined);
    }
    let finest = integrate_rule(
        projection,
        face,
        nodal_temperature_k,
        ambient_k,
        emissivity,
        &GAUSS_LEGENDRE_SIXTEEN_UNIT,
    )?;
    if converged(&refined, &finest) {
        Ok(finest)
    } else {
        Err(RadiationError::IntegrationNotConverged)
    }
}

fn integrate_rule(
    projection: &AnalysisProjection,
    face: &ProjectedBoundaryFace,
    temperatures: &[f64],
    ambient: f64,
    emissivity: SurfaceEmissivity,
    rule: &[(f64, f64)],
) -> Result<RadiationFaceForm, RadiationError> {
    let mut form = RadiationFaceForm {
        outward_nodal_power_w: [0.0; 6],
        tangent_w_per_k: [[0.0; 6]; 6],
        node_count: temperatures.len(),
        outward_power_w: 0.0,
        absolute_transfer_w: 0.0,
        integrated_tangent_w_per_k: 0.0,
    };
    let mut invalid = false;
    projection
        .visit_boundary_tensor_rule(face, rule, |point| {
            let reference = temperatures[0];
            let temperature = reference
                + temperatures
                    .iter()
                    .zip(point.shape_values())
                    .map(|(t, n)| (t - reference) * n)
                    .sum::<f64>();
            let Ok((flux, derivative)) = outward_flux_and_tangent(temperature, ambient, emissivity)
            else {
                invalid = true;
                return;
            };
            let power = flux * point.area_weight_m2;
            let tangent = derivative * point.area_weight_m2;
            form.outward_power_w += power;
            form.absolute_transfer_w += power.abs();
            form.integrated_tangent_w_per_k += tangent;
            for (i, &ni) in point.shape_values().iter().enumerate() {
                form.outward_nodal_power_w[i] += ni * power;
                for (j, &nj) in point.shape_values().iter().enumerate() {
                    form.tangent_w_per_k[i][j] += ni * nj * tangent;
                }
            }
        })
        .map_err(|error| RadiationError::InvalidGeometry(error.to_string()))?;
    if invalid
        || !form.absolute_transfer_w.is_finite()
        || !form.integrated_tangent_w_per_k.is_finite()
    {
        Err(RadiationError::InvalidState)
    } else {
        Ok(form)
    }
}

fn converged(left: &RadiationFaceForm, right: &RadiationFaceForm) -> bool {
    let power_scale = left.absolute_transfer_w.max(right.absolute_transfer_w);
    let tangent_scale = left
        .integrated_tangent_w_per_k
        .max(right.integrated_tangent_w_per_k);
    let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-10 * scale;
    close(left.outward_power_w, right.outward_power_w, power_scale)
        && left
            .outward_nodal_power_w
            .iter()
            .zip(right.outward_nodal_power_w)
            .all(|(a, b)| close(*a, b, power_scale))
        && left
            .tangent_w_per_k
            .iter()
            .flatten()
            .zip(right.tangent_w_per_k.iter().flatten())
            .all(|(a, b)| close(*a, *b, tangent_scale))
}

#[cfg(test)]
mod tests;
