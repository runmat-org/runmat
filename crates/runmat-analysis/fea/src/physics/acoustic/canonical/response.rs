use super::{assembly::AcousticAssembly, AcousticHarmonicError, AcousticHarmonicPolicy};
use crate::{
    excitation::{ProjectionScalar, ProjectionVector3, SpatialProjectionEvidence},
    topology::AnalysisProjection,
};
mod power;
mod reduced;
mod refinement;

#[derive(Debug, Clone)]
pub struct CanonicalAcousticResponse {
    pub frequency_hz: f64,
    pub projection: AnalysisProjection,
    /// Peak pressure phasors in Pa, in canonical projection node order.
    pub pressure_pa: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Pressure,
        Vec<ProjectionScalar>,
    >,
    /// Volume-averaged particle velocity phasors in m/s, in projection element order.
    pub element_velocity_m_per_s: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Velocity,
        Vec<ProjectionVector3>,
    >,
    /// Volume average of 0.5 Re(p conjugate(v)), from peak phasors, in element order.
    pub element_mean_intensity_w_per_m2: runmat_types::physical::Quantity<
        runmat_types::physical::kind::AcousticIntensity,
        Vec<[f64; 3]>,
    >,
    /// Plane-wave power density crossing the element's port faces, W/m², in element order:
    /// ½|p_inc|²/(ρc) of the authored incident wave on incident-pressure faces, ½|p|²/(ρc) of
    /// the outgoing wave on anechoic faces, averaged over the element's port-face area; zero
    /// for elements without port faces.
    pub element_port_intensity_w_per_m2:
        runmat_types::physical::Quantity<runmat_types::physical::kind::AcousticIntensity, Vec<f64>>,
    /// Volume average of (|p|²/(rho c²) + rho |v|²)/4, in element order.
    pub element_mean_energy_density_j_per_m3:
        runmat_types::physical::Quantity<runmat_types::physical::kind::EnergyDensity, Vec<f64>>,
    /// Iterative refinement steps of the certified direct solve.
    pub iterations: u32,
    pub relative_residual: f64,
    /// Positive into the fluid; includes prescribed-pressure reaction flux.
    pub mean_input_power_w:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Power, f64>,
    /// Total mean power dissipated by passive admittance: material loss in damped regions,
    /// positive real impedance and first-order local outgoing radiation.
    pub mean_impedance_loss_w:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Power, f64>,
    pub power_relative_error: f64,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
    /// Nonsingularity of the reduced operator, from the factorization that solved it.
    pub certificate: AcousticSolveCertificate,
}

/// Retained nonsingularity evidence of one harmonic solve: the shared factorization
/// certificate (`solve::certificate`) of the reduced real block operator.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticSolveCertificate {
    pub operator_identity: runmat_execution::Digest,
    pub dimension: usize,
    /// Estimated `‖I − A F⁻¹‖₁`.
    pub defect_estimate: f64,
    /// Estimated `‖F⁻¹‖₁`.
    pub inverse_norm_estimate: f64,
    /// `‖A‖₁`.
    pub operator_norm: f64,
    pub condition_estimate: f64,
    /// Safety-scaled contraction bound; strictly below `limit`.
    pub contraction_bound: f64,
    pub limit: f64,
}

impl AcousticSolveCertificate {
    /// Every unknown is prescribed: there is no operator left to certify.
    pub(super) fn no_free_unknowns(
        operator_identity: runmat_execution::Digest,
        limit: f64,
    ) -> Self {
        Self {
            operator_identity,
            dimension: 0,
            defect_estimate: 0.0,
            inverse_norm_estimate: 0.0,
            operator_norm: 0.0,
            condition_estimate: 0.0,
            contraction_bound: 0.0,
            limit,
        }
    }

    pub fn validate(&self, limit: f64, maximum_dimension: usize) -> Result<(), &'static str> {
        let values = [
            self.defect_estimate,
            self.inverse_norm_estimate,
            self.operator_norm,
            self.condition_estimate,
            self.contraction_bound,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || self.limit != limit
            || !(self.contraction_bound < self.limit)
            || self.dimension > maximum_dimension
        {
            return Err("acoustic factorization certificate is invalid or exceeds its limit");
        }
        Ok(())
    }
}

impl From<&crate::solve::certificate::NonsingularityCertificate> for AcousticSolveCertificate {
    fn from(certificate: &crate::solve::certificate::NonsingularityCertificate) -> Self {
        Self {
            operator_identity: certificate.operator_identity,
            dimension: certificate.dimension,
            defect_estimate: certificate.defect_estimate,
            inverse_norm_estimate: certificate.inverse_norm_estimate,
            operator_norm: certificate.operator_norm,
            condition_estimate: certificate.condition_estimate,
            contraction_bound: certificate.contraction_bound,
            limit: certificate.limit,
        }
    }
}

pub(super) fn solve(
    assembly: AcousticAssembly,
    frequency_hz: f64,
    policy: AcousticHarmonicPolicy,
    power_tolerance: f64,
) -> Result<CanonicalAcousticResponse, AcousticHarmonicError> {
    let (solution, power) = refinement::solve(&assembly, policy, power_tolerance)?;
    let reduced::ReducedSolution {
        real,
        imaginary,
        iterations,
        relative_residual,
        certificate,
    } = solution;
    let power::PowerClosure {
        mean_input_power_w,
        mean_impedance_loss_w,
        power_relative_error,
    } = power;
    let recovered = super::recovery::recover(&assembly, &real, &imaginary)?;
    let port_intensity = port_intensity(&assembly, &real, &imaginary)?;
    if real
        .iter()
        .chain(&imaginary)
        .any(|value| !value.is_finite())
        || recovered.velocity.iter().any(|v| {
            v.real
                .iter()
                .chain(&v.imaginary)
                .any(|value| !value.is_finite())
        })
        || [
            mean_input_power_w.canonical_value(),
            mean_impedance_loss_w.canonical_value(),
            power_relative_error,
            relative_residual,
        ]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(AcousticHarmonicError::Operator(
            "nonfinite recovered field or power".into(),
        ));
    }
    let pressure_pa = real
        .into_iter()
        .zip(imaginary)
        .map(|(real, imaginary)| ProjectionScalar { real, imaginary })
        .collect();
    Ok(CanonicalAcousticResponse {
        frequency_hz,
        projection: assembly.projection,
        pressure_pa:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Pressure, _>::new(
                pressure_pa,
            ),
        element_velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(recovered.velocity),
        element_mean_intensity_w_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::AcousticIntensity,
            _,
        >::new(recovered.intensity),
        element_port_intensity_w_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::AcousticIntensity,
            _,
        >::new(port_intensity),
        element_mean_energy_density_j_per_m3: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::EnergyDensity,
            _,
        >::new(recovered.energy),
        iterations,
        relative_residual,
        mean_input_power_w,
        mean_impedance_loss_w,
        power_relative_error,
        source_evidence: assembly.source_evidence,
        certificate,
    })
}

/// Per-element plane-wave port intensity: the face-quadrature power of the authored incident
/// wave over incident-pressure faces, and of the outgoing wave over anechoic faces, divided by
/// the element's port-face area. An element's port faces must all be of one kind, so that a
/// face-area integral over a port region recovers the exact port power.
fn port_intensity(
    assembly: &AcousticAssembly,
    real: &[f64],
    imaginary: &[f64],
) -> Result<Vec<f64>, AcousticHarmonicError> {
    let elements = assembly.projection.elements.len();
    let mut power = vec![0.0; elements];
    let mut area = vec![0.0; elements];
    let mut incident_owner = vec![None::<bool>; elements];
    for port in &assembly.port_faces {
        let [owner] = port.face.adjacent_element_indices.as_slice() else {
            return Err(AcousticHarmonicError::Boundary(
                "an acoustic port face needs exactly one owning element".into(),
            ));
        };
        let incident = port.incident.is_some();
        match incident_owner.get_mut(*owner) {
            Some(Some(kind)) if *kind != incident => {
                return Err(AcousticHarmonicError::Boundary(
                    "an element touches both an incident-pressure face and an anechoic face; refine the mesh between the ports".into(),
                ))
            }
            Some(kind) => *kind = Some(incident),
            None => {
                return Err(AcousticHarmonicError::Topology(
                    "acoustic port face owner is outside the projection".into(),
                ))
            }
        }
        assembly
            .projection
            .visit_boundary_quadrature(&port.face, |point| {
                let squared = match port.incident {
                    Some(pressure) => pressure.real.powi(2) + pressure.imaginary.powi(2),
                    None => {
                        let mut pressure = [0.0; 2];
                        for (&node, &shape) in
                            port.face.node_indices.iter().zip(point.shape_values())
                        {
                            pressure[0] += shape * real[node];
                            pressure[1] += shape * imaginary[node];
                        }
                        pressure[0].powi(2) + pressure[1].powi(2)
                    }
                };
                power[*owner] += 0.5 * port.admittance * squared * point.area_weight_m2;
                area[*owner] += point.area_weight_m2;
            })
            .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
    }
    power
        .into_iter()
        .zip(area)
        .map(|(power, area)| {
            let value = if area > 0.0 { power / area } else { 0.0 };
            if value.is_finite() && value >= 0.0 {
                Ok(value)
            } else {
                Err(AcousticHarmonicError::Operator(
                    "nonfinite acoustic port intensity".into(),
                ))
            }
        })
        .collect()
}
