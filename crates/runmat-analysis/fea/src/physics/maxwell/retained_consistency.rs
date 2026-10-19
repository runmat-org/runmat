//! Validate original retained arrays against their original edge coefficients.
//! No fields are replaced, no sparse operator is allocated and no solve executes.
use super::{
    continuity::ChargeContinuity, geometry, invalid, topology, ComplexPhasor, MaxwellError,
    MaxwellInput, MaxwellResponse,
};
use crate::topology::AnalysisProjection;
use runmat_meshing_core::SolverMeshArtifact;

pub(crate) fn validate_retained_consistency(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    response: &MaxwellResponse,
    tolerance: f64,
) -> Result<(), MaxwellError> {
    let volumes = input.ordered_volumes(mesh)?;
    let projection = AnalysisProjection::derive(mesh).map_err(topology)?;
    let (space, exterior) = geometry::spaces(&projection)?;
    let edges = &response.edge_circulations_v;
    if edges.len() != space.edges().len()
        || !response
            .edge_node_identities
            .iter()
            .copied()
            .eq(space.edges().iter().map(|edge| edge.node_identities))
        || exterior.iter().any(|&i| edges[i] != ComplexPhasor::ZERO)
        || response.samples.len() != input.sampling.integration_samples().len()
    {
        return Err(invalid(
            "retained Maxwell edges differ from original oriented PEC space",
        ));
    }
    // Linear edge buffers only; existing complete source-validation reservations
    // cover this independently checked original equation, not an assembled solve.
    let mut action = vec![ComplexPhasor::ZERO; edges.len()];
    let mut rhs = action.clone();
    let mut rounding_scale = vec![0.0; edges.len()];
    let omega = input.angular_frequency_rad_per_s;
    let mut continuity = ChargeContinuity::new(&projection, &space, &exterior);
    for (element, law) in volumes.iter().enumerate() {
        let indices = space
            .element_edge_indices(element)
            .ok_or_else(|| topology("retained Maxwell local edge inventory absent"))?;
        let moments = space
            .element_moments(&projection, element)
            .map_err(topology)?;
        for (i, &edge) in indices.iter().enumerate() {
            for (j, &other) in indices.iter().enumerate() {
                let coefficient = ComplexPhasor::new(
                    moments.curl_curl[i][j] / law.permeability_h_per_m
                        - omega * omega * law.permittivity_f_per_m * moments.mass[i][j],
                    omega * law.conductivity_s_per_m * moments.mass[i][j],
                );
                let term = coefficient * edges[other];
                action[edge] += term;
                rounding_scale[edge] += coefficient.abs() * edges[other].abs();
            }
        }
        let mut ordinal = 0;
        let mut mismatch = false;
        space
            .visit_element_quadrature(&projection, element, |point| {
                let offset = element * 15 + ordinal;
                let sample = &response.samples[offset];
                let electric: [_; 3] = std::array::from_fn(|axis| {
                    (0..6).fold(ComplexPhasor::ZERO, |sum, i| {
                        sum + edges[indices[i]] * point.values[i][axis]
                    })
                });
                let curl: [_; 3] = std::array::from_fn(|axis| {
                    (0..6).fold(ComplexPhasor::ZERO, |sum, i| {
                        sum + edges[indices[i]] * point.curls[i][axis]
                    })
                });
                mismatch |= sample.electric_field_v_per_m != electric
                    || sample.flux_density_t != curl.map(|v| v.times_i() / omega);
                continuity.accumulate(
                    &projection.elements[element].node_indices[..4],
                    &point,
                    electric.map(|v| v * law.conductivity_s_per_m),
                    electric.map(|v| v * law.permittivity_f_per_m),
                    input.source_current[offset].current_density_a_per_m2,
                    omega,
                );
                for (i, &edge) in indices.iter().enumerate() {
                    for axis in 0..3 {
                        let source = input.source_current[offset].current_density_a_per_m2[axis];
                        let factor = omega * point.values[i][axis] * point.volume_weight_m3;
                        rhs[edge] += -source.times_i() * factor;
                        rounding_scale[edge] += source.abs() * factor.abs();
                    }
                }
                ordinal += 1;
            })
            .map_err(topology)?;
        if mismatch || ordinal != 15 {
            return Err(invalid(
                "retained Maxwell E/B samples disagree with original edge phasors",
            ));
        }
    }
    let mut exterior_flags = vec![false; edges.len()];
    for edge in exterior {
        exterior_flags[edge] = true;
    }
    let mut load_norm = 0.0_f64;
    let mut defect_norm = 0.0_f64;
    let mut round_norm = 0.0_f64;
    for edge in 0..edges.len() {
        if exterior_flags[edge] {
            continue;
        }
        load_norm = load_norm.hypot(rhs[edge].real).hypot(rhs[edge].imaginary);
        let defect = action[edge] - rhs[edge];
        defect_norm = defect_norm.hypot(defect.real).hypot(defect.imaginary);
        round_norm = round_norm.hypot(rounding_scale[edge]);
    }
    let divisor = if load_norm > 0.0 { load_norm } else { 1.0 };
    let residual = defect_norm / divisor;
    // Different accumulation order from canonical CSR, not numerical damping.
    // Scale by the actual uncancelled original operator/source magnitudes.
    let rounding = 4096.0 * f64::EPSILON * round_norm / divisor;
    if !residual.is_finite()
        || !rounding.is_finite()
        || residual > 8.0 * tolerance + rounding
        || (residual - response.relative_physical_residual).abs() > rounding
    {
        return Err(invalid(
            "retained Maxwell original physical residual differs",
        ));
    }
    let charge = continuity.relative_defect();
    if !charge.is_finite()
        || charge > 16.0 * tolerance
        || charge != response.relative_charge_continuity_residual
    {
        return Err(invalid(
            "retained Maxwell original charge continuity differs",
        ));
    }
    Ok(())
}
