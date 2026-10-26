//! Recover actual basis fields, complex Poynting balance and harmonic continuity.
use super::{
    assembly::MaxwellAssembly,
    complex::{dot_conjugate, vector_norm_squared},
    continuity::ChargeContinuity,
    topology, ComplexPhasor, MaxwellError, MaxwellFieldSample, MaxwellResponse,
};

pub(super) fn recover(
    assembled: &MaxwellAssembly<'_>,
    edges: Vec<ComplexPhasor>,
    physical: f64,
    iterations: u32,
    tolerance: f64,
) -> Result<MaxwellResponse, MaxwellError> {
    let omega = assembled
        .input
        .angular_frequency_rad_per_s
        .phasor_derivative_rate()
        .canonical_value();
    let mut samples = Vec::with_capacity(assembled.input.sampling.integration_samples().len());
    let exterior = assembled
        .edge_space
        .exterior_edge_indices(&assembled.projection)
        .map_err(topology)?;
    let mut continuity =
        ChargeContinuity::new(&assembled.projection, &assembled.edge_space, &exterior);
    let mut electric_energy = 0.0;
    let mut magnetic_energy = 0.0;
    let mut ohmic = 0.0;
    let mut source_power = ComplexPhasor::ZERO;
    for (index, data) in assembled.volumes.iter().enumerate() {
        let indices = assembled
            .edge_space
            .element_edge_indices(index)
            .ok_or_else(|| topology("missing Maxwell element edges"))?;
        let mut ordinal = 0;
        assembled
            .edge_space
            .visit_element_quadrature(&assembled.projection, index, |point| {
                let position = index * 15 + ordinal;
                let authority = &assembled.input.sampling.integration_samples()[position];
                let source = assembled.input.source_current[position]
                    .current_density_a_per_m2
                    .canonical_value();
                let electric = std::array::from_fn(|axis| {
                    (0..6).fold(ComplexPhasor::ZERO, |sum, i| {
                        sum + edges[indices[i]] * point.values[i][axis]
                    })
                });
                let curl: [ComplexPhasor; 3] = std::array::from_fn(|axis| {
                    (0..6).fold(ComplexPhasor::ZERO, |sum, i| {
                        sum + edges[indices[i]] * point.curls[i][axis]
                    })
                });
                let flux = curl.map(|value| value.times_i() / omega);
                let magnetic =
                    flux.map(|value| value / data.permeability_h_per_m.canonical_value());
                let displacement =
                    electric.map(|value| value * data.permittivity_f_per_m.canonical_value());
                let conduction =
                    electric.map(|value| value * data.conductivity_s_per_m.canonical_value());
                let weight = point.volume_weight_m3;
                electric_energy += 0.25
                    * weight
                    * data.permittivity_f_per_m.canonical_value()
                    * vector_norm_squared(electric);
                magnetic_energy += 0.25
                    * weight
                    * data.permeability_h_per_m.canonical_value()
                    * vector_norm_squared(magnetic);
                ohmic += 0.5
                    * weight
                    * data.conductivity_s_per_m.canonical_value()
                    * vector_norm_squared(electric);
                source_power += dot_conjugate(electric, source) * (-0.5 * weight);
                continuity.accumulate(
                    &assembled.projection.elements[index].node_indices[..4],
                    &point,
                    conduction,
                    displacement,
                    source,
                    omega,
                );
                samples.push(MaxwellFieldSample {
                    sample_identity: authority.identity,
                    element_identity: authority.element_identity,
                    ordinal,
                    position_m: point.position_m,
                    volume_weight_m3: weight,
                    electric_field_v_per_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::ElectricField,
                        _,
                    >::new(electric),
                    flux_density_t: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::MagneticFluxDensity,
                        _,
                    >::new(flux),
                    magnetic_field_a_per_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::MagneticFieldIntensity,
                        _,
                    >::new(magnetic),
                    electric_displacement_c_per_m2: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::ElectricDisplacement,
                        _,
                    >::new(displacement),
                    conduction_current_density_a_per_m2: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::ElectricCurrentDensity,
                        _,
                    >::new(conduction),
                });
                ordinal += 1;
            })
            .map_err(topology)?;
    }
    let reactive = 2.0 * omega * (magnetic_energy - electric_energy);
    let predicted = ComplexPhasor::new(ohmic, reactive);
    let scale = predicted.abs().max(source_power.abs());
    let closure = if scale > 0.0 {
        (source_power - predicted).abs() / scale
    } else {
        (source_power - predicted).abs()
    };
    let charge = continuity.relative_defect();
    if [
        electric_energy,
        magnetic_energy,
        ohmic,
        reactive,
        closure,
        charge,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || !source_power.is_finite()
        || ohmic < 0.0
        || closure > tolerance * 16.0
        || charge > tolerance * 16.0
        || samples.iter().any(|s| {
            [
                s.electric_field_v_per_m.canonical_value(),
                s.flux_density_t.canonical_value(),
                s.magnetic_field_a_per_m.canonical_value(),
                s.electric_displacement_c_per_m2.canonical_value(),
                s.conduction_current_density_a_per_m2.canonical_value(),
            ]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        })
    {
        return Err(MaxwellError::NotConverged(format!(
            "complex power error={closure:e}, harmonic charge continuity={charge:e}"
        )));
    }
    Ok(MaxwellResponse {
        input_identity: assembled.input.identity()?,
        system_plan_identity: assembled.plan.identity()?,
        mesh_digest: assembled.projection.mesh_digest,
        projection_digest: assembled.projection.projection_digest,
        sampling_digest: assembled.input.sampling.sampling_digest(),
        angular_frequency_rad_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::AngularFrequency,
            _,
        >::new(omega),
        edge_node_identities: assembled
            .edge_space
            .edges()
            .iter()
            .map(|edge| edge.node_identities)
            .collect(),
        edge_circulations_v: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ElectricPotential,
            _,
        >::new(edges),
        samples,
        electric_energy_j: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Energy,
            _,
        >::new(electric_energy),
        magnetic_energy_j: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Energy,
            _,
        >::new(magnetic_energy),
        source_power_va: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ApparentPower,
            _,
        >::new(source_power),
        ohmic_power_w:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Power, _>::new(ohmic),
        reactive_power_var: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::ReactivePower,
            _,
        >::new(reactive),
        relative_physical_residual: physical,
        relative_complex_power_error: closure,
        relative_charge_continuity_residual: charge,
        iterations,
    })
}
