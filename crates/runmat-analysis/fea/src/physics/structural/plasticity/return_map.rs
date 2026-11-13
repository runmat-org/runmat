use super::{
    IsotropicConstitutiveModel, PlasticityError, PlasticityPointState, PlasticityResponse,
};

pub(super) fn elastic_tangent(bulk: f64, shear: f64) -> [[f64; 6]; 6] {
    let mut tangent = [[0.0; 6]; 6];
    for (i, row) in tangent.iter_mut().enumerate() {
        if i < 3 {
            for (j, value) in row.iter_mut().enumerate().take(3) {
                *value = bulk + 2.0 * shear * (if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 });
            }
        } else {
            row[i] = shear;
        }
    }
    tangent
}

pub(super) struct TrialStress {
    pub pressure: f64,
    pub deviator: [f64; 6],
    pub equivalent: f64,
}

pub(super) fn trial_stress(
    model: &IsotropicConstitutiveModel,
    total: [f64; 6],
    committed: &PlasticityPointState,
) -> Result<TrialStress, PlasticityError> {
    let bulk = model.bulk_modulus_pa;
    let shear = model.shear_modulus_pa;
    let elastic_trial: [f64; 6] =
        std::array::from_fn(|i| total[i] - committed.engineering_plastic_strain[i]);
    let trace = elastic_trial[0] + elastic_trial[1] + elastic_trial[2];
    let pressure = bulk * trace;
    let mut deviator: [f64; 6] = std::array::from_fn(|i| {
        if i < 3 {
            2.0 * shear * (elastic_trial[i] - trace / 3.0)
        } else {
            shear * elastic_trial[i]
        }
    });
    // Preserve the deviatoric subspace at working precision, including a large hydrostatic
    // bias. Otherwise cancellation in three separate mean subtractions can introduce a
    // spurious volumetric plastic increment which should not be softened by state validation.
    deviator[2] = -deviator[0] - deviator[1];
    // q = sqrt(3/2 s:s); physical shear stresses occur twice in the tensor norm.
    let equivalent_trial = deviator
        .iter()
        .enumerate()
        .fold(0.0_f64, |norm, (i, value)| {
            norm.hypot(
                value
                    * if i < 3 {
                        (1.5_f64).sqrt()
                    } else {
                        (3.0_f64).sqrt()
                    },
            )
        });
    if !pressure.is_finite() || !equivalent_trial.is_finite() {
        return Err(PlasticityError::NonFiniteResponse);
    }
    Ok(TrialStress {
        pressure,
        deviator,
        equivalent: equivalent_trial,
    })
}

pub(super) fn update(
    model: &IsotropicConstitutiveModel,
    total: [f64; 6],
    committed: &PlasticityPointState,
) -> Result<PlasticityResponse, PlasticityError> {
    let bulk = model.bulk_modulus_pa;
    let shear = model.shear_modulus_pa;
    let TrialStress {
        pressure,
        deviator,
        equivalent: equivalent_trial,
    } = trial_stress(model, total, committed)?;
    let mut state = committed.clone();
    let mut tangent = model.elastic_tangent_pa();
    let mut radial_scale = 1.0;
    let mut dissipation = 0.0;
    let hardening = model.plastic.map_or(0.0, |law| law.modulus_pa);
    if let Some(law) = model.plastic {
        let current_yield = law.yield_stress_pa.canonical_value()
            + law.modulus_pa * committed.equivalent_plastic_strain;
        if !current_yield.is_finite() {
            return Err(PlasticityError::NonFiniteResponse);
        }
        let yield_excess = equivalent_trial - current_yield;
        if yield_excess > 0.0 {
            let denominator = 3.0 * shear + law.modulus_pa;
            let increment = yield_excess / denominator;
            radial_scale = (current_yield + law.modulus_pa * increment) / equivalent_trial;
            state.equivalent_plastic_strain += increment;
            for (i, value) in state.engineering_plastic_strain.iter_mut().enumerate() {
                *value +=
                    if i < 3 { 1.5 } else { 3.0 } * increment * deviator[i] / equivalent_trial;
            }
            dissipation = law.yield_stress_pa.canonical_value() * increment;
            // Differentiate sigma = K tr(e) I + beta*s_trial with committed history fixed.
            // dq/de_engineering = 3G*s_trial/q. The rank-one term is symmetric because
            // physical stress and engineering strain are energy-conjugate six-vectors.
            let rank_one =
                -3.0 * shear * (3.0 * shear / denominator) * (current_yield / equivalent_trial);
            for (i, row) in tangent.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    let volumetric = if i < 3 && j < 3 { bulk } else { 0.0 };
                    let elastic_deviatoric = *value - volumetric;
                    *value = volumetric
                        + radial_scale * elastic_deviatoric
                        + rank_one
                            * (deviator[i] / equivalent_trial)
                            * (deviator[j] / equivalent_trial);
                }
            }
        }
    }
    let stress_pa =
        std::array::from_fn(|i| radial_scale * deviator[i] + if i < 3 { pressure } else { 0.0 });
    let elastic_energy_density_pa = 0.5
        * stress_pa
            .iter()
            .enumerate()
            .map(|(i, stress)| stress * (total[i] - state.engineering_plastic_strain[i]))
            .sum::<f64>();
    let hardening_energy_density_pa = 0.5 * hardening * state.equivalent_plastic_strain.powi(2);
    state.validate()?;
    if stress_pa
        .iter()
        .chain(tangent.iter().flatten())
        .any(|value| !value.is_finite())
        || !elastic_energy_density_pa.is_finite()
        || elastic_energy_density_pa < 0.0
        || !hardening_energy_density_pa.is_finite()
        || !dissipation.is_finite()
        || dissipation < 0.0
    {
        return Err(PlasticityError::NonFiniteResponse);
    }
    Ok(PlasticityResponse {
        stress_pa,
        tangent_pa: tangent,
        state,
        elastic_energy_density_pa,
        hardening_energy_density_pa,
        plastic_dissipation_increment_pa: dissipation,
    })
}
