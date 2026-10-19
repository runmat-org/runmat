use super::*;
pub(super) fn evaluate(
    law: &PenaltyContactLaw,
    gap: f64,
    increment: [f64; 2],
    committed: &ContactPointState,
) -> Result<ContactPointResponse, ContactPointError> {
    let mut response = ContactPointResponse {
        mode: ContactPointMode::Open,
        normal_pressure_pa: 0.0,
        internal_traction_local_pa: [0.0; 3],
        tangent_pa_per_m: [[0.0; 3]; 3],
        trial_state: committed.clone(),
        normal_elastic_energy_j_per_m2: 0.0,
        tangential_elastic_energy_j_per_m2: 0.0,
        frictional_dissipation_increment_j_per_m2: 0.0,
        separation_release_energy_j_per_m2: 0.0,
        tangential_algorithmic_dissipation_increment_j_per_m2: 0.0,
    };
    response.trial_state.gap_m = gap;
    response.trial_state.tangential_traction_pa = [0.0; 2];
    if gap >= 0.0 {
        if let Some(k) = law.policy.tangential_stiffness_pa_per_m {
            let magnitude =
                committed.tangential_traction_pa[0].hypot(committed.tangential_traction_pa[1]);
            response.separation_release_energy_j_per_m2 = 0.5 * magnitude * (magnitude / k);
        }
        return finish(law, response);
    }
    let kn = law.policy.normal_stiffness_pa_per_m;
    let pressure = law.normal_pressure_at_gap(gap)?;
    response.normal_pressure_pa = pressure;
    response.internal_traction_local_pa[2] = -pressure;
    response.tangent_pa_per_m[2][2] = kn;
    response.normal_elastic_energy_j_per_m2 = 0.5 * pressure * (-gap);
    let Some(mu) = law.coefficient else {
        response.mode = ContactPointMode::FrictionlessActive;
        return finish(law, response);
    };
    let kt = law
        .policy
        .tangential_stiffness_pa_per_m
        .ok_or(ContactPointError::InvalidPolicy)?;
    let trial =
        std::array::from_fn::<_, 2, _>(|i| committed.tangential_traction_pa[i] + kt * increment[i]);
    let magnitude = trial[0].hypot(trial[1]);
    let radius = mu * pressure;
    if !magnitude.is_finite() || !radius.is_finite() {
        return Err(ContactPointError::NonFiniteResponse);
    }
    let traction;
    if radius > 0.0 && magnitude <= radius {
        response.mode = ContactPointMode::Stick;
        traction = trial;
        response.tangent_pa_per_m[0][0] = kt;
        response.tangent_pa_per_m[1][1] = kt;
    } else {
        response.mode = ContactPointMode::Slip;
        let direction = if magnitude > 0.0 {
            trial.map(|q| q / magnitude)
        } else {
            [0.0; 2]
        };
        traction = direction.map(|e| radius * e);
        let sliding_distance = (magnitude - radius) / kt;
        for i in 0..2 {
            response.trial_state.inelastic_slip_m[i] += (trial[i] - traction[i]) / kt;
            for j in 0..2 {
                response.tangent_pa_per_m[i][j] = if magnitude > 0.0 {
                    (radius / magnitude)
                        * kt
                        * (if i == j { 1.0 } else { 0.0 } - direction[i] * direction[j])
                } else {
                    0.0
                };
            }
            response.tangent_pa_per_m[i][2] = -mu * kn * direction[i];
        }
        response.trial_state.accumulated_sliding_distance_m += sliding_distance;
        response.frictional_dissipation_increment_j_per_m2 = radius * sliding_distance;
        response.trial_state.frictional_work_j_per_m2 +=
            response.frictional_dissipation_increment_j_per_m2;
    }
    let delta = [
        traction[0] - committed.tangential_traction_pa[0],
        traction[1] - committed.tangential_traction_pa[1],
    ];
    let delta = delta[0].hypot(delta[1]);
    response.tangential_algorithmic_dissipation_increment_j_per_m2 = 0.5 * delta * (delta / kt);
    response.trial_state.tangential_traction_pa = traction;
    response.internal_traction_local_pa[..2].copy_from_slice(&traction);
    let magnitude = traction[0].hypot(traction[1]);
    response.tangential_elastic_energy_j_per_m2 = 0.5 * magnitude * (magnitude / kt);
    finish(law, response)
}
fn finish(
    law: &PenaltyContactLaw,
    response: ContactPointResponse,
) -> Result<ContactPointResponse, ContactPointError> {
    if crate::is_fea_cancelled() {
        return Err(ContactPointError::Cancelled);
    }
    if response
        .internal_traction_local_pa
        .iter()
        .chain(response.tangent_pa_per_m.iter().flatten())
        .any(|x| !x.is_finite())
        || [
            response.normal_pressure_pa,
            response.normal_elastic_energy_j_per_m2,
            response.tangential_elastic_energy_j_per_m2,
            response.frictional_dissipation_increment_j_per_m2,
            response.separation_release_energy_j_per_m2,
            response.tangential_algorithmic_dissipation_increment_j_per_m2,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(ContactPointError::NonFiniteResponse);
    }
    law.validate_state(&response.trial_state)
        .map_err(|_| ContactPointError::NonFiniteResponse)?;
    Ok(response)
}
