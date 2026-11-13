use super::state::RawPointState;
use super::*;
pub(super) fn evaluate(
    law: &PenaltyContactLaw,
    gap: f64,
    increment: [f64; 2],
    committed_state: &ContactPointState,
) -> Result<ContactPointResponse, ContactPointError> {
    let committed = RawPointState::of(committed_state);
    let tangential_stiffness = law
        .policy
        .tangential_stiffness_pa_per_m
        .map(|k| k.canonical_value());
    let mut response = RawResponse {
        mode: ContactPointMode::Open,
        normal_pressure_pa: 0.0,
        internal_traction_local_pa: [0.0; 3],
        tangent_pa_per_m: [[0.0; 3]; 3],
        trial_state: committed,
        normal_elastic_energy_j_per_m2: 0.0,
        tangential_elastic_energy_j_per_m2: 0.0,
        frictional_dissipation_increment_j_per_m2: 0.0,
        separation_release_energy_j_per_m2: 0.0,
        tangential_algorithmic_dissipation_increment_j_per_m2: 0.0,
    };
    response.trial_state.gap_m = gap;
    response.trial_state.tangential_traction_pa = [0.0; 2];
    if gap >= 0.0 {
        if let Some(k) = tangential_stiffness {
            let magnitude =
                committed.tangential_traction_pa[0].hypot(committed.tangential_traction_pa[1]);
            response.separation_release_energy_j_per_m2 = 0.5 * magnitude * (magnitude / k);
        }
        return finish(law, response);
    }
    let kn = law.policy.normal_stiffness_pa_per_m.canonical_value();
    let pressure = law.normal_pressure_at_canonical_gap(gap)?;
    response.normal_pressure_pa = pressure;
    response.internal_traction_local_pa[2] = -pressure;
    response.tangent_pa_per_m[2][2] = kn;
    response.normal_elastic_energy_j_per_m2 = 0.5 * pressure * (-gap);
    let Some(mu) = law.coefficient else {
        response.mode = ContactPointMode::FrictionlessActive;
        return finish(law, response);
    };
    let kt = tangential_stiffness.ok_or(ContactPointError::InvalidPolicy)?;
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
/// Canonical SI response of the private penalty/Coulomb kernel before typed publication.
struct RawResponse {
    mode: ContactPointMode,
    normal_pressure_pa: f64,
    internal_traction_local_pa: [f64; 3],
    tangent_pa_per_m: [[f64; 3]; 3],
    trial_state: RawPointState,
    normal_elastic_energy_j_per_m2: f64,
    tangential_elastic_energy_j_per_m2: f64,
    frictional_dissipation_increment_j_per_m2: f64,
    tangential_algorithmic_dissipation_increment_j_per_m2: f64,
    separation_release_energy_j_per_m2: f64,
}
fn finish(
    law: &PenaltyContactLaw,
    response: RawResponse,
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
    let trial_state = response.trial_state.into_state(law.identity);
    law.validate_state(&trial_state)
        .map_err(|_| ContactPointError::NonFiniteResponse)?;
    let energy = runmat_types::physical::SurfaceEnergyDensity::new;
    Ok(ContactPointResponse {
        mode: response.mode,
        normal_pressure_pa: Pressure::new(response.normal_pressure_pa),
        internal_traction_local_pa: Quantity::new(response.internal_traction_local_pa),
        tangent_pa_per_m: Quantity::new(response.tangent_pa_per_m),
        trial_state,
        normal_elastic_energy_j_per_m2: energy(response.normal_elastic_energy_j_per_m2),
        tangential_elastic_energy_j_per_m2: energy(response.tangential_elastic_energy_j_per_m2),
        frictional_dissipation_increment_j_per_m2: energy(
            response.frictional_dissipation_increment_j_per_m2,
        ),
        tangential_algorithmic_dissipation_increment_j_per_m2: energy(
            response.tangential_algorithmic_dissipation_increment_j_per_m2,
        ),
        separation_release_energy_j_per_m2: energy(response.separation_release_energy_j_per_m2),
    })
}
