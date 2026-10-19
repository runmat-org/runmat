use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactPointState {
    pub(super) law_identity: Digest,
    pub(super) gap_m: f64,
    pub(super) tangential_traction_pa: [f64; 2],
    pub(super) inelastic_slip_m: [f64; 2],
    pub(super) accumulated_sliding_distance_m: f64,
    pub(super) frictional_work_j_per_m2: f64,
}
impl ContactPointState {
    pub fn law_identity(&self) -> Digest {
        self.law_identity
    }
    pub fn gap_m(&self) -> f64 {
        self.gap_m
    }
    pub fn tangential_traction_pa(&self) -> [f64; 2] {
        self.tangential_traction_pa
    }
    pub fn inelastic_slip_m(&self) -> [f64; 2] {
        self.inelastic_slip_m
    }
    pub fn accumulated_sliding_distance_m(&self) -> f64 {
        self.accumulated_sliding_distance_m
    }
    pub fn frictional_work_j_per_m2(&self) -> f64 {
        self.frictional_work_j_per_m2
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactPointMode {
    Open,
    FrictionlessActive,
    Stick,
    Slip,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ContactPointResponse {
    pub mode: ContactPointMode,
    pub normal_pressure_pa: f64,
    /// Internal generalized traction [q1,q2,-p], conjugate to [s1,s2,g].
    /// The physical contact force on the secondary side has the opposite sign.
    pub internal_traction_local_pa: [f64; 3],
    /// Derivative wrt current [tangential increment1, increment2, gap],
    /// holding committed history and the material frame fixed. NOT the complete FE tangent.
    pub tangent_pa_per_m: [[f64; 3]; 3],
    pub trial_state: ContactPointState,
    pub normal_elastic_energy_j_per_m2: f64,
    pub tangential_elastic_energy_j_per_m2: f64,
    /// Backward-Euler point dissipation at current Coulomb capacity; not exact path work.
    pub frictional_dissipation_increment_j_per_m2: f64,
    /// Backward-Euler elastic update loss; a numerical term, not Coulomb heat.
    pub tangential_algorithmic_dissipation_increment_j_per_m2: f64,
    /// Tangential spring energy discarded upon separation; not classified as frictional heat.
    pub separation_release_energy_j_per_m2: f64,
}
impl PenaltyContactLaw {
    pub fn validate_state(&self, state: &ContactPointState) -> Result<(), ContactPointError> {
        if state.law_identity != self.identity
            || !state.gap_m.is_finite()
            || state
                .tangential_traction_pa
                .iter()
                .chain(state.inelastic_slip_m.iter())
                .any(|x| !x.is_finite())
            || !state.accumulated_sliding_distance_m.is_finite()
            || state.accumulated_sliding_distance_m < 0.0
            || !state.frictional_work_j_per_m2.is_finite()
            || state.frictional_work_j_per_m2 < 0.0
        {
            return Err(ContactPointError::InvalidHistory);
        }
        let q = state.tangential_traction_pa[0].hypot(state.tangential_traction_pa[1]);
        let slip = state.inelastic_slip_m[0].hypot(state.inelastic_slip_m[1]);
        let pressure = if state.gap_m < 0.0 {
            -self.policy.normal_stiffness_pa_per_m * state.gap_m
        } else {
            0.0
        };
        let capacity = self.coefficient.unwrap_or(0.0) * pressure;
        if !q.is_finite()
            || !slip.is_finite()
            || !pressure.is_finite()
            || !capacity.is_finite()
            || q > capacity + 128.0 * f64::EPSILON * capacity.max(q)
            || slip
                > state.accumulated_sliding_distance_m
                    + 128.0 * f64::EPSILON * slip.max(state.accumulated_sliding_distance_m)
            || (state.gap_m >= 0.0 && q != 0.0)
            || (state.accumulated_sliding_distance_m == 0.0
                && state.frictional_work_j_per_m2 != 0.0)
            || (self.coefficient == Some(0.0) && state.frictional_work_j_per_m2 != 0.0)
            || (self.coefficient.is_none()
                && (q != 0.0
                    || slip != 0.0
                    || state.accumulated_sliding_distance_m != 0.0
                    || state.frictional_work_j_per_m2 != 0.0))
        {
            return Err(ContactPointError::InvalidHistory);
        }
        Ok(())
    }
}
