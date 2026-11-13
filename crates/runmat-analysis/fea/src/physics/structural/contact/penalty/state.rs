use super::*;
use runmat_types::physical::SurfaceEnergyDensity;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactPointState {
    pub(super) law_identity: Digest,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub(super) gap_m: Length,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub(super) tangential_traction_pa: Quantity<kind::Traction, [f64; 2]>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub(super) inelastic_slip_m: TangentialMotion,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub(super) accumulated_sliding_distance_m: Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub(super) frictional_work_j_per_m2: SurfaceEnergyDensity,
}
impl ContactPointState {
    pub fn law_identity(&self) -> Digest {
        self.law_identity
    }
    pub fn gap_m(&self) -> Length {
        self.gap_m
    }
    pub fn tangential_traction_pa(&self) -> Quantity<kind::Traction, [f64; 2]> {
        self.tangential_traction_pa
    }
    pub fn inelastic_slip_m(&self) -> TangentialMotion {
        self.inelastic_slip_m
    }
    pub fn accumulated_sliding_distance_m(&self) -> Length {
        self.accumulated_sliding_distance_m
    }
    pub fn frictional_work_j_per_m2(&self) -> SurfaceEnergyDensity {
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
    pub normal_pressure_pa: Pressure,
    /// Internal generalized traction [q1,q2,-p], conjugate to [s1,s2,g].
    /// The physical contact force on the secondary side has the opposite sign.
    pub internal_traction_local_pa: Quantity<kind::Traction, [f64; 3]>,
    /// Derivative wrt current [tangential increment1, increment2, gap],
    /// holding committed history and the material frame fixed. NOT the complete FE tangent.
    pub tangent_pa_per_m: Quantity<kind::TractionDisplacementStiffness, [[f64; 3]; 3]>,
    pub trial_state: ContactPointState,
    pub normal_elastic_energy_j_per_m2: SurfaceEnergyDensity,
    pub tangential_elastic_energy_j_per_m2: SurfaceEnergyDensity,
    /// Backward-Euler point dissipation at current Coulomb capacity; not exact path work.
    pub frictional_dissipation_increment_j_per_m2: SurfaceEnergyDensity,
    /// Backward-Euler elastic update loss; a numerical term, not Coulomb heat.
    pub tangential_algorithmic_dissipation_increment_j_per_m2: SurfaceEnergyDensity,
    /// Tangential spring energy discarded upon separation; not classified as frictional heat.
    pub separation_release_energy_j_per_m2: SurfaceEnergyDensity,
}
impl PenaltyContactLaw {
    pub fn validate_state(&self, state: &ContactPointState) -> Result<(), ContactPointError> {
        let raw = RawPointState::of(state);
        if state.law_identity != self.identity
            || !raw.gap_m.is_finite()
            || raw
                .tangential_traction_pa
                .iter()
                .chain(raw.inelastic_slip_m.iter())
                .any(|x| !x.is_finite())
            || !raw.accumulated_sliding_distance_m.is_finite()
            || raw.accumulated_sliding_distance_m < 0.0
            || !raw.frictional_work_j_per_m2.is_finite()
            || raw.frictional_work_j_per_m2 < 0.0
        {
            return Err(ContactPointError::InvalidHistory);
        }
        let q = raw.tangential_traction_pa[0].hypot(raw.tangential_traction_pa[1]);
        let slip = raw.inelastic_slip_m[0].hypot(raw.inelastic_slip_m[1]);
        let pressure = if raw.gap_m < 0.0 {
            -self.policy.normal_stiffness_pa_per_m.canonical_value() * raw.gap_m
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
                > raw.accumulated_sliding_distance_m
                    + 128.0 * f64::EPSILON * slip.max(raw.accumulated_sliding_distance_m)
            || (raw.gap_m >= 0.0 && q != 0.0)
            || (raw.accumulated_sliding_distance_m == 0.0 && raw.frictional_work_j_per_m2 != 0.0)
            || (self.coefficient == Some(0.0) && raw.frictional_work_j_per_m2 != 0.0)
            || (self.coefficient.is_none()
                && (q != 0.0
                    || slip != 0.0
                    || raw.accumulated_sliding_distance_m != 0.0
                    || raw.frictional_work_j_per_m2 != 0.0))
        {
            return Err(ContactPointError::InvalidHistory);
        }
        Ok(())
    }
}

/// Canonical SI view of one point history for the private penalty/Coulomb kernel.
#[derive(Clone, Copy, Debug)]
pub(super) struct RawPointState {
    pub(super) gap_m: f64,
    pub(super) tangential_traction_pa: [f64; 2],
    pub(super) inelastic_slip_m: [f64; 2],
    pub(super) accumulated_sliding_distance_m: f64,
    pub(super) frictional_work_j_per_m2: f64,
}
impl RawPointState {
    pub(super) fn of(state: &ContactPointState) -> Self {
        Self {
            gap_m: state.gap_m.canonical_value(),
            tangential_traction_pa: state.tangential_traction_pa.canonical_value(),
            inelastic_slip_m: state.inelastic_slip_m.canonical_value(),
            accumulated_sliding_distance_m: state.accumulated_sliding_distance_m.canonical_value(),
            frictional_work_j_per_m2: state.frictional_work_j_per_m2.canonical_value(),
        }
    }
    pub(super) fn into_state(self, law_identity: Digest) -> ContactPointState {
        ContactPointState {
            law_identity,
            gap_m: Length::new(self.gap_m),
            tangential_traction_pa: Quantity::new(self.tangential_traction_pa),
            inelastic_slip_m: Quantity::new(self.inelastic_slip_m),
            accumulated_sliding_distance_m: Length::new(self.accumulated_sliding_distance_m),
            frictional_work_j_per_m2: SurfaceEnergyDensity::new(self.frictional_work_j_per_m2),
        }
    }
}
