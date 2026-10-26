//! Pure penalty/Coulomb point operator in a material-attached orthonormal tangent frame.
//! Trial updates never mutate accepted history. Frame/search derivatives belong to assembly.
use runmat_analysis_core::{ContactFriction, ContactInterfaceModel};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;
const LAW_LIMITS: runmat_canonical_codec::CanonicalLimits =
    runmat_canonical_codec::CanonicalLimits::new(4096, 16, 128, 16);
mod frame;
mod state;
mod update;
pub use state::{ContactPointMode, ContactPointResponse, ContactPointState};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactPenaltyPolicy {
    pub normal_stiffness_pa_per_m: f64,
    /// None for frictionless; strictly positive for Coulomb, including coefficient zero.
    pub tangential_stiffness_pa_per_m: Option<f64>,
}
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ContactPointError {
    #[error("invalid explicit contact penalty controls")]
    InvalidPolicy,
    #[error("invalid Coulomb coefficient")]
    InvalidFriction,
    #[error("contact point history does not match its law or physical invariants")]
    InvalidHistory,
    #[error("contact point update contains nonfinite kinematics")]
    InvalidInput,
    #[error("contact point response is nonfinite")]
    NonFiniteResponse,
    #[error("contact law identity cannot be encoded")]
    IdentityEncoding,
    #[error("contact point update cancelled")]
    Cancelled,
}
#[derive(Clone, Copy, Debug)]
pub struct PenaltyContactLaw {
    identity: Digest,
    policy: ContactPenaltyPolicy,
    coefficient: Option<f64>,
}
impl PenaltyContactLaw {
    pub fn derive(
        model: &ContactInterfaceModel,
        policy: ContactPenaltyPolicy,
    ) -> Result<Self, ContactPointError> {
        if crate::is_fea_cancelled() {
            return Err(ContactPointError::Cancelled);
        }
        if !policy.normal_stiffness_pa_per_m.is_finite() || policy.normal_stiffness_pa_per_m <= 0.0
        {
            return Err(ContactPointError::InvalidPolicy);
        }
        let coefficient = match model.friction {
            ContactFriction::Frictionless => {
                if policy.tangential_stiffness_pa_per_m.is_some() {
                    return Err(ContactPointError::InvalidPolicy);
                }
                None
            }
            ContactFriction::Coulomb { coefficient } => {
                if !coefficient.canonical_value().is_finite() || coefficient.canonical_value() < 0.0
                {
                    return Err(ContactPointError::InvalidFriction);
                }
                if policy
                    .tangential_stiffness_pa_per_m
                    .is_none_or(|k| !k.is_finite() || k <= 0.0)
                {
                    return Err(ContactPointError::InvalidPolicy);
                }
                Some(coefficient.canonical_value())
            }
        };
        let encoded = runmat_canonical_codec::encode_contract(
            b"runmat-fea-contact-penalty-point/v1\0",
            "contact-penalty-point",
            &(1_u16, model, policy),
            LAW_LIMITS,
        )
        .map_err(|_| ContactPointError::IdentityEncoding)?;
        let identity = Digest::sha256(&encoded);
        if crate::is_fea_cancelled() {
            return Err(ContactPointError::Cancelled);
        }
        Ok(Self {
            identity,
            policy,
            coefficient,
        })
    }
    pub(crate) fn construction_resident_bytes(
        model: &ContactInterfaceModel,
        policy: ContactPenaltyPolicy,
    ) -> Result<u64, ContactPointError> {
        let profile =
            runmat_canonical_codec::profile_serialized(&(1_u16, model, policy), LAW_LIMITS)
                .map_err(|_| ContactPointError::IdentityEncoding)?;
        profile
            .materialization_resident_bytes_estimate
            .checked_add(profile.encoded_bytes_upper_bound)
            .ok_or(ContactPointError::IdentityEncoding)
    }
    pub fn policy(&self) -> ContactPenaltyPolicy {
        self.policy
    }
    pub fn identity(&self) -> Digest {
        self.identity
    }
    /// Stress-free local history, not an equilibrated physical body/initial-penetration verdict.
    pub fn initial_state(&self) -> ContactPointState {
        ContactPointState {
            law_identity: self.identity,
            gap_m: 0.0,
            tangential_traction_pa: [0.0; 2],
            inelastic_slip_m: [0.0; 2],
            accumulated_sliding_distance_m: 0.0,
            frictional_work_j_per_m2: 0.0,
        }
    }
    pub(crate) fn normal_pressure_at_gap(&self, gap_m: f64) -> Result<f64, ContactPointError> {
        if !gap_m.is_finite() {
            return Err(ContactPointError::InvalidInput);
        }
        let pressure = if gap_m >= 0. {
            0.
        } else {
            -self.policy.normal_stiffness_pa_per_m * gap_m
        };
        if !pressure.is_finite() {
            return Err(ContactPointError::NonFiniteResponse);
        }
        Ok(pressure)
    }
    pub fn update(
        &self,
        gap_m: f64,
        relative_tangential_increment_m: [f64; 2],
        committed: &ContactPointState,
    ) -> Result<ContactPointResponse, ContactPointError> {
        if crate::is_fea_cancelled() {
            return Err(ContactPointError::Cancelled);
        }
        self.validate_state(committed)?;
        if !gap_m.is_finite()
            || relative_tangential_increment_m
                .iter()
                .any(|x| !x.is_finite())
        {
            return Err(ContactPointError::InvalidInput);
        }
        update::evaluate(self, gap_m, relative_tangential_increment_m, committed)
    }
}
#[cfg(test)]
mod tests;

pub use frame::ContactPointFrame;
