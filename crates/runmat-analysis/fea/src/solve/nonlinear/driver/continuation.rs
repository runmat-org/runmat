//! Original controller identity and irreversible accepted history, not restart authority
//! by themselves. The execution layer authenticates an immutable object's expected digest.
use super::super::{AcceptedSolidState, LoadIncrementSchedule, NonlinearSolveOptions};
use crate::FeaRunError;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

/// A continuation carries one accepted history (mesh-sized state). Its stored size is admitted
/// by the nonlinear resource plan (`history_limits`, block transport) and checked by the object
/// inventory, so the codec bounds it only by the wire range. Controller, scope and content
/// identities are digested from the streamed encoding and never materialize it.
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));
const PREFIX: &[u8] = b"runmat-j2-load-continuation-wire/v4\0";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearLoadWork {
    pub(super) accepted_increments: usize,
    pub(super) rejected_increments: usize,
    pub(super) newton_iterations: usize,
    pub(super) linear_solves: usize,
    pub(super) backtracks: usize,
    /// All trial evaluations, including failed cutbacks and accepted-state validation.
    pub(super) tangent_evaluations: usize,
    /// Prefix/equilibrium replay, initial output evaluation and block-end validation.
    /// These real extra block costs are separate from attempted Newton work.
    pub(super) controller_evaluations: usize,
}

impl NonlinearLoadWork {
    pub fn accepted_increments(self) -> usize {
        self.accepted_increments
    }
    pub fn rejected_increments(self) -> usize {
        self.rejected_increments
    }
    pub fn newton_iterations(self) -> usize {
        self.newton_iterations
    }
    pub fn backtracks(self) -> usize {
        self.backtracks
    }
    pub fn linear_solves(self) -> usize {
        self.linear_solves
    }
    pub fn tangent_evaluations(self) -> usize {
        self.tangent_evaluations
    }
    pub fn controller_evaluations(self) -> usize {
        self.controller_evaluations
    }

    pub(crate) fn validate(self, options: &NonlinearSolveOptions) -> Result<(), FeaRunError> {
        self.validate_with_contact(options, false)
    }

    pub(super) fn validate_with_contact(
        self,
        options: &NonlinearSolveOptions,
        contact: bool,
    ) -> Result<(), FeaRunError> {
        let limits = if contact {
            super::super::NonlinearWorkLimits::derive_contact(options)?
        } else {
            super::super::NonlinearWorkLimits::derive(options)?
        };
        if self.accepted_increments > limits.accepted
            || self.rejected_increments > limits.rejected
            || self.newton_iterations > limits.iterations
            || self.linear_solves > limits.linear_solves
            || self.backtracks > limits.backtracks
            || self.tangent_evaluations > limits.tangent_evaluations
            || self.controller_evaluations > limits.controller_evaluations
        {
            return Err(FeaRunError::Solve(
                "global J2 controller work limit exhausted".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearLoadContinuation {
    pub(super) schema_version: u16,
    pub(super) controller_scope: Digest,
    pub(super) content_digest: Digest,
    pub(super) original_start: f64,
    pub(super) original_end: f64,
    pub(super) schedule: LoadIncrementSchedule,
    pub(super) nominal_increment: f64,
    /// Origin of the current pending nominal target, or last completed target.
    pub(super) scheduled_origin: f64,
    pub(super) pending_target: Option<f64>,
    pub(super) accepted_state: AcceptedSolidState,
    pub(super) work: NonlinearLoadWork,
    /// All accepted blocks in the original trajectory, not merely the current suffix.
    pub(super) maximum_relative_residual: f64,
    pub(super) maximum_relative_increment: f64,
    /// Largest achieved relative residual of any completed tangent solve in the trajectory.
    pub(super) maximum_linear_relative_residual: f64,
}

impl NonlinearLoadContinuation {
    /// Integrity transport only. The consumer must additionally authenticate the
    /// immutable object and validate original controller scope and equilibrium.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate_content_digest(self.content_digest)?;
        encode_contract(PREFIX, "j2-load-continuation", self, LIMITS)
            .map_err(|e| FeaRunError::InvalidModel(e.to_string()))
    }

    /// Expected content comes from the authenticated object closure, never from
    /// the payload's own digest. This does not grant numerical restart authority.
    pub fn from_canonical_bytes(
        bytes: &[u8],
        expected_digest: Digest,
    ) -> Result<Self, FeaRunError> {
        let state: Self = decode_contract(PREFIX, "j2-load-continuation", bytes, LIMITS)
            .map_err(|e| FeaRunError::InvalidModel(e.to_string()))?;
        state.validate_content_digest(expected_digest)?;
        Ok(state)
    }

    /// Integrity of the content digest this continuation carries (no encoding is built).
    /// Like `canonical_bytes`, this grants no restart authority by itself.
    pub fn validate_integrity(&self) -> Result<(), FeaRunError> {
        self.validate_content_digest(self.content_digest)
    }

    fn validate_content_digest(&self, expected: Digest) -> Result<(), FeaRunError> {
        if self.schema_version != 4
            || !self.maximum_relative_residual.is_finite()
            || self.maximum_relative_residual < 0.0
            || !self.maximum_relative_increment.is_finite()
            || self.maximum_relative_increment < 0.0
            || !self.maximum_linear_relative_residual.is_finite()
            || self.maximum_linear_relative_residual < 0.0
            || self.content_digest != expected
            || self.derive_digest()? != expected
        {
            return Err(FeaRunError::InvalidModel(
                "J2 continuation content differs from original object".into(),
            ));
        }
        Ok(())
    }

    /// Stored digest remains an untrusted claim until immutable object authentication.
    pub fn content_digest(&self) -> Digest {
        self.content_digest
    }
    pub fn controller_scope(&self) -> Digest {
        self.controller_scope
    }
    pub fn accepted_state(&self) -> &AcceptedSolidState {
        &self.accepted_state
    }
    pub fn work(&self) -> NonlinearLoadWork {
        self.work
    }
    pub fn maximum_relative_residual(&self) -> f64 {
        self.maximum_relative_residual
    }
    pub fn maximum_relative_increment(&self) -> f64 {
        self.maximum_relative_increment
    }
    pub fn maximum_linear_relative_residual(&self) -> f64 {
        self.maximum_linear_relative_residual
    }
    pub fn complete(&self) -> bool {
        self.accepted_state.load_factor().to_bits() == self.original_end.to_bits()
            && self.pending_target.is_none()
    }
    pub fn pending_target(&self) -> Option<f64> {
        self.pending_target
    }
    pub fn nominal_increment(&self) -> f64 {
        self.nominal_increment
    }

    pub(super) fn derive_digest(&self) -> Result<Digest, FeaRunError> {
        digest(
            b"runmat-j2-load-continuation/v4\0",
            &(
                self.schema_version,
                self.controller_scope,
                self.original_start,
                self.original_end,
                &self.schedule,
                self.nominal_increment,
                self.scheduled_origin,
                self.pending_target,
                &self.accepted_state,
                self.work,
                self.maximum_relative_residual,
                self.maximum_relative_increment,
                self.maximum_linear_relative_residual,
            ),
        )
    }

    pub(super) fn validate_for(
        &self,
        expected_digest: Digest,
        controller_scope: Digest,
        start: f64,
        end: f64,
        schedule: &LoadIncrementSchedule,
        options: &NonlinearSolveOptions,
    ) -> Result<(), FeaRunError> {
        let nominal = (end - start) / options.increment_count as f64;
        let current = self.accepted_state.load_factor();
        self.validate_content_digest(expected_digest)?;
        if self.schema_version != 4
            || self.maximum_relative_residual > options.tolerance
            || self.maximum_relative_increment > options.tolerance
            || self.content_digest != expected_digest
            || self.derive_digest()? != expected_digest
            || self.controller_scope != controller_scope
            || self.original_start.to_bits() != start.to_bits()
            || self.original_end.to_bits() != end.to_bits()
            || &self.schedule != schedule
            || self.nominal_increment.to_bits() != nominal.to_bits()
            || !self.scheduled_origin.is_finite()
            || self.scheduled_origin < start
            || !current.is_finite()
            || current <= start
            || current > end
            || self.scheduled_origin > current
            || self.work.accepted_increments == 0
        {
            return Err(FeaRunError::InvalidModel(
                "foreign or malformed J2 controller continuation".into(),
            ));
        }
        match self.pending_target {
            Some(target)
                if target.is_finite()
                    && target > current
                    && target <= end
                    && self.scheduled_origin < current
                    && target.to_bits()
                        == schedule
                            .next_factor(self.scheduled_origin, nominal, end)
                            .to_bits() => {}
            None if self.scheduled_origin.to_bits() == current.to_bits() => {}
            _ => {
                return Err(FeaRunError::InvalidModel(
                    "J2 continuation lost its original pending target".into(),
                ))
            }
        }
        if let Some(contact) = self.accepted_state.contact_configuration() {
            if contact.ordinal()
                != u64::try_from(self.work.accepted_increments).map_err(|_| {
                    FeaRunError::InvalidModel("contact accepted ordinal overflow".into())
                })?
            {
                return Err(FeaRunError::InvalidModel(
                    "contact point/controller accepted ordinals differ".into(),
                ));
            }
        }
        self.work.validate_with_contact(
            options,
            self.accepted_state.contact_configuration().is_some(),
        )
    }
}

pub(super) fn digest(
    value_prefix: &'static [u8],
    value: &impl Serialize,
) -> Result<Digest, FeaRunError> {
    crate::contracts::canonical_digest::canonical_contract_digest(
        value_prefix,
        "j2-load-controller",
        value,
        LIMITS,
    )
    .map_err(|e| FeaRunError::InvalidModel(e.to_string()))
}

pub(super) fn add(a: usize, b: usize) -> Result<usize, FeaRunError> {
    a.checked_add(b)
        .ok_or_else(|| FeaRunError::Solve("J2 work counter overflow".into()))
}
