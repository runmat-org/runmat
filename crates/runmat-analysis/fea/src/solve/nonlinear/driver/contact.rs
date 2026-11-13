//! Prepared contact inside the existing load controller. This component admission
//! excludes authored preparation, borrowed inputs, transport and measured RSS.
use super::*;
use crate::solve::nonlinear::solid::SolidContactCoupling;
use runmat_execution::Digest;

#[derive(Clone, Copy)]
pub(crate) struct ContactBlockExecution<'a> {
    pub coupling: SolidContactCoupling<'a>,
    pub maximum_retained_output_bytes: u64,
}
impl ContactBlockExecution<'_> {
    pub(super) fn admit_outputs(&self, accepted_cap: usize) -> Result<(), FeaRunError> {
        let overflow = || FeaRunError::InvalidModel("contact output inventory overflow".into());
        let samples = accepted_cap.checked_add(1).ok_or_else(overflow)?;
        let points = self.coupling.operator.point_count();
        let numeric = points
            .checked_mul(2)
            .and_then(|n| n.checked_mul(samples))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(overflow)?;
        let outer = samples
            .checked_mul(2)
            .and_then(|n| n.checked_mul(std::mem::size_of::<Vec<f64>>()))
            .ok_or_else(overflow)?;
        let required = numeric.checked_add(outer).ok_or_else(overflow)?;
        if u64::try_from(required).map_err(|_| overflow())? > self.maximum_retained_output_bytes {
            return Err(FeaRunError::InvalidModel(
                "contact retained point-output admission exceeded".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn controller_scope(&self, body_scope: Digest) -> Result<Digest, FeaRunError> {
        super::continuation::digest(
            b"runmat-contact-solid-load-controller/v1\0",
            &(
                body_scope,
                self.coupling.operator.execution_identity(),
                self.coupling.solve_policy,
                self.coupling.solve_admission,
                self.coupling.maximum_additional_resident_bytes,
                self.coupling.maximum_geometry_trials_per_advance,
                self.coupling.maximum_symbolic_scalar_operations,
                self.maximum_retained_output_bytes,
            ),
        )
    }
}
