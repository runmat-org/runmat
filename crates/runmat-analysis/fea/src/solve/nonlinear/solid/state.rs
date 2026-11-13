use super::*;
use serde::{Deserialize, Serialize};

/// A converged physical state, not an unconverged guess. Scope binds the constitutive
/// basis, fixed force vector and complete affine equality constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedSolidState {
    pub(crate) scope_identity: Digest,
    pub(crate) load_factor: f64,
    pub(crate) displacement: Vec<f64>,
    pub(crate) constitutive: NonlinearSolidState,
    pub(crate) accumulated_plastic_dissipation_j: f64,
    pub(crate) contact_energy: Option<super::contact::AcceptedContactEnergy>,
    pub(crate) contact: Option<crate::physics::structural::contact::ContactConfigurationState>,
}

impl AcceptedSolidState {
    pub fn contact_energy(&self) -> Option<&super::contact::AcceptedContactEnergy> {
        self.contact_energy.as_ref()
    }
    pub fn contact_configuration(
        &self,
    ) -> Option<&crate::physics::structural::contact::ContactConfigurationState> {
        self.contact.as_ref()
    }
    pub fn scope_identity(&self) -> Digest {
        self.scope_identity
    }

    pub fn basis_identity(&self) -> Digest {
        self.constitutive.basis_identity
    }

    pub fn integration_identity(&self) -> Digest {
        self.constitutive.integration_identity
    }

    pub fn load_factor(&self) -> f64 {
        self.load_factor
    }

    pub fn displacement(&self) -> &[f64] {
        &self.displacement
    }

    pub fn accumulated_plastic_dissipation_j(&self) -> f64 {
        self.accumulated_plastic_dissipation_j
    }

    /// Original ordered point history, never reconstructed from element averages.
    /// Deserialized carriers remain untrusted until validated against their physical scope.
    pub fn point_history(
        &self,
    ) -> impl Iterator<
        Item = (
            runmat_meshing_core::StableDigest,
            usize,
            &crate::physics::structural::plasticity::PlasticityPointState,
        ),
    > {
        self.constitutive
            .point_identities
            .iter()
            .zip(&self.constitutive.points)
            .map(|(identity, history)| {
                (
                    identity.element_identity,
                    identity.quadrature_ordinal,
                    history,
                )
            })
    }
}

impl SolidNewtonProblem<'_> {
    pub(crate) fn initial_state(&self) -> Result<AcceptedSolidState, SolidNewtonError> {
        self.initial_state_record(&mut SolidAttemptWork::default())
    }

    pub(crate) fn initial_state_record(
        &self,
        work: &mut SolidAttemptWork,
    ) -> Result<AcceptedSolidState, SolidNewtonError> {
        let constitutive = self.basis.zero_state();
        let displacement = vec![0.0; self.basis.dof_count()];
        let contact = self.initial_contact(&displacement)?;
        let evaluation = self
            .evaluate_trial(&displacement, &constitutive, contact.as_ref(), 0.0)?
            .evaluation;
        let residual = self.residual(&evaluation, 0.0)?;
        // Fixed nonzero prescribed displacements need their own equilibrated initial
        // transition. Do not invent it by accepting a plastified trial at factor zero.
        if !norm(&residual).is_finite()
            || norm(&residual) != 0.0
            || evaluation.trial_state != constitutive
        {
            return Err(SolidNewtonError::Invalid(
                "nonzero initial lifting requires an equilibrated initial-state transition".into(),
            ));
        }
        self.validate_contact_tangent(&evaluation.tangent, 0., work)?;
        Ok(AcceptedSolidState {
            scope_identity: self.scope_identity,
            load_factor: 0.0,
            displacement,
            constitutive,
            accumulated_plastic_dissipation_j: 0.0,
            contact_energy: contact
                .as_ref()
                .map(|_| super::contact::AcceptedContactEnergy::default()),
            contact,
        })
    }

    pub(crate) fn validate_accepted(
        &self,
        state: &AcceptedSolidState,
    ) -> Result<(), SolidNewtonError> {
        self.validate_accepted_record(state, &mut SolidAttemptWork::default())
    }
    pub(crate) fn validate_accepted_record(
        &self,
        state: &AcceptedSolidState,
        work: &mut SolidAttemptWork,
    ) -> Result<(), SolidNewtonError> {
        if state.scope_identity != self.scope_identity
            || !state.load_factor.is_finite()
            || state.load_factor < 0.0
            || state.displacement.len() != self.basis.dof_count()
            || state.displacement.iter().any(|v| !v.is_finite())
            || !state.accumulated_plastic_dissipation_j.is_finite()
            || state.accumulated_plastic_dissipation_j < 0.0
        {
            return Err(SolidNewtonError::Invalid(
                "foreign/malformed accepted solid state".into(),
            ));
        }
        self.basis
            .validate_state(&state.constitutive)
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        let free = self
            .reduction
            .free_dof_indices
            .iter()
            .map(|i| state.displacement[*i] - state.load_factor * self.prescribed[*i])
            .collect::<Vec<_>>();
        let expanded = self
            .reduction
            .expand(&free)
            .map_err(constraint_error)?
            .into_iter()
            .zip(&self.prescribed)
            .map(|(u, p)| u + state.load_factor * p)
            .collect::<Vec<_>>();
        if expanded.iter().any(|v| !v.is_finite())
            || expanded.iter().zip(&state.displacement).any(|(a, b)| {
                (a - b).abs() > 128.0 * f64::EPSILON * a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
            })
        {
            return Err(SolidNewtonError::Invalid(
                "accepted state violates equality constraints".into(),
            ));
        }
        self.validate_contact_state(state)?;
        match (&self.contact, &state.contact, &state.contact_energy) {
            (None, None, None) => {}
            (Some(contact), Some(configuration), Some(energy)) => {
                energy.validate(configuration.ordinal())?;
                let original_work = contact
                    .input
                    .operator
                    .frictional_work_j(configuration)
                    .map_err(super::contact::contact_error)?;
                let scale = original_work.max(energy.frictional_dissipation_j());
                if (original_work - energy.frictional_dissipation_j()).abs()
                    > 512. * f64::EPSILON * scale
                {
                    return Err(SolidNewtonError::Invalid(
                        "contact friction work differs from original point history".into(),
                    ));
                }
            }
            _ => {
                return Err(SolidNewtonError::Invalid(
                    "missing or foreign accepted contact energy".into(),
                ))
            }
        }
        let trial = self.evaluate_trial(
            &state.displacement,
            &state.constitutive,
            state.contact.as_ref(),
            state.load_factor,
        )?;
        if let (Some(energy), Some(current)) = (&state.contact_energy, &trial.contact_diagnostics) {
            if state.load_factor > 0. {
                energy.validate_current_energy(current)?;
            }
        }
        let evaluation = trial.evaluation;
        let residual = self.residual(&evaluation, state.load_factor)?;
        let scale = norm(
            &self
                .reduction
                .transpose_apply(&self.scale_force)
                .map_err(constraint_error)?,
        ) * state.load_factor;
        let relative_residual = relative_norm(&residual, scale);
        if !scale.is_finite()
            || !relative_residual.is_finite()
            || relative_residual > self.policy.relative_residual_tolerance
        {
            return Err(SolidNewtonError::Invalid(
                "accepted state is not in physical equilibrium".into(),
            ));
        }
        self.validate_contact_tangent(&evaluation.tangent, state.load_factor, work)?;
        self.basis
            .validate_replayed_history(&state.displacement, &state.constitutive)
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        let derived_dissipation = self
            .basis
            .accumulated_plastic_dissipation_j(&state.constitutive)
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        let dissipation_scale = derived_dissipation.max(state.accumulated_plastic_dissipation_j);
        if (derived_dissipation - state.accumulated_plastic_dissipation_j).abs()
            > 512.0 * f64::EPSILON * dissipation_scale
        {
            return Err(SolidNewtonError::Invalid(
                "accepted accumulated dissipation differs from material-owned point history".into(),
            ));
        }
        Ok(())
    }
}
