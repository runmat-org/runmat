use super::*;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};

impl<'a> SolidNewtonProblem<'a> {
    pub(crate) fn prepare(
        basis: &'a CanonicalNonlinearSolidBasis,
        summary: &'a AssemblySummary,
        policy: SolidNewtonPolicy,
    ) -> Result<Self, SolidNewtonError> {
        Self::prepare_with_contact(basis, summary, policy, None)
    }
    pub(crate) fn prepare_with_contact(
        basis: &'a CanonicalNonlinearSolidBasis,
        summary: &'a AssemblySummary,
        policy: SolidNewtonPolicy,
        coupling: Option<contact::SolidContactCoupling<'a>>,
    ) -> Result<Self, SolidNewtonError> {
        if crate::is_fea_cancelled() {
            return Err(SolidNewtonError::Cancelled);
        }
        policy.validate()?;
        if basis.dof_count() != summary.dof_count
            || summary.physical_load.len() != summary.dof_count
            || summary.physical_load.iter().any(|v| !v.is_finite())
        {
            return Err(SolidNewtonError::Invalid(
                "force/solid DOF inventory differs".into(),
            ));
        }
        let reduction = policy::constraint_reduction(summary)?;
        // Streamed: the scope holds mesh-sized constraint and load data; no buffer of it is built.
        let mut scope_identity = crate::contracts::canonical_digest::canonical_contract_digest(
            b"runmat-fea-small-strain-solid-equilibrium/v1\0",
            "canonical-solid-fixed-load-scope",
            &(
                basis.identity(),
                &summary.physical_load,
                &reduction,
                &summary.constraint_matrix,
                &summary.spatial_projection_evidence,
                &summary.structural_sources,
                &summary.structural_boundary_transfers,
            ),
            CanonicalLimits::new(0, 0, 65_536, 32),
        )
        .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        if coupling
            .as_ref()
            .is_some_and(|c| c.solve_policy.relative_tolerance != policy.linear_relative_tolerance)
        {
            return Err(SolidNewtonError::Invalid(
                "contact linear target differs from Newton policy".into(),
            ));
        }
        let mut reduction = reduction;
        let prescribed = std::mem::replace(&mut reduction.offset, vec![0.0; summary.dof_count]);
        let contact = coupling
            .map(|input| contact::PreparedSolidContact::prepare(basis, &reduction, input, policy))
            .transpose()?;
        if let Some(contact) = &contact {
            let bytes = encode_contract(
                b"runmat-fea-contact-solid-equilibrium/v1\0",
                "contact-solid-scope",
                &(scope_identity, contact.input.operator.identity()),
                CanonicalLimits::new(4096, 256, 256, 8),
            )
            .map_err(constraint_error)?;
            scope_identity = Digest::sha256(bytes);
        }
        Ok(Self {
            basis,
            reduction,
            prescribed,
            physical_force: &summary.physical_load,
            scale_force: summary.physical_load.clone(),
            follower: None,
            scope_identity,
            contact,
            policy,
            solver_admission: None,
        })
    }

    /// Adds configuration-dependent follower loads. Their reference-configuration force joins
    /// the residual scale and their identity joins the equilibrium scope.
    pub(crate) fn with_follower_loads(
        mut self,
        follower: Option<&'a crate::assembly::nonlinear_solid::FollowerLoadOperator>,
    ) -> Result<Self, SolidNewtonError> {
        let Some(follower) = follower else {
            return Ok(self);
        };
        let reference = self
            .basis
            .evaluate_follower_loads(follower, &vec![0.0; self.basis.dof_count()])
            .map_err(|e| SolidNewtonError::Invalid(e.to_string()))?;
        for (scale, force) in self.scale_force.iter_mut().zip(&reference.force) {
            *scale += force;
        }
        let bytes = encode_contract(
            b"runmat-fea-follower-solid-equilibrium/v1\0",
            "follower-solid-scope",
            &(self.scope_identity, follower.identity()),
            CanonicalLimits::new(4096, 256, 256, 8),
        )
        .map_err(constraint_error)?;
        self.scope_identity = Digest::sha256(bytes);
        self.follower = Some(follower);
        Ok(self)
    }

    pub(crate) fn with_solver_admission(
        mut self,
        admission: Option<&'a crate::planning::resource::SolverAdmissionResourcePolicy>,
    ) -> Self {
        self.solver_admission = admission;
        self
    }

    pub(crate) fn scope_identity(&self) -> Digest {
        self.scope_identity
    }
}
