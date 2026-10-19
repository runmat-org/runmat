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
        let encoded = encode_contract(
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
            CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 32),
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
        let contact = coupling
            .map(|input| contact::PreparedSolidContact::prepare(basis, &reduction, input, policy))
            .transpose()?;
        let mut scope_identity = Digest::sha256(encoded);
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
            physical_force: &summary.physical_load,
            scope_identity,
            contact,
            policy,
            solver_admission: None,
        })
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
