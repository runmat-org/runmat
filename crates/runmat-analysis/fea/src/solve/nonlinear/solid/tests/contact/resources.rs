use super::*;
#[test]
fn coupled_trial_requested_payload_covers_observed_heap_with_borrowed_sources_excluded() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = CoupledFixture::new(order, true);
        let operator = f.operator(true);
        let basis =
            CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
        let problem = SolidNewtonProblem::prepare_with_contact(
            &basis,
            &f.summary,
            policy(),
            Some(CoupledFixture::coupling(&operator)),
        )
        .unwrap();
        let initial = problem.initial_state().unwrap();
        let u = problem.reduction.expand(&[0.00035, 0.00015]).unwrap();
        let contact = problem.contact.as_ref().unwrap();
        let contact_bytes = super::super::super::contact::resources::additional_peak(
            &operator,
            &contact.pattern,
            &contact.reduced_pattern,
        )
        .unwrap();
        let plan = crate::topology::CanonicalSolidIntegrationPlan::derive(
            &f.surface.mesh,
            crate::topology::CanonicalSolidIntegrationRule::Keast15,
        )
        .unwrap();
        let material =
            crate::assembly::nonlinear_solid::resources::NonlinearSolidStorageDemand::derive(
                &plan,
                f.summary.structural_property_map_identity.as_ref().unwrap(),
                f.summary.dof_count,
                contact.pattern.column_indices().len(),
                1024,
                true,
            )
            .unwrap();
        let (trial, retained, peak) = crate::topology::measure_test_allocations(|| {
            problem
                .evaluate_trial(&u, &initial.constitutive, initial.contact.as_ref(), 1.0)
                .unwrap()
        });
        let bound = material.evaluation_bytes + material.tangent_contribution_bytes + contact_bytes;
        assert!(peak as u64 <= bound, "peak={peak} bound={bound}");
        assert!(retained <= peak);
        assert_eq!(trial.contact.as_ref().unwrap().ordinal(), 1);
        assert_eq!(initial.contact_configuration().unwrap().ordinal(), 0);
    }
}
