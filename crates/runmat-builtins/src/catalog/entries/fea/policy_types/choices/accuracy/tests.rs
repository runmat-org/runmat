use crate::fea_contract::policy_types::conformance::assert_family;
use crate::BuiltinCatalogEntry;

const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &super::contribution_geometry::geometry_admission::CATALOG_ENTRY,
    &super::contribution_geometry::boundary_approximation::CATALOG_ENTRY,
    &super::contribution_geometry::spatial_discretization::CATALOG_ENTRY,
    &super::contribution_solution::algebraic::CATALOG_ENTRY,
    &super::contribution_solution::nonlinear::CATALOG_ENTRY,
    &super::contribution_solution::temporal::CATALOG_ENTRY,
    &super::contribution_solution::frequency_sampling::CATALOG_ENTRY,
    &super::contribution_solution::modal_truncation::CATALOG_ENTRY,
    &super::contribution_coupling::stochastic::CATALOG_ENTRY,
    &super::contribution_coupling::coupling::CATALOG_ENTRY,
    &super::contribution_coupling::interface_projection::CATALOG_ENTRY,
    &super::contribution_coupling::field_transfer::CATALOG_ENTRY,
    &super::contribution_coupling::signal_interpolation::CATALOG_ENTRY,
    &super::contribution_output::recovery::CATALOG_ENTRY,
    &super::contribution_output::observation_sampling::CATALOG_ENTRY,
    &super::contribution_output::precision::CATALOG_ENTRY,
    &super::contribution_output::execution_equivalence::CATALOG_ENTRY,
    &super::evidence_estimates::a_posteriori_estimator::CATALOG_ENTRY,
    &super::evidence_estimates::convergence_study::CATALOG_ENTRY,
    &super::evidence_estimates::conservation_balance::CATALOG_ENTRY,
    &super::evidence_estimates::residual_bound::CATALOG_ENTRY,
    &super::evidence_references::reference_comparison::CATALOG_ENTRY,
    &super::evidence_references::statistical_confidence::CATALOG_ENTRY,
    &super::evidence_references::transfer_bound::CATALOG_ENTRY,
    &super::unavailable_evidence::fail::CATALOG_ENTRY,
    &super::unavailable_evidence::mark_indeterminate::CATALOG_ENTRY,
    &super::limit_action::refine::CATALOG_ENTRY,
    &super::limit_action::tighten_solver::CATALOG_ENTRY,
    &super::limit_action::extend_sampling::CATALOG_ENTRY,
    &super::limit_action::fail::CATALOG_ENTRY,
];

#[test]
fn accuracy_policy_choices_conform() {
    assert_family(&[
        (
            &super::contribution_geometry::geometry_admission::CATALOG_ENTRY,
            &super::contribution_geometry::geometry_admission::POLICY_CALL,
        ),
        (
            &super::contribution_geometry::boundary_approximation::CATALOG_ENTRY,
            &super::contribution_geometry::boundary_approximation::POLICY_CALL,
        ),
        (
            &super::contribution_geometry::spatial_discretization::CATALOG_ENTRY,
            &super::contribution_geometry::spatial_discretization::POLICY_CALL,
        ),
        (
            &super::contribution_solution::algebraic::CATALOG_ENTRY,
            &super::contribution_solution::algebraic::POLICY_CALL,
        ),
        (
            &super::contribution_solution::nonlinear::CATALOG_ENTRY,
            &super::contribution_solution::nonlinear::POLICY_CALL,
        ),
        (
            &super::contribution_solution::temporal::CATALOG_ENTRY,
            &super::contribution_solution::temporal::POLICY_CALL,
        ),
        (
            &super::contribution_solution::frequency_sampling::CATALOG_ENTRY,
            &super::contribution_solution::frequency_sampling::POLICY_CALL,
        ),
        (
            &super::contribution_solution::modal_truncation::CATALOG_ENTRY,
            &super::contribution_solution::modal_truncation::POLICY_CALL,
        ),
        (
            &super::contribution_coupling::stochastic::CATALOG_ENTRY,
            &super::contribution_coupling::stochastic::POLICY_CALL,
        ),
        (
            &super::contribution_coupling::coupling::CATALOG_ENTRY,
            &super::contribution_coupling::coupling::POLICY_CALL,
        ),
        (
            &super::contribution_coupling::interface_projection::CATALOG_ENTRY,
            &super::contribution_coupling::interface_projection::POLICY_CALL,
        ),
        (
            &super::contribution_coupling::field_transfer::CATALOG_ENTRY,
            &super::contribution_coupling::field_transfer::POLICY_CALL,
        ),
        (
            &super::contribution_coupling::signal_interpolation::CATALOG_ENTRY,
            &super::contribution_coupling::signal_interpolation::POLICY_CALL,
        ),
        (
            &super::contribution_output::recovery::CATALOG_ENTRY,
            &super::contribution_output::recovery::POLICY_CALL,
        ),
        (
            &super::contribution_output::observation_sampling::CATALOG_ENTRY,
            &super::contribution_output::observation_sampling::POLICY_CALL,
        ),
        (
            &super::contribution_output::precision::CATALOG_ENTRY,
            &super::contribution_output::precision::POLICY_CALL,
        ),
        (
            &super::contribution_output::execution_equivalence::CATALOG_ENTRY,
            &super::contribution_output::execution_equivalence::POLICY_CALL,
        ),
        (
            &super::evidence_estimates::a_posteriori_estimator::CATALOG_ENTRY,
            &super::evidence_estimates::a_posteriori_estimator::POLICY_CALL,
        ),
        (
            &super::evidence_estimates::convergence_study::CATALOG_ENTRY,
            &super::evidence_estimates::convergence_study::POLICY_CALL,
        ),
        (
            &super::evidence_estimates::conservation_balance::CATALOG_ENTRY,
            &super::evidence_estimates::conservation_balance::POLICY_CALL,
        ),
        (
            &super::evidence_estimates::residual_bound::CATALOG_ENTRY,
            &super::evidence_estimates::residual_bound::POLICY_CALL,
        ),
        (
            &super::evidence_references::reference_comparison::CATALOG_ENTRY,
            &super::evidence_references::reference_comparison::POLICY_CALL,
        ),
        (
            &super::evidence_references::statistical_confidence::CATALOG_ENTRY,
            &super::evidence_references::statistical_confidence::POLICY_CALL,
        ),
        (
            &super::evidence_references::transfer_bound::CATALOG_ENTRY,
            &super::evidence_references::transfer_bound::POLICY_CALL,
        ),
        (
            &super::unavailable_evidence::fail::CATALOG_ENTRY,
            &super::unavailable_evidence::fail::POLICY_CALL,
        ),
        (
            &super::unavailable_evidence::mark_indeterminate::CATALOG_ENTRY,
            &super::unavailable_evidence::mark_indeterminate::POLICY_CALL,
        ),
        (
            &super::limit_action::refine::CATALOG_ENTRY,
            &super::limit_action::refine::POLICY_CALL,
        ),
        (
            &super::limit_action::tighten_solver::CATALOG_ENTRY,
            &super::limit_action::tighten_solver::POLICY_CALL,
        ),
        (
            &super::limit_action::extend_sampling::CATALOG_ENTRY,
            &super::limit_action::extend_sampling::POLICY_CALL,
        ),
        (
            &super::limit_action::fail::CATALOG_ENTRY,
            &super::limit_action::fail::POLICY_CALL,
        ),
    ]);
}

#[test]
fn family_entries_are_valid_documented_and_registered() {
    let errors = crate::catalog::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:#?}");
    let mut registered = Vec::new();
    super::extend_entries(&mut registered);
    assert_eq!(registered.len(), ENTRIES.len());
    for entry in ENTRIES {
        let documentation = &entry.documentation;
        let name = entry.identity.name;
        assert_eq!(documentation.title, Some(name));
        let headings: Vec<_> = documentation
            .sections
            .iter()
            .map(|section| section.heading)
            .collect();
        assert!(
            headings.contains(&"Behavior") && headings.contains(&"Use in a study"),
            "{name}"
        );
        assert!(
            !documentation.faqs.is_empty() || !documentation.examples.is_empty(),
            "{name}"
        );
        assert!(
            !documentation.related.contains(&name),
            "{name} lists itself as related"
        );
        for example in documentation.examples {
            assert!(
                example.program.contains(name),
                "{name} example must call it"
            );
            assert!(matches!(
                example.verification,
                crate::BuiltinExampleVerification::Assertions { .. }
            ));
        }
        assert!(registered
            .iter()
            .any(|candidate| std::ptr::eq(*candidate, *entry) || candidate.identity.name == name));
    }
}
