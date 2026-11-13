use crate::fea_contract::policy_types::conformance::assert_family;
use crate::BuiltinCatalogEntry;

const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &super::interpolation::left_endpoint::CATALOG_ENTRY,
    &super::interpolation::right_endpoint::CATALOG_ENTRY,
    &super::interpolation::linear::CATALOG_ENTRY,
    &super::extrapolation::reject::CATALOG_ENTRY,
    &super::extrapolation::hold_endpoint::CATALOG_ENTRY,
    &super::extrapolation::zero::CATALOG_ENTRY,
    &super::sample_approach::at::CATALOG_ENTRY,
    &super::sample_approach::left_limit::CATALOG_ENTRY,
    &super::sample_approach::right_limit::CATALOG_ENTRY,
    &super::point_continuity::left::CATALOG_ENTRY,
    &super::point_continuity::right::CATALOG_ENTRY,
];

#[test]
fn signal_policy_choices_conform() {
    assert_family(&[
        (
            &super::interpolation::left_endpoint::CATALOG_ENTRY,
            &super::interpolation::left_endpoint::POLICY_CALL,
        ),
        (
            &super::interpolation::right_endpoint::CATALOG_ENTRY,
            &super::interpolation::right_endpoint::POLICY_CALL,
        ),
        (
            &super::interpolation::linear::CATALOG_ENTRY,
            &super::interpolation::linear::POLICY_CALL,
        ),
        (
            &super::extrapolation::reject::CATALOG_ENTRY,
            &super::extrapolation::reject::POLICY_CALL,
        ),
        (
            &super::extrapolation::hold_endpoint::CATALOG_ENTRY,
            &super::extrapolation::hold_endpoint::POLICY_CALL,
        ),
        (
            &super::extrapolation::zero::CATALOG_ENTRY,
            &super::extrapolation::zero::POLICY_CALL,
        ),
        (
            &super::sample_approach::at::CATALOG_ENTRY,
            &super::sample_approach::at::POLICY_CALL,
        ),
        (
            &super::sample_approach::left_limit::CATALOG_ENTRY,
            &super::sample_approach::left_limit::POLICY_CALL,
        ),
        (
            &super::sample_approach::right_limit::CATALOG_ENTRY,
            &super::sample_approach::right_limit::POLICY_CALL,
        ),
        (
            &super::point_continuity::left::CATALOG_ENTRY,
            &super::point_continuity::left::POLICY_CALL,
        ),
        (
            &super::point_continuity::right::CATALOG_ENTRY,
            &super::point_continuity::right::POLICY_CALL,
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
