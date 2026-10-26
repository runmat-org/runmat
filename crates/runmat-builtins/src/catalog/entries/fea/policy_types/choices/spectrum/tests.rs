use crate::fea_contract::policy_types::conformance::assert_family;
use crate::BuiltinCatalogEntry;

const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &super::response_relative::relative_displacement::CATALOG_ENTRY,
    &super::response_relative::relative_velocity::CATALOG_ENTRY,
    &super::response_relative::relative_acceleration::CATALOG_ENTRY,
    &super::response_absolute::absolute_acceleration::CATALOG_ENTRY,
    &super::response_absolute::pseudo_velocity::CATALOG_ENTRY,
    &super::response_absolute::pseudo_acceleration::CATALOG_ENTRY,
    &super::envelope::positive::CATALOG_ENTRY,
    &super::envelope::negative::CATALOG_ENTRY,
    &super::envelope::maximax::CATALOG_ENTRY,
    &super::interpolation::linear_frequency::CATALOG_ENTRY,
    &super::interpolation::linear_period::CATALOG_ENTRY,
    &super::interpolation::log_log::CATALOG_ENTRY,
    &super::extrapolation::reject::CATALOG_ENTRY,
    &super::extrapolation::hold_endpoint::CATALOG_ENTRY,
    &super::modal_combination::srss::CATALOG_ENTRY,
    &super::modal_combination::complete_quadratic::CATALOG_ENTRY,
    &super::modal_combination::ten_percent::CATALOG_ENTRY,
    &super::modal_combination::absolute_sum::CATALOG_ENTRY,
    &super::directional_combination::srss::CATALOG_ENTRY,
    &super::directional_combination::absolute_sum::CATALOG_ENTRY,
    &super::truncation_correction::none::CATALOG_ENTRY,
    &super::truncation_correction::residual_mode::CATALOG_ENTRY,
    &super::rigid_response::disabled::CATALOG_ENTRY,
];

#[test]
fn spectrum_policy_choices_conform() {
    assert_family(&[
        (
            &super::response_relative::relative_displacement::CATALOG_ENTRY,
            &super::response_relative::relative_displacement::POLICY_CALL,
        ),
        (
            &super::response_relative::relative_velocity::CATALOG_ENTRY,
            &super::response_relative::relative_velocity::POLICY_CALL,
        ),
        (
            &super::response_relative::relative_acceleration::CATALOG_ENTRY,
            &super::response_relative::relative_acceleration::POLICY_CALL,
        ),
        (
            &super::response_absolute::absolute_acceleration::CATALOG_ENTRY,
            &super::response_absolute::absolute_acceleration::POLICY_CALL,
        ),
        (
            &super::response_absolute::pseudo_velocity::CATALOG_ENTRY,
            &super::response_absolute::pseudo_velocity::POLICY_CALL,
        ),
        (
            &super::response_absolute::pseudo_acceleration::CATALOG_ENTRY,
            &super::response_absolute::pseudo_acceleration::POLICY_CALL,
        ),
        (
            &super::envelope::positive::CATALOG_ENTRY,
            &super::envelope::positive::POLICY_CALL,
        ),
        (
            &super::envelope::negative::CATALOG_ENTRY,
            &super::envelope::negative::POLICY_CALL,
        ),
        (
            &super::envelope::maximax::CATALOG_ENTRY,
            &super::envelope::maximax::POLICY_CALL,
        ),
        (
            &super::interpolation::linear_frequency::CATALOG_ENTRY,
            &super::interpolation::linear_frequency::POLICY_CALL,
        ),
        (
            &super::interpolation::linear_period::CATALOG_ENTRY,
            &super::interpolation::linear_period::POLICY_CALL,
        ),
        (
            &super::interpolation::log_log::CATALOG_ENTRY,
            &super::interpolation::log_log::POLICY_CALL,
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
            &super::modal_combination::srss::CATALOG_ENTRY,
            &super::modal_combination::srss::POLICY_CALL,
        ),
        (
            &super::modal_combination::complete_quadratic::CATALOG_ENTRY,
            &super::modal_combination::complete_quadratic::POLICY_CALL,
        ),
        (
            &super::modal_combination::ten_percent::CATALOG_ENTRY,
            &super::modal_combination::ten_percent::POLICY_CALL,
        ),
        (
            &super::modal_combination::absolute_sum::CATALOG_ENTRY,
            &super::modal_combination::absolute_sum::POLICY_CALL,
        ),
        (
            &super::directional_combination::srss::CATALOG_ENTRY,
            &super::directional_combination::srss::POLICY_CALL,
        ),
        (
            &super::directional_combination::absolute_sum::CATALOG_ENTRY,
            &super::directional_combination::absolute_sum::POLICY_CALL,
        ),
        (
            &super::truncation_correction::none::CATALOG_ENTRY,
            &super::truncation_correction::none::POLICY_CALL,
        ),
        (
            &super::truncation_correction::residual_mode::CATALOG_ENTRY,
            &super::truncation_correction::residual_mode::POLICY_CALL,
        ),
        (
            &super::rigid_response::disabled::CATALOG_ENTRY,
            &super::rigid_response::disabled::POLICY_CALL,
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
