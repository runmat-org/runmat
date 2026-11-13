use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn signal_domain_identities_conform() {
    assert_family(&[
        (
            &time::FEA_SIGNAL_DOMAIN_TIME_CATALOG_ENTRY,
            &time::POLICY_CALL,
        ),
        (
            &frequency::FEA_SIGNAL_DOMAIN_FREQUENCY_CATALOG_ENTRY,
            &frequency::POLICY_CALL,
        ),
        (
            &study_parameter::FEA_SIGNAL_DOMAIN_STUDY_PARAMETER_CATALOG_ENTRY,
            &study_parameter::POLICY_CALL,
        ),
        (
            &study_parameter_with_units::FEA_SIGNAL_DOMAIN_STUDY_PARAMETER_WITH_UNITS_CATALOG_ENTRY,
            &study_parameter_with_units::POLICY_CALL,
        ),
    ]);
}
