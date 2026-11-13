use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn spectrum_axis_identities_conform() {
    assert_family(&[
        (
            &frequency_hertz::FEA_SPECTRUM_AXIS_FREQUENCY_HERTZ_CATALOG_ENTRY,
            &frequency_hertz::POLICY_CALL,
        ),
        (
            &period_seconds::FEA_SPECTRUM_AXIS_PERIOD_SECONDS_CATALOG_ENTRY,
            &period_seconds::POLICY_CALL,
        ),
    ]);
}
