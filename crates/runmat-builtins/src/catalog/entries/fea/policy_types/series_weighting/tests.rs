use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn series_weighting_identities_conform() {
    assert_family(&[
        (&uniform_samples::FEA_SERIES_WEIGHTING_UNIFORM_SAMPLES_CATALOG_ENTRY, &uniform_samples::POLICY_CALL),
        (&trapezoidal_independent_domain::FEA_SERIES_WEIGHTING_TRAPEZOIDAL_INDEPENDENT_DOMAIN_CATALOG_ENTRY, &trapezoidal_independent_domain::POLICY_CALL),
    ]);
}
