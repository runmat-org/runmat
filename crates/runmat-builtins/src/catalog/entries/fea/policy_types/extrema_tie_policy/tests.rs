use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn extrema_tie_policy_identities_conform() {
    assert_family(&[
        (
            &first_canonical::FEA_EXTREMA_TIE_POLICY_FIRST_CANONICAL_CATALOG_ENTRY,
            &first_canonical::POLICY_CALL,
        ),
        (
            &all_locations::FEA_EXTREMA_TIE_POLICY_ALL_LOCATIONS_CATALOG_ENTRY,
            &all_locations::POLICY_CALL,
        ),
    ]);
}
