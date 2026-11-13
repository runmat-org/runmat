use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn missing_mass_correction_identity_conforms() {
    assert_family(&[(
        &missing_mass::FEA_MODAL_TRUNCATION_CORRECTION_MISSING_MASS_CATALOG_ENTRY,
        &missing_mass::POLICY_CALL,
    )]);
}
