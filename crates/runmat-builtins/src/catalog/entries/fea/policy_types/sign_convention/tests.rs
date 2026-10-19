use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn sign_convention_identities_conform() {
    assert_family(&[
        (
            &native::FEA_SIGN_CONVENTION_NATIVE_CATALOG_ENTRY,
            &native::POLICY_CALL,
        ),
        (
            &tension_positive::FEA_SIGN_CONVENTION_TENSION_POSITIVE_CATALOG_ENTRY,
            &tension_positive::POLICY_CALL,
        ),
        (
            &compression_positive::FEA_SIGN_CONVENTION_COMPRESSION_POSITIVE_CATALOG_ENTRY,
            &compression_positive::POLICY_CALL,
        ),
        (
            &outward_positive::FEA_SIGN_CONVENTION_OUTWARD_POSITIVE_CATALOG_ENTRY,
            &outward_positive::POLICY_CALL,
        ),
        (
            &inward_positive::FEA_SIGN_CONVENTION_INWARD_POSITIVE_CATALOG_ENTRY,
            &inward_positive::POLICY_CALL,
        ),
    ]);
}
