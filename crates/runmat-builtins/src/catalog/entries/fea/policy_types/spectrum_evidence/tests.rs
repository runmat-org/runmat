use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn spectrum_evidence_identities_conform() {
    assert_family(&[(
        &bounded::FEA_SPECTRUM_EVIDENCE_BOUNDED_CATALOG_ENTRY,
        &bounded::POLICY_CALL,
    )]);
}
