use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn partial_access_policy_identities_conform() {
    assert_family(&[
        (
            &manifest_and_chunks::FEA_PARTIAL_ACCESS_POLICY_MANIFEST_AND_CHUNKS_CATALOG_ENTRY,
            &manifest_and_chunks::POLICY_CALL,
        ),
        (
            &whole_artifact::FEA_PARTIAL_ACCESS_POLICY_WHOLE_ARTIFACT_CATALOG_ENTRY,
            &whole_artifact::POLICY_CALL,
        ),
    ]);
}
