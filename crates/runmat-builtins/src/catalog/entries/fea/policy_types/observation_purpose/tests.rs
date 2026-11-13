use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn observation_purpose_identities_conform() {
    assert_family(&[
        (
            &publication::FEA_OBSERVATION_PURPOSE_PUBLICATION_CATALOG_ENTRY,
            &publication::POLICY_CALL,
        ),
        (
            &solver_control::FEA_OBSERVATION_PURPOSE_SOLVER_CONTROL_CATALOG_ENTRY,
            &solver_control::POLICY_CALL,
        ),
        (
            &visualization::FEA_OBSERVATION_PURPOSE_VISUALIZATION_CATALOG_ENTRY,
            &visualization::POLICY_CALL,
        ),
        (
            &downstream_state::FEA_OBSERVATION_PURPOSE_DOWNSTREAM_STATE_CATALOG_ENTRY,
            &downstream_state::POLICY_CALL,
        ),
    ]);
}
