use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn solver_options_identities_conform() {
    assert_family(&[(
        &defaults::FEA_SOLVER_OPTIONS_DEFAULTS_CATALOG_ENTRY,
        &defaults::POLICY_CALL,
    )]);
}
