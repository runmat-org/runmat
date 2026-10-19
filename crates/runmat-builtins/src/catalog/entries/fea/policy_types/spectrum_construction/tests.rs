use super::*;
use crate::fea_contract::policy_types::conformance::assert_family;

#[test]
fn spectrum_construction_identities_conform() {
    assert_family(&[
        (&authored_design::FEA_SPECTRUM_CONSTRUCTION_AUTHORED_DESIGN_CATALOG_ENTRY, &authored_design::POLICY_CALL),
        (&from_acceleration_history_exact::FEA_SPECTRUM_CONSTRUCTION_FROM_ACCELERATION_HISTORY_EXACT_CATALOG_ENTRY, &from_acceleration_history_exact::POLICY_CALL),
        (&from_acceleration_history_recursive::FEA_SPECTRUM_CONSTRUCTION_FROM_ACCELERATION_HISTORY_RECURSIVE_CATALOG_ENTRY, &from_acceleration_history_recursive::POLICY_CALL),
    ]);
}
