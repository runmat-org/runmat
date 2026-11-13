use super::*;

#[test]
fn contract_has_owned_identity_and_bounded_outputs() {
    assert_eq!(COLORORDER_CATALOG_ENTRY.identity.name, "colororder");
    assert!(matches!(
        COLORORDER_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(COLORORDER_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == COLORORDER_ERROR_TOO_MANY_OUTPUTS.identifier));
}
