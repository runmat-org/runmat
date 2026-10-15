use super::*;

#[test]
fn contract_has_owned_identity_and_bounded_outputs() {
    assert_eq!(SUBPLOT_CATALOG_ENTRY.identity.name, "subplot");
    assert!(matches!(
        SUBPLOT_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(SUBPLOT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == SUBPLOT_ERROR_TOO_MANY_OUTPUTS.identifier));
}
