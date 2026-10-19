use super::*;

#[test]
fn contract_has_owned_identity_and_bounded_outputs() {
    assert_eq!(GCA_CATALOG_ENTRY.identity.name, "gca");
    assert!(matches!(
        GCA_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(GCA_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == GCA_ERROR_TOO_MANY_OUTPUTS.identifier));
}
