use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(YTICKANGLE_CATALOG_ENTRY.identity.name, "ytickangle");
    assert!(matches!(
        YTICKANGLE_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(YTICKANGLE_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == YTICKANGLE_ERROR_TOO_MANY_OUTPUTS.identifier));
}
