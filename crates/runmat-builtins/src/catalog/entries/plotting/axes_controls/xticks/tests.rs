use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(XTICKS_CATALOG_ENTRY.identity.name, "xticks");
    assert!(matches!(
        XTICKS_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(XTICKS_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == XTICKS_ERROR_TOO_MANY_OUTPUTS.identifier));
}
