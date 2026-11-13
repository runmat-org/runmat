use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(YTICKLABELS_CATALOG_ENTRY.identity.name, "yticklabels");
    assert!(matches!(
        YTICKLABELS_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(YTICKLABELS_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == YTICKLABELS_ERROR_TOO_MANY_OUTPUTS.identifier));
}
