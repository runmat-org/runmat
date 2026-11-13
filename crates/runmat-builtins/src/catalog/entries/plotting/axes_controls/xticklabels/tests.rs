use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(XTICKLABELS_CATALOG_ENTRY.identity.name, "xticklabels");
    assert!(matches!(
        XTICKLABELS_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(XTICKLABELS_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == XTICKLABELS_ERROR_TOO_MANY_OUTPUTS.identifier));
}
