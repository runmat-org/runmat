use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(YSCALE_CATALOG_ENTRY.identity.name, "yscale");
    assert!(matches!(
        YSCALE_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(YSCALE_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == YSCALE_ERROR_TOO_MANY_OUTPUTS.identifier));
}
