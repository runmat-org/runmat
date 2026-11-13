use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(XSCALE_CATALOG_ENTRY.identity.name, "xscale");
    assert!(matches!(
        XSCALE_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(XSCALE_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == XSCALE_ERROR_TOO_MANY_OUTPUTS.identifier));
}
