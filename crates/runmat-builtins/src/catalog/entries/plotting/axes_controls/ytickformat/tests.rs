use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(YTICKFORMAT_CATALOG_ENTRY.identity.name, "ytickformat");
    assert!(matches!(
        YTICKFORMAT_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(YTICKFORMAT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == YTICKFORMAT_ERROR_TOO_MANY_OUTPUTS.identifier));
}
