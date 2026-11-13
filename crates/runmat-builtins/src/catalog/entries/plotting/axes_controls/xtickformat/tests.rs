use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(XTICKFORMAT_CATALOG_ENTRY.identity.name, "xtickformat");
    assert!(matches!(
        XTICKFORMAT_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(XTICKFORMAT_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == XTICKFORMAT_ERROR_TOO_MANY_OUTPUTS.identifier));
}
