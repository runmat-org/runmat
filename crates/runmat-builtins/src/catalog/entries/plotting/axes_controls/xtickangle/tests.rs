use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(XTICKANGLE_CATALOG_ENTRY.identity.name, "xtickangle");
    assert!(matches!(
        XTICKANGLE_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(XTICKANGLE_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == XTICKANGLE_ERROR_TOO_MANY_OUTPUTS.identifier));
}
