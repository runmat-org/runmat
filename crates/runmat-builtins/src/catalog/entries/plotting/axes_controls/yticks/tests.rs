use super::*;

#[test]
fn owned_contract_has_output_limit() {
    assert_eq!(YTICKS_CATALOG_ENTRY.identity.name, "yticks");
    assert!(matches!(
        YTICKS_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert!(YTICKS_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == YTICKS_ERROR_TOO_MANY_OUTPUTS.identifier));
}
