use super::*;

#[test]
fn contract_is_identity_owned_and_output_bounded() {
    assert_eq!(CAXIS_CATALOG_ENTRY.identity.name, "caxis");
    assert!(matches!(
        CAXIS_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert_eq!(CAXIS_DESCRIPTOR.signatures.len(), 6);
    assert!(CAXIS_DESCRIPTOR
        .signatures
        .iter()
        .any(|form| form.label == "limits = caxis(axes)"));
    assert!(CAXIS_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == CAXIS_ERROR_TOO_MANY_OUTPUTS.identifier));
}
