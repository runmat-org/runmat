use super::*;

#[test]
fn contract_is_identity_owned_and_output_bounded() {
    assert_eq!(XLIM_CATALOG_ENTRY.identity.name, "xlim");
    assert!(matches!(
        XLIM_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert_eq!(XLIM_DESCRIPTOR.signatures.len(), 9);
    assert!(XLIM_DESCRIPTOR
        .signatures
        .iter()
        .any(|form| form.label == "limits = xlim(axes)"));
    assert!(XLIM_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == XLIM_ERROR_TOO_MANY_OUTPUTS.identifier));
}
