use super::*;

#[test]
fn contract_is_identity_owned_and_output_bounded() {
    assert_eq!(CLIM_CATALOG_ENTRY.identity.name, "clim");
    assert!(matches!(
        CLIM_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert_eq!(CLIM_DESCRIPTOR.signatures.len(), 6);
    assert!(CLIM_DESCRIPTOR
        .signatures
        .iter()
        .any(|form| form.label == "limits = clim(axes)"));
    assert!(CLIM_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == CLIM_ERROR_TOO_MANY_OUTPUTS.identifier));
}
