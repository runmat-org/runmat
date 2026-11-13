use super::*;

#[test]
fn contract_is_identity_owned_and_output_bounded() {
    assert_eq!(ZLIM_CATALOG_ENTRY.identity.name, "zlim");
    assert!(matches!(
        ZLIM_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert_eq!(ZLIM_DESCRIPTOR.signatures.len(), 9);
    assert!(ZLIM_DESCRIPTOR
        .signatures
        .iter()
        .any(|form| form.label == "limits = zlim(axes)"));
    assert!(ZLIM_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == ZLIM_ERROR_TOO_MANY_OUTPUTS.identifier));
}
