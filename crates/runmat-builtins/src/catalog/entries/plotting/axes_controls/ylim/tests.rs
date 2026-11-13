use super::*;

#[test]
fn contract_is_identity_owned_and_output_bounded() {
    assert_eq!(YLIM_CATALOG_ENTRY.identity.name, "ylim");
    assert!(matches!(
        YLIM_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    ));
    assert_eq!(YLIM_DESCRIPTOR.signatures.len(), 9);
    assert!(YLIM_DESCRIPTOR
        .signatures
        .iter()
        .any(|form| form.label == "limits = ylim(axes)"));
    assert!(YLIM_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == YLIM_ERROR_TOO_MANY_OUTPUTS.identifier));
}
