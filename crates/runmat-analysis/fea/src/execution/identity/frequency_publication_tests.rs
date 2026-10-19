use super::*;

#[test]
fn frequency_publication_identity_uses_its_contract_budget_and_preserves_existing_ids() {
    let limits = super::super::frequency_publication::FREQUENCY_PUBLICATION_LIMITS;
    let small = vec![1_u64, 2, 3];
    assert_eq!(
        StructuralFrequencyPublicationIdentity::derive("structural-frequency-publication", &small)
            .unwrap(),
        StructuralFrequencyPublicationIdentity::derive_publication(&small, limits).unwrap(),
    );

    // A bounded publication can exceed the compact task-identity budget without loading
    // numerical vectors. Large retained metadata must still bind every inventory entry.
    let mut inventory = vec!["a".repeat(64); 20_000];
    assert!(StructuralFrequencyPublicationIdentity::derive(
        "structural-frequency-publication",
        &inventory,
    )
    .is_err());
    let original =
        StructuralFrequencyPublicationIdentity::derive_publication(&inventory, limits).unwrap();
    inventory[10_000].push('b');
    assert_ne!(
        original,
        StructuralFrequencyPublicationIdentity::derive_publication(&inventory, limits).unwrap(),
    );
    assert!(StructuralFrequencyPublicationIdentity::derive_publication(
        &inventory,
        IDENTITY_LIMITS,
    )
    .is_err());
}
