use super::*;

#[test]
fn planned_contracts_are_complete_and_self_consistent() {
    for reference in [
        ForeignAdapterContractReference::planned(PlannedForeignAdapter::DotNet),
        ForeignAdapterContractReference::planned(PlannedForeignAdapter::WindowsCom),
    ] {
        reference.validate().unwrap();
        let contract = reference.definition();
        contract.validate().unwrap();
        assert_eq!(contract.adapter, reference.adapter.adapter_id());
        assert_eq!(contract.contract_version, reference.contract_version);
        assert_eq!(contract.delivery, ForeignAdapterDelivery::Planned);
        let bytes = serde_json::to_vec(&contract).unwrap();
        assert_eq!(
            serde_json::from_slice::<ForeignAdapterContract>(&bytes).unwrap(),
            contract
        );
    }
}

#[test]
fn com_contract_is_windows_only_and_requires_apartment_rules() {
    let contract = windows_com_contract();
    assert_eq!(
        contract.platforms,
        vec![ForeignAdapterPlatform::WindowsX86_64]
    );
    assert!(contract
        .threading
        .contains(&ForeignThreadingModel::ComSingleThreadedApartment));
    assert!(contract
        .threading
        .contains(&ForeignThreadingModel::ApartmentMarshalling));
    assert!(contract
        .packaging
        .contains(&ForeignPackagingModel::ExternalRegistrationPrerequisite));
}
