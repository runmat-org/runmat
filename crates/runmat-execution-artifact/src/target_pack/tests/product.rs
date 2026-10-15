//! The shared fixture remains in the parent; these are metadata/byte-closure
//! assertions, not target execution or provider qualification witnesses.
use super::*;

pub(super) fn check_selected_binding(
    manifest: &TargetPackManifest,
    required: &TargetPackRequirements,
    selected: &TargetPackSelection,
    retained: &BTreeMap<ComponentId, Vec<u8>>,
) {
    let verified = manifest.verify_selected_bytes(selected, retained).unwrap();
    assert_eq!(verified.abi_probe().binding().unwrap(), manifest.abi);
    let binding = verified.selected_product_binding().unwrap();
    assert_eq!(binding.pack, manifest.id().unwrap());
    assert_eq!(binding.selection, selected.fingerprint());
    assert_eq!(binding.target, manifest.target);
    assert_eq!(binding.numerical, manifest.numerical);
    assert!(
        binding.services.is_empty(),
        "unused declared clock service is not a runtime grant"
    );
    let mode = NativeProductMode::Selected {
        binding: Box::new(binding.clone()),
    };
    let decoded: NativeProductMode =
        serde_json::from_slice(&serde_json::to_vec(&mode).unwrap()).unwrap();
    assert_eq!(decoded, mode);
    decoded.validate().unwrap();
    assert!(
        serde_json::from_str::<NativeProductMode>("{}").is_err(),
        "missing mode cannot downgrade a selected product"
    );
    assert!(
        serde_json::from_value::<NativeProductMode>(serde_json::json!({"mode":"selected"}))
            .is_err()
    );
    let mut expanded = required.clone();
    expanded
        .authorized_runtime_capabilities
        .0
        .insert(runmat_types::CapabilityRequirement::Network);
    let expanded_selection = manifest.select(&expanded, None).unwrap();
    let expanded_binding = manifest
        .verify_selected_bytes(&expanded_selection, retained)
        .unwrap()
        .selected_product_binding()
        .unwrap();
    assert_ne!(
        expanded_binding.selection, binding.selection,
        "same bytes under a different caller grant have a different selected identity"
    );
    assert_ne!(
        expanded_binding.fingerprint().unwrap(),
        binding.fingerprint().unwrap()
    );
}

pub(super) fn check_probe_rejection() {
    let (mut changed, mut expected, contents) = fixture();
    changed.target.environment = NativeEnvironment::Musl;
    expected.target = changed.target.clone();
    let selected = changed.select(&expected, None).unwrap();
    let retained = selected
        .components()
        .iter()
        .map(|id| (id.clone(), contents[id].clone()))
        .collect();
    assert!(
        changed.verify_selected_bytes(&selected, &retained).is_err(),
        "metadata equality cannot replace the physical probe environment"
    );

    let (mut changed, mut expected, mut contents) = fixture();
    let bytes = contents.get_mut(&id("abi-probe")).unwrap();
    bytes[8] += 1;
    let digest = Digest::sha256(bytes);
    changed
        .components
        .iter_mut()
        .find(|entry| entry.id == id("abi-probe"))
        .unwrap()
        .object
        .digest = digest;
    changed.abi.layout_fingerprint = digest;
    changed.target.target.abi = changed.abi.identity().unwrap();
    expected.abi = changed.abi.clone();
    expected.target = changed.target.clone();
    let selected = changed.select(&expected, None).unwrap();
    let retained = selected
        .components()
        .iter()
        .map(|id| (id.clone(), contents[id].clone()))
        .collect();
    assert!(
        changed.verify_selected_bytes(&selected, &retained).is_err(),
        "a fresh matching checksum cannot authorize an unknown probe schema"
    );
    assert!(
        TargetPackManifest::from_canonical_bytes(br#"{"schema_version":1}"#).is_err(),
        "pack v1 cannot silently omit the v2 probe locator"
    );
}
