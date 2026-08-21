use runmat_package::{
    ContentDigest, NormalizedRelativePath, PathSourceId, SourceId, VendorManifest, VendoredPackage,
};

fn package(identity: &str, path: &str) -> VendoredPackage {
    VendoredPackage {
        identity: ContentDigest::sha256(identity),
        source: SourceId::Path(PathSourceId {
            workspace_path: NormalizedRelativePath::new(format!("deps/{identity}")).unwrap(),
            manifest_digest: ContentDigest::sha256(format!("{identity}-manifest")),
            tree_digest: ContentDigest::sha256(format!("{identity}-tree")),
        }),
        path: NormalizedRelativePath::new(path).unwrap(),
    }
}

#[test]
fn vendor_manifest_is_canonical_strict_and_path_unique() {
    let manifest = VendorManifest::new(
        ContentDigest::sha256("lock"),
        vec![
            package("beta", "vendor/beta"),
            package("alpha", "vendor/alpha"),
        ],
    )
    .unwrap();
    assert!(manifest.packages[0].identity < manifest.packages[1].identity);
    let encoded = serde_json::to_vec(&manifest).unwrap();
    assert_eq!(
        serde_json::from_slice::<VendorManifest>(&encoded).unwrap(),
        manifest
    );
    assert!(VendorManifest::new(
        ContentDigest::sha256("lock"),
        vec![
            package("alpha", "vendor/shared"),
            package("beta", "vendor/shared"),
        ],
    )
    .is_err());

    let mut value = serde_json::to_value(&manifest).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_string(), serde_json::Value::Bool(true));
    assert!(serde_json::from_value::<VendorManifest>(value).is_err());
}
