use runmat_package::{CanonicalPackageId, NormalizedRelativePath, PackageVersion, RegistryId};
use runmat_package_cache::{
    ArtifactEntryRole, PublicationEntry, PublicationPolicy, ReleaseArtifactBuilder, ReleaseManifest,
};

#[test]
fn identical_input_builds_byte_identical_release_artifacts_and_manifests() {
    let entries = || {
        vec![
            PublicationEntry::file(
                "src/z.m",
                b"function z; end".to_vec(),
                false,
                ArtifactEntryRole::Source,
            )
            .unwrap(),
            PublicationEntry::file(
                "src/a.m",
                b"function a; end".to_vec(),
                false,
                ArtifactEntryRole::Source,
            )
            .unwrap(),
            PublicationEntry::file(
                "resources/data.csv",
                b"x,y\n1,2\n".to_vec(),
                false,
                ArtifactEntryRole::Resource,
            )
            .unwrap(),
            PublicationEntry::file(
                "README.md",
                b"# Tools\n".to_vec(),
                false,
                ArtifactEntryRole::Resource,
            )
            .unwrap(),
        ]
    };
    let policy = PublicationPolicy::default();
    let first = ReleaseArtifactBuilder::build(entries(), &policy).unwrap();
    let mut reversed = entries();
    reversed.reverse();
    let second = ReleaseArtifactBuilder::build(reversed, &policy).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.inventory.entries[0].path.as_str(), "README.md");

    let package =
        CanonicalPackageId::new("default".parse::<RegistryId>().unwrap(), "acme", "tools").unwrap();
    let manifest = ReleaseManifest::new(
        package,
        "1.2.3".parse::<PackageVersion>().unwrap(),
        Some("^0.6".to_string()),
        &first,
        Some("MIT".to_string()),
        Some(NormalizedRelativePath::new("README.md").unwrap()),
    )
    .unwrap();
    assert_eq!(
        manifest,
        serde_json::from_slice(&manifest.canonical_bytes().unwrap()).unwrap()
    );
    assert_eq!(manifest.digest().unwrap(), manifest.digest().unwrap());
}

#[test]
fn release_manifest_rejects_a_readme_omitted_from_the_artifact() {
    let bundle = ReleaseArtifactBuilder::build(
        vec![PublicationEntry::file(
            "src/tool.m",
            b"function tool; end".to_vec(),
            false,
            ArtifactEntryRole::Source,
        )
        .unwrap()],
        &PublicationPolicy::default(),
    )
    .unwrap();
    let package =
        CanonicalPackageId::new("default".parse::<RegistryId>().unwrap(), "acme", "tools").unwrap();
    let error = ReleaseManifest::new(
        package,
        "1.2.3".parse::<PackageVersion>().unwrap(),
        None,
        &bundle,
        None,
        Some(NormalizedRelativePath::new("README.md").unwrap()),
    )
    .unwrap_err();
    assert!(error.to_string().contains("README"));
}

#[test]
fn publication_policy_excludes_private_build_state_and_gates_native_artifacts() {
    let entries = vec![
        PublicationEntry::file(
            ".git/config",
            b"secret".to_vec(),
            false,
            ArtifactEntryRole::Resource,
        )
        .unwrap(),
        PublicationEntry::file(
            "src/tool.m",
            b"function tool; end".to_vec(),
            false,
            ArtifactEntryRole::Source,
        )
        .unwrap(),
        PublicationEntry::file(
            "bin/tool.mexw64",
            b"native".to_vec(),
            true,
            ArtifactEntryRole::Native,
        )
        .unwrap(),
    ];
    let error =
        ReleaseArtifactBuilder::build(entries.clone(), &PublicationPolicy::default()).unwrap_err();
    assert!(error.to_string().contains("native"));

    let policy = PublicationPolicy::new(
        &["src/**".to_string(), "bin/**".to_string()],
        &["**/*.tmp".to_string()],
        true,
    )
    .unwrap();
    let bundle = ReleaseArtifactBuilder::build(entries, &policy).unwrap();
    assert_eq!(bundle.inventory.entries.len(), 2);
    assert!(bundle
        .inventory
        .entries
        .iter()
        .all(|entry| !entry.path.as_str().starts_with(".git")));

    let policy = PublicationPolicy::new(&["src/**".to_string()], &[], false).expect("valid policy");
    let selected = ReleaseArtifactBuilder::build(
        vec![
            PublicationEntry::file(
                "src/tool.m",
                b"function tool; end".to_vec(),
                false,
                ArtifactEntryRole::Source,
            )
            .unwrap(),
            PublicationEntry::file(
                "bin/tool.mexw64",
                b"native".to_vec(),
                true,
                ArtifactEntryRole::Native,
            )
            .unwrap(),
        ],
        &policy,
    )
    .expect("an excluded native entry does not require native publication permission");
    assert_eq!(selected.inventory.entries.len(), 1);
}

#[test]
fn publication_rejects_duplicate_paths_and_unselected_symlink_targets() {
    let duplicate = PublicationEntry::file(
        "src/tool.m",
        b"a".to_vec(),
        false,
        ArtifactEntryRole::Source,
    )
    .unwrap();
    assert!(ReleaseArtifactBuilder::build(
        vec![duplicate.clone(), duplicate],
        &PublicationPolicy::default()
    )
    .is_err());

    let link = PublicationEntry::symlink(
        "src/current.m",
        "private/target.m",
        ArtifactEntryRole::Source,
    )
    .unwrap();
    assert!(ReleaseArtifactBuilder::build(vec![link], &PublicationPolicy::default()).is_err());
}
