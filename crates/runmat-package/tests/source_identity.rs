use runmat_package::{
    CanonicalPackageId, ContentDigest, GitCommitId, GitSourceId, NormalizedRelativePath,
    PackageAlias, PackageInstanceId, PackageVersion, PathSourceId, RegistryId, SourceId,
};
use std::str::FromStr;

fn digest(label: &str) -> ContentDigest {
    ContentDigest::sha256(label)
}

#[test]
fn canonical_package_ids_and_aliases_enforce_portable_grammar() {
    let package = CanonicalPackageId::from_str("default:runmat/linear-algebra").unwrap();
    assert_eq!(package.registry(), &RegistryId::default());
    assert_eq!(package.organization(), "runmat");
    assert_eq!(package.name(), "linear-algebra");
    assert_eq!(package.to_string(), "default:runmat/linear-algebra");
    assert_eq!(
        serde_json::from_str::<CanonicalPackageId>(&serde_json::to_string(&package).unwrap())
            .unwrap(),
        package
    );

    assert!(CanonicalPackageId::from_str("runmat/linear-algebra").is_err());
    assert!(CanonicalPackageId::from_str("default:RunMat/linear-algebra").is_err());
    assert!(CanonicalPackageId::from_str("default:runmat/con").is_err());
    assert!(PackageAlias::from_str("helper-lib").is_ok());
    assert!(PackageAlias::from_str("HelperLib").is_err());
}

#[test]
fn digests_are_algorithm_tagged_and_canonical() {
    let digest = digest("stable bytes");
    let encoded = digest.to_string();
    assert!(encoded.starts_with("sha256:"));
    assert_eq!(ContentDigest::from_str(&encoded).unwrap(), digest);
    assert!(ContentDigest::from_str(encoded.to_uppercase().as_str()).is_err());
    assert!(ContentDigest::from_str("sha1:0000").is_err());
}

#[test]
fn relative_paths_are_checkout_independent_and_use_forward_slashes() {
    assert_eq!(
        NormalizedRelativePath::from_str(r#"deps\shared\.\src"#)
            .unwrap()
            .as_str(),
        "deps/shared/src"
    );
    assert_eq!(
        NormalizedRelativePath::from_str("./").unwrap().as_str(),
        "."
    );
    assert!(NormalizedRelativePath::from_str("../outside").is_err());
    assert!(NormalizedRelativePath::from_str("/absolute").is_err());
    assert!(NormalizedRelativePath::from_str(r#"C:\workspace\dep"#).is_err());
}

#[test]
fn git_sources_normalize_transport_details_and_reject_secret_bearing_urls() {
    let source = GitSourceId::new(
        "https://EXAMPLE.com:443/Org/Repo.git/",
        "0123456789abcdef0123456789abcdef01234567"
            .parse::<GitCommitId>()
            .unwrap(),
        "packages/core".parse().unwrap(),
        digest("git tree"),
    )
    .unwrap();
    assert_eq!(
        source.repository.as_str(),
        "https://example.com/Org/Repo.git"
    );
    assert!(GitSourceId::new(
        "https://token@example.com/Org/Repo.git",
        source.commit.clone(),
        ".".parse().unwrap(),
        digest("tree")
    )
    .is_err());
    assert!(GitSourceId::new(
        "https://example.com/Org/Repo.git?token=secret",
        source.commit,
        ".".parse().unwrap(),
        digest("tree")
    )
    .is_err());
}

#[test]
fn package_instance_identity_is_stable_and_serializable() {
    let package = CanonicalPackageId::from_str("default:runmat/demo").unwrap();
    let source = SourceId::Path(PathSourceId {
        workspace_path: "deps/demo".parse().unwrap(),
        manifest_digest: digest("manifest"),
        tree_digest: digest("tree"),
    });
    let first = PackageInstanceId::new(
        package.clone(),
        source.clone(),
        Some("1.2.3".parse::<PackageVersion>().unwrap()),
        digest("tree"),
    );
    let second = PackageInstanceId::new(
        package,
        source,
        Some("1.2.3".parse::<PackageVersion>().unwrap()),
        digest("tree"),
    );
    assert_eq!(first, second);
    assert_eq!(
        serde_json::from_str::<PackageInstanceId>(&serde_json::to_string(&first).unwrap()).unwrap(),
        first
    );
}
