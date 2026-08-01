use runmat_package::{
    resolve, CandidateIndex, CandidateMetadata, CanonicalPackageId, ContentDigest, DependencyGroup,
    HostCapability, PackageInstanceId, PathSourceId, ResolutionRequest, ResolutionRequirement,
    SourceId, TargetEnvironment,
};
use semver::{Version, VersionReq};
use std::collections::{BTreeMap, BTreeSet};

fn corpus_digest() -> String {
    let package: CanonicalPackageId = "default:runmat/cross-host".parse().unwrap();
    let tree = ContentDigest::sha256("cross-host tree");
    let candidate = CandidateMetadata {
        instance: PackageInstanceId::new(
            package.clone(),
            SourceId::Path(PathSourceId {
                workspace_path: "deps/cross-host".parse().unwrap(),
                manifest_digest: ContentDigest::sha256("cross-host manifest"),
                tree_digest: tree.clone(),
            }),
            Some("1.2.3".parse().unwrap()),
            tree,
        ),
        dependencies: Vec::new(),
        features: BTreeMap::from([("default".to_string(), BTreeSet::new())]),
        required_capabilities: BTreeSet::new(),
        runmat_version: Some(VersionReq::parse("^0.6").unwrap()),
        singleton: false,
        yanked: false,
        available_offline: true,
        target_artifacts: BTreeSet::new(),
        registry_metadata: None,
    };
    let mut index = CandidateIndex::default();
    index.insert(candidate);
    let resolution = resolve(
        &ResolutionRequest {
            root: "cross-host-root".to_string(),
            requirements: vec![ResolutionRequirement {
                alias: "cross-host".parse().unwrap(),
                package,
                version: VersionReq::parse("^1").unwrap(),
                group: DependencyGroup::Runtime,
                optional: false,
                default_features: true,
                features: BTreeSet::new(),
                target: None,
            }],
            groups: [DependencyGroup::Runtime].into_iter().collect(),
            root_features: BTreeSet::new(),
            environment: TargetEnvironment {
                triple: "wasm32-unknown-unknown".to_string(),
                capabilities: [HostCapability::Worker].into_iter().collect(),
            },
            runmat_version: Version::new(0, 6, 1),
            offline: true,
            locked_instances: BTreeSet::new(),
            update_packages: None,
        },
        &index,
    )
    .unwrap();
    ContentDigest::sha256(serde_json::to_vec(&resolution).unwrap()).to_string()
}

fn assert_cross_host_corpus() {
    assert_eq!(
        corpus_digest(),
        "sha256:707f9fa5671aeae3be8f5184d4d253dfeceb26df14b35dfa2066199ef5106776"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_solver_corpus_matches_golden_digest() {
    assert_cross_host_corpus();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn wasm_solver_corpus_matches_native_golden_digest() {
    assert_cross_host_corpus();
}
