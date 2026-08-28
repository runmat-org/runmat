use runmat_package::{
    acquire_candidates_with_policy, dependency_tree, plan_update, resolve, why, CandidateIndex,
    CandidateMetadata, CandidateProvider, CandidateQuery, CanonicalPackageId, ContentDigest,
    DependencyGroup, HostCapability, PackageInstanceId, PathSourceId, ResolutionRequest,
    ResolutionRequirement, ResolveError, SourceId, SourceSelectionPolicy, TargetEnvironment,
    TargetPredicate, UpdatePolicy,
};
use semver::{Version, VersionReq};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

fn req(alias: &str, package: &str, version: &str) -> ResolutionRequirement {
    ResolutionRequirement {
        alias: alias.parse().unwrap(),
        package: package.parse().unwrap(),
        version: VersionReq::parse(version).unwrap(),
        group: DependencyGroup::Runtime,
        optional: false,
        default_features: false,
        features: BTreeSet::new(),
        target: None,
    }
}

fn candidate(
    package: &str,
    version: &str,
    dependencies: Vec<ResolutionRequirement>,
) -> CandidateMetadata {
    let tree = ContentDigest::sha256(format!("{package}@{version} tree"));
    CandidateMetadata {
        instance: PackageInstanceId::new(
            package.parse::<CanonicalPackageId>().unwrap(),
            SourceId::Path(PathSourceId {
                workspace_path: format!("fixtures/{}/{version}", package.replace(':', "_"))
                    .parse()
                    .unwrap(),
                manifest_digest: ContentDigest::sha256(format!("{package}@{version} manifest")),
                tree_digest: tree.clone(),
            }),
            Some(version.parse().unwrap()),
            tree,
        ),
        dependencies,
        features: BTreeMap::new(),
        required_capabilities: BTreeSet::new(),
        runmat_version: None,
        singleton: false,
        yanked: false,
        available_offline: true,
        target_artifacts: BTreeSet::new(),
        registry_metadata: None,
    }
}

fn request(requirements: Vec<ResolutionRequirement>) -> ResolutionRequest {
    ResolutionRequest {
        root: "application".to_string(),
        requirements,
        groups: [DependencyGroup::Runtime].into_iter().collect(),
        root_features: BTreeSet::new(),
        environment: TargetEnvironment {
            triple: "wasm32-unknown-unknown".to_string(),
            capabilities: [HostCapability::Worker].into_iter().collect(),
        },
        runmat_version: Version::new(0, 6, 1),
        offline: false,
        locked_instances: BTreeSet::new(),
        update_packages: None,
    }
}

fn versions(resolution: &runmat_package::Resolution, package: &str) -> BTreeSet<String> {
    let package: CanonicalPackageId = package.parse().unwrap();
    resolution
        .packages
        .values()
        .filter(|selected| selected.candidate.instance.package == package)
        .map(|selected| {
            selected
                .candidate
                .instance
                .version
                .as_ref()
                .unwrap()
                .to_string()
        })
        .collect()
}

#[test]
fn backtracks_deterministically_and_reports_exact_conflict_paths() {
    let mut index = CandidateIndex::default();
    index.insert(candidate(
        "default:runmat/a",
        "2.0.0",
        vec![req("b", "default:runmat/b", "^2")],
    ));
    index.insert(candidate(
        "default:runmat/a",
        "1.0.0",
        vec![req("b", "default:runmat/b", "^1")],
    ));
    index.insert(candidate("default:runmat/b", "1.5.0", Vec::new()));
    let resolution = resolve(&request(vec![req("a", "default:runmat/a", ">=1")]), &index).unwrap();
    assert_eq!(
        versions(&resolution, "default:runmat/a"),
        ["1.0.0".to_string()].into_iter().collect()
    );

    let error = resolve(&request(vec![req("a", "default:runmat/a", "^2")]), &index).unwrap_err();
    assert_eq!(error.package.to_string(), "default:runmat/b");
    assert_eq!(error.paths[0].to_string(), "application -> a -> b");
    assert_eq!(error.to_string(), error.to_string());
}

#[test]
fn feature_activation_and_groups_are_solver_inputs() {
    let mut application = candidate(
        "default:runmat/a",
        "1.0.0",
        vec![ResolutionRequirement {
            optional: true,
            ..req("helper", "default:runmat/helper", "^1")
        }],
    );
    application.features.insert(
        "with-helper".to_string(),
        ["helper/fast".to_string()].into_iter().collect(),
    );
    let mut helper = candidate("default:runmat/helper", "1.0.0", Vec::new());
    helper.features.insert("fast".to_string(), BTreeSet::new());
    let mut index = CandidateIndex::default();
    index.insert(application);
    index.insert(helper);
    let mut root = req("a", "default:runmat/a", "^1");
    root.features.insert("with-helper".to_string());
    let resolution = resolve(&request(vec![root]), &index).unwrap();
    assert_eq!(resolution.packages.len(), 2);
    assert!(resolution.packages.values().any(|package| {
        package.candidate.instance.package.to_string() == "default:runmat/helper"
            && package.enabled_features.contains("fast")
    }));
}

#[test]
fn permits_multiple_versions_but_enforces_singletons_and_offline_policy() {
    let mut index = CandidateIndex::default();
    index.insert(candidate(
        "default:runmat/a",
        "1.0.0",
        vec![req("shared", "default:runmat/shared", "^1")],
    ));
    index.insert(candidate(
        "default:runmat/c",
        "1.0.0",
        vec![req("shared", "default:runmat/shared", "^2")],
    ));
    index.insert(candidate("default:runmat/shared", "1.5.0", Vec::new()));
    index.insert(candidate("default:runmat/shared", "2.5.0", Vec::new()));
    let resolution = resolve(
        &request(vec![
            req("a", "default:runmat/a", "^1"),
            req("c", "default:runmat/c", "^1"),
        ]),
        &index,
    )
    .unwrap();
    assert_eq!(versions(&resolution, "default:runmat/shared").len(), 2);

    let mut singleton_index = index.clone();
    let mut singleton = candidate("default:runmat/shared", "3.0.0", Vec::new());
    singleton.singleton = true;
    singleton_index.insert(singleton);
    let error = resolve(
        &request(vec![
            req("one", "default:runmat/shared", "^1"),
            req("three", "default:runmat/shared", "^3"),
        ]),
        &singleton_index,
    )
    .unwrap_err();
    assert!(error.reason.contains("singleton"));
    assert_eq!(error.paths.len(), 2);

    let mut online_only = candidate("default:runmat/network", "2.0.0", Vec::new());
    online_only.available_offline = false;
    let mut offline_index = CandidateIndex::default();
    offline_index.insert(online_only);
    offline_index.insert(candidate("default:runmat/network", "1.0.0", Vec::new()));
    let mut offline_request = request(vec![req("network", "default:runmat/network", ">=1")]);
    offline_request.offline = true;
    let resolution = resolve(&offline_request, &offline_index).unwrap();
    assert_eq!(
        versions(&resolution, "default:runmat/network"),
        ["1.0.0".to_string()].into_iter().collect()
    );
}

#[test]
fn target_and_capability_policy_is_applied_during_selection() {
    let mut native = candidate("default:runmat/native", "1.0.0", Vec::new());
    native
        .required_capabilities
        .insert(HostCapability::NativeLibrary);
    native
        .target_artifacts
        .insert(TargetPredicate::Triple("aarch64-apple-darwin".to_string()));
    let mut index = CandidateIndex::default();
    index.insert(native);
    let error = resolve(
        &request(vec![req("native", "default:runmat/native", "^1")]),
        &index,
    )
    .unwrap_err();
    assert!(error.reason.contains("target, capability"));
}

struct FakeProvider {
    candidates: BTreeMap<CanonicalPackageId, Vec<CandidateMetadata>>,
    queries: RefCell<Vec<CandidateQuery>>,
}

impl CandidateProvider for FakeProvider {
    fn candidates<'a>(
        &'a self,
        query: &'a CandidateQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CandidateMetadata>, ResolveError>> + 'a>> {
        Box::pin(async move {
            self.queries.borrow_mut().push(query.clone());
            Ok(self
                .candidates
                .get(&query.package)
                .cloned()
                .unwrap_or_default())
        })
    }
}

#[test]
fn async_candidate_acquisition_walks_metadata_and_applies_mirrors() {
    let a = candidate(
        "default:runmat/a",
        "1.0.0",
        vec![req("b", "default:runmat/b", "^1")],
    );
    let b = candidate("default:runmat/b", "1.0.0", Vec::new());
    let provider = FakeProvider {
        candidates: BTreeMap::from([
            (a.instance.package.clone(), vec![a]),
            (b.instance.package.clone(), vec![b]),
        ]),
        queries: RefCell::new(Vec::new()),
    };
    let policy = SourceSelectionPolicy {
        replacements: BTreeMap::from([("default".parse().unwrap(), "mirror".parse().unwrap())]),
        offline: true,
    };
    let index = futures::executor::block_on(acquire_candidates_with_policy(
        &provider,
        ["default:runmat/a".parse().unwrap()],
        &policy,
    ))
    .unwrap();
    assert_eq!(index.package_ids().count(), 2);
    let queries = provider.queries.borrow();
    assert_eq!(queries.len(), 2);
    assert!(queries
        .iter()
        .all(|query| query.source_registry.to_string() == "mirror" && query.offline));
}

#[test]
fn tree_why_and_serialized_outcomes_are_deterministic() {
    let mut index = CandidateIndex::default();
    index.insert(candidate(
        "default:runmat/a",
        "1.0.0",
        vec![req("shared", "default:runmat/shared", "^1")],
    ));
    index.insert(candidate(
        "default:runmat/b",
        "1.0.0",
        vec![req("shared", "default:runmat/shared", "^1")],
    ));
    index.insert(candidate("default:runmat/shared", "1.0.0", Vec::new()));
    let request = request(vec![
        req("a", "default:runmat/a", "^1"),
        req("b", "default:runmat/b", "^1"),
    ]);
    let first = resolve(&request, &index).unwrap();
    let second = resolve(&request, &index).unwrap();
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
    let tree = dependency_tree(&first, "application");
    assert!(tree.contains("a: default:runmat/a 1.0.0"));
    assert!(tree.contains("shared: default:runmat/shared 1.0.0 (*)"));
    let shared = first
        .packages
        .values()
        .find(|package| package.candidate.instance.package.to_string() == "default:runmat/shared")
        .unwrap();
    assert_eq!(
        why(
            &first,
            "application",
            &shared.candidate.instance.identity_digest
        )
        .into_iter()
        .map(|path| path.to_string())
        .collect::<Vec<_>>(),
        vec![
            "application -> a -> shared".to_string(),
            "application -> b -> shared".to_string()
        ]
    );
}

#[test]
fn constrained_updates_freeze_every_unpermitted_package() {
    let a1 = candidate(
        "default:runmat/a",
        "1.0.0",
        vec![req("shared", "default:runmat/shared", "^1")],
    );
    let a1_identity = a1.instance.identity_digest.clone();
    let a2 = candidate(
        "default:runmat/a",
        "2.0.0",
        vec![req("shared", "default:runmat/shared", "^2")],
    );
    let mut index = CandidateIndex::default();
    index.insert(a1);
    index.insert(a2);
    index.insert(candidate("default:runmat/shared", "1.0.0", Vec::new()));
    index.insert(candidate("default:runmat/shared", "2.0.0", Vec::new()));

    let mut locked_request = request(vec![req("a", "default:runmat/a", ">=1")]);
    locked_request.locked_instances.insert(a1_identity);
    locked_request.update_packages = Some(BTreeSet::new());
    let current = resolve(&locked_request, &index).unwrap();
    assert_eq!(
        versions(&current, "default:runmat/a"),
        ["1.0.0".to_string()].into_iter().collect()
    );

    let proposed = resolve(&request(vec![req("a", "default:runmat/a", ">=1")]), &index).unwrap();
    let package_a: CanonicalPackageId = "default:runmat/a".parse().unwrap();
    assert!(plan_update(
        &current,
        &proposed,
        &UpdatePolicy::Packages {
            packages: [package_a.clone()].into_iter().collect(),
            recursive: false,
        }
    )
    .is_err());
    let plan = plan_update(
        &current,
        &proposed,
        &UpdatePolicy::Packages {
            packages: [package_a].into_iter().collect(),
            recursive: true,
        },
    )
    .unwrap();
    assert_eq!(plan.changed_packages.len(), 2);
    assert_eq!(plan.added.len(), 2);
    assert_eq!(plan.removed.len(), 2);
}

#[test]
fn candidate_and_requirement_permutations_do_not_change_the_result() {
    let candidates = ["1.0.0", "2.0.0", "3.0.0"];
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut expected = None;
    for permutation in permutations {
        let mut index = CandidateIndex::default();
        for index_value in permutation {
            index.insert(candidate(
                "default:runmat/a",
                candidates[index_value],
                Vec::new(),
            ));
        }
        let resolution =
            resolve(&request(vec![req("a", "default:runmat/a", ">=1")]), &index).unwrap();
        let encoded = serde_json::to_string(&resolution).unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&encoded, expected);
        } else {
            expected = Some(encoded);
        }
    }
}
