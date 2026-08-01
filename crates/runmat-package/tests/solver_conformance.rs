use runmat_package::{
    acquire_candidates_with_policy, resolve, CandidateIndex, CandidateMetadata, CandidateProvider,
    CandidateQuery, CanonicalPackageId, ContentDigest, DependencyGroup, HostCapability,
    PackageInstanceId, PathSourceId, ResolutionRequest, ResolutionRequirement, ResolveError,
    SourceId, SourceSelectionPolicy, TargetEnvironment, TargetPredicate,
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
        allowed_capabilities: [HostCapability::Worker].into_iter().collect(),
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
