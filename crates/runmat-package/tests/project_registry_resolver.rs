use futures::executor::block_on;
use runmat_package::{
    resolve_project_async, CanonicalPackageId, ContentDigest, DependencyGroup, GitAcquisitionPlan,
    GitPackageMount, PackageSourceProvider, PackageVersion, ProjectResolveOptions,
    RegistryAcquisitionPlan, RegistryCandidatePlan, RegistryCandidateRecord, RegistryOrigin,
    RegistryPackageMount, RegistryReleaseId, RegistryReleaseMetadata, RegistrySourceId,
    SourceAcquisitionIntent, SourceAcquisitionPolicy, SourceId,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex;
use tempfile::TempDir;

struct FixtureRegistry {
    releases: Vec<(RegistryCandidateRecord, PathBuf)>,
    candidate_plans: Mutex<Vec<RegistryCandidatePlan>>,
    acquisition_plans: Mutex<Vec<RegistryAcquisitionPlan>>,
}

impl PackageSourceProvider for FixtureRegistry {
    fn acquire_git<'a>(
        &'a self,
        _plan: &'a GitAcquisitionPlan,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<GitPackageMount, String>> + 'a>>
    {
        Box::pin(async { Err("Git is not configured in this fixture".to_string()) })
    }

    fn acquire_registry<'a>(
        &'a self,
        plan: &'a RegistryAcquisitionPlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<RegistryPackageMount, String>> + 'a>,
    > {
        Box::pin(async move {
            self.acquisition_plans.lock().unwrap().push(plan.clone());
            let expected = plan
                .expected
                .as_ref()
                .ok_or_else(|| "resolver must acquire an exact selected release".to_string())?;
            let (record, root) = self
                .releases
                .iter()
                .find(|(record, _)| &record.source == expected)
                .ok_or_else(|| "selected release is absent from the fixture".to_string())?;
            Ok(RegistryPackageMount {
                source: record.source.clone(),
                root: root.clone(),
                metadata: Some(record.metadata.clone()),
            })
        })
    }

    fn registry_candidates<'a>(
        &'a self,
        plan: &'a RegistryCandidatePlan,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Vec<RegistryCandidateRecord>, String>> + 'a>,
    > {
        Box::pin(async move {
            self.candidate_plans.lock().unwrap().push(plan.clone());
            Ok(self
                .releases
                .iter()
                .filter(|(record, _)| record.source.package == plan.package)
                .map(|(record, _)| record.clone())
                .collect())
        })
    }
}

fn options() -> ProjectResolveOptions {
    ProjectResolveOptions {
        target: "x86_64-unknown-linux-gnu".to_string(),
        default_server_origin: "https://api.runmat.test".to_string(),
        default_registry_index: "https://packages.runmat.test".to_string(),
        groups: [DependencyGroup::Runtime].into_iter().collect(),
        root_features: BTreeSet::new(),
        host_capabilities: BTreeSet::new(),
        source_intent: SourceAcquisitionIntent::Execute,
        source_policy: SourceAcquisitionPolicy::default(),
    }
}

#[test]
fn fresh_resolution_selects_exact_non_yanked_release_and_locked_replay_skips_candidates() {
    let temp = TempDir::new().unwrap();
    let root = write_root(&temp, "^1");
    let active = release_fixture(&temp, "1.0.0", false, false);
    let yanked = release_fixture(&temp, "1.1.0", true, false);
    let provider = FixtureRegistry {
        releases: vec![yanked.clone(), active.clone()],
        candidate_plans: Mutex::new(Vec::new()),
        acquisition_plans: Mutex::new(Vec::new()),
    };

    let first = block_on(resolve_project_async(&root, None, options(), &provider)).unwrap();
    let registry_sources = first
        .lock
        .packages
        .iter()
        .filter_map(|package| match &package.instance.source {
            SourceId::Registry(source) => Some(source),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(registry_sources, vec![&active.0.source]);
    assert_eq!(provider.candidate_plans.lock().unwrap().len(), 1);
    let acquisition = provider.acquisition_plans.lock().unwrap();
    assert_eq!(acquisition.len(), 1);
    assert_eq!(acquisition[0].expected.as_ref(), Some(&active.0.source));
    drop(acquisition);

    provider.candidate_plans.lock().unwrap().clear();
    provider.acquisition_plans.lock().unwrap().clear();
    let mut frozen = options();
    frozen.source_policy = SourceAcquisitionPolicy {
        locked: true,
        frozen: true,
        offline: true,
    };
    let replay = block_on(resolve_project_async(
        &root,
        Some(&first.lock),
        frozen,
        &provider,
    ))
    .unwrap();
    assert_eq!(replay.lock, first.lock);
    assert!(provider.candidate_plans.lock().unwrap().is_empty());
    let acquisition = provider.acquisition_plans.lock().unwrap();
    assert_eq!(acquisition.len(), 1);
    assert!(!acquisition[0].allow_network);
    assert_eq!(acquisition[0].expected.as_ref(), Some(&active.0.source));
}

#[test]
fn signed_metadata_must_match_the_materialized_artifact_manifest() {
    let temp = TempDir::new().unwrap();
    let root = write_root(&temp, "=1.0.0");
    let release = release_fixture(&temp, "1.0.0", false, true);
    let provider = FixtureRegistry {
        releases: vec![release],
        candidate_plans: Mutex::new(Vec::new()),
        acquisition_plans: Mutex::new(Vec::new()),
    };

    let error = block_on(resolve_project_async(&root, None, options(), &provider))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("differs from signed metadata"),
        "unexpected error: {error}"
    );
}

fn write_root(temp: &TempDir, requirement: &str) -> PathBuf {
    write_file(temp.path().join("root/src/main.m"), "value = tools();\n");
    write_file(
        temp.path().join("root/runmat.toml"),
        &format!(
            r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
tools = {{ package = "acme/tools", version = "{requirement}" }}
"#
        ),
    );
    temp.path().join("root/runmat.toml")
}

fn release_fixture(
    temp: &TempDir,
    version: &str,
    yanked: bool,
    artifact_singleton: bool,
) -> (RegistryCandidateRecord, PathBuf) {
    let root = temp.path().join(format!("registry/tools-{version}"));
    write_file(
        root.join("src/tools.m"),
        "function value = tools(); value = 42; end\n",
    );
    write_file(
        root.join("runmat.toml"),
        &format!(
            r#"
[package]
name = "tools"
organization = "acme"
version = "{version}"
singleton = {artifact_singleton}

[sources]
roots = ["src"]
"#
        ),
    );
    let package = CanonicalPackageId::from_str("default:acme/tools").unwrap();
    let metadata = RegistryReleaseMetadata {
        singleton: false,
        runmat_requirement: None,
        dependencies: Vec::new(),
        features: Default::default(),
        required_capabilities: Vec::new(),
        optional_capabilities: Vec::new(),
        readme_digest: None,
        license: Some("MIT".to_string()),
        supply_chain: None,
    };
    let mut source = RegistrySourceId {
        registry_origin: RegistryOrigin::new("https://packages.runmat.test").unwrap(),
        package,
        release: RegistryReleaseId::new(format!(
            "rel_{:032x}",
            version
                .bytes()
                .fold(0_u128, |value, byte| value + u128::from(byte))
        ))
        .unwrap(),
        version: PackageVersion::from_str(version).unwrap(),
        release_digest: ContentDigest::sha256("placeholder"),
        artifact_digest: ContentDigest::sha256(format!("artifact-{version}")),
        tree_digest: ContentDigest::sha256(format!("tree-{version}")),
    };
    source.release_digest = metadata.compute_digest(&source).unwrap();
    (
        RegistryCandidateRecord {
            source,
            metadata,
            yanked,
        },
        root,
    )
}

fn write_file(path: impl AsRef<Path>, contents: &str) {
    let path = path.as_ref();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}
