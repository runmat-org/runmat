use futures::executor::block_on;
use runmat_package::{
    build_frozen_project, build_frozen_project_async, encode_lock, DependencyGroup,
    FrozenProjectError, LockSelection, PackageLock,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn fixture() -> (TempDir, PathBuf) {
    let temp = TempDir::new().unwrap();
    fs::create_dir_all(temp.path().join("src")).unwrap();
    fs::create_dir_all(temp.path().join("deps/helper/src")).unwrap();
    fs::write(temp.path().join("src/main.m"), "result = helper();\n").unwrap();
    fs::write(
        temp.path().join("deps/helper/src/helper.m"),
        "function y = helper(); y = 42; end\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
helper = { path = "deps/helper", version = "1.0.0" }
"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("deps/helper/runmat.toml"),
        r#"
[package]
name = "helper"
version = "1.0.0"

[sources]
roots = ["src"]
"#,
    )
    .unwrap();
    let manifest = temp.path().join("runmat.toml");
    (temp, manifest)
}

fn graph(manifest: &std::path::Path) -> runmat_package::PackageGraph {
    build_frozen_project(manifest, BTreeSet::new())
        .unwrap()
        .graph
}

#[test]
fn current_path_projects_produce_checkout_independent_graphs() {
    let (_first_temp, first_manifest) = fixture();
    let (_second_temp, second_manifest) = fixture();
    let first = graph(&first_manifest);
    let second = graph(&second_manifest);
    assert_eq!(first, second);
    assert_eq!(first.packages.len(), 2);
}

#[test]
fn async_virtual_filesystem_boundary_produces_the_same_graph() {
    let (_temp, manifest) = fixture();
    let sync = graph(&manifest);
    let async_graph = block_on(build_frozen_project_async(&manifest, BTreeSet::new()))
        .unwrap()
        .graph;
    assert_eq!(async_graph, sync);
    let selection = LockSelection {
        target: "wasm32-unknown-unknown".to_string(),
        groups: [DependencyGroup::Runtime].into_iter().collect(),
        root_features: BTreeSet::new(),
        host_capabilities: BTreeSet::new(),
    };
    assert_eq!(
        encode_lock(&PackageLock::from_graph(&sync, selection.clone()).unwrap()).unwrap(),
        encode_lock(&PackageLock::from_graph(&async_graph, selection).unwrap()).unwrap()
    );
}

#[test]
fn path_dependency_version_assertions_are_enforced() {
    let (_temp, manifest) = fixture();
    let root_manifest = fs::read_to_string(&manifest)
        .unwrap()
        .replace(r#"version = "1.0.0" }"#, r#"version = ">=2.0.0" }"#);
    fs::write(&manifest, root_manifest).unwrap();
    let error = build_frozen_project(&manifest, BTreeSet::new()).unwrap_err();
    assert!(error.to_string().contains("requires >=2.0.0"));
}

#[test]
fn frozen_source_catalogs_are_complete_and_checkout_independent() {
    let (_first_temp, first_manifest) = fixture();
    let (_second_temp, second_manifest) = fixture();
    let first = build_frozen_project(&first_manifest, BTreeSet::new()).unwrap();
    let second = build_frozen_project(&second_manifest, BTreeSet::new()).unwrap();
    assert_eq!(first.graph, second.graph);
    assert_eq!(first.sources, second.sources);
    assert_eq!(first.source_revision(), second.source_revision());
    assert_ne!(first.access_paths, second.access_paths);
    assert_eq!(first.all_sources().count(), 2);
}

#[test]
fn async_frozen_handoff_matches_native_stable_state() {
    let (_temp, manifest) = fixture();
    let sync = build_frozen_project(&manifest, BTreeSet::new()).unwrap();
    let async_project = block_on(build_frozen_project_async(&manifest, BTreeSet::new())).unwrap();
    assert_eq!(sync.graph_digest(), async_project.graph_digest());
    assert_eq!(sync.sources, async_project.sources);
    assert_eq!(sync.access_paths, async_project.access_paths);
}

#[test]
fn missing_dependency_manifest_is_a_package_loader_error() {
    let temp = TempDir::new().unwrap();
    fs::create_dir_all(temp.path().join("src")).unwrap();
    fs::create_dir_all(temp.path().join("deps/missing")).unwrap();
    fs::write(temp.path().join("src/main.m"), "x = 1;\n").unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        r#"
[package]
name = "application"

[sources]
roots = ["src"]

[dependencies]
missing = { path = "deps/missing" }
"#,
    )
    .unwrap();
    let error =
        build_frozen_project(&temp.path().join("runmat.toml"), BTreeSet::new()).unwrap_err();
    assert!(matches!(
        error,
        FrozenProjectError::MissingDependencyManifest { dependency, .. }
            if dependency == "missing"
    ));
}

#[test]
fn equal_declared_names_remain_distinct_path_instances() {
    let temp = TempDir::new().unwrap();
    for directory in ["src", "deps/other/src"] {
        fs::create_dir_all(temp.path().join(directory)).unwrap();
    }
    fs::write(temp.path().join("src/main.m"), "x = other.helper();\n").unwrap();
    fs::write(
        temp.path().join("deps/other/src/helper.m"),
        "function y = helper(); y = 1; end\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        r#"
[package]
name = "shared-name"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
other = { path = "deps/other", version = "2.0.0" }
"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("deps/other/runmat.toml"),
        r#"
[package]
name = "shared-name"
version = "2.0.0"

[sources]
roots = ["src"]
"#,
    )
    .unwrap();

    let frozen = build_frozen_project(&temp.path().join("runmat.toml"), BTreeSet::new()).unwrap();
    assert_eq!(frozen.graph.packages.len(), 2);
    let instances = frozen
        .graph
        .packages
        .values()
        .map(|package| package.instance.identity_digest.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(instances.len(), 2);
}
