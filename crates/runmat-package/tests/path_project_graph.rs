use futures::executor::block_on;
use runmat_config::project::{
    build_project_composition_graph, build_project_composition_graph_async,
};
use runmat_package::{
    build_project_path_graph, build_project_path_graph_async, encode_lock, DependencyGroup,
    LockSelection, PackageLock,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
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

fn graph(root: &Path, manifest: &Path) -> runmat_package::PackageGraph {
    let composition = build_project_composition_graph(manifest).unwrap();
    build_project_path_graph(&composition, root, BTreeSet::new()).unwrap()
}

#[test]
fn current_path_projects_produce_checkout_independent_graphs() {
    let (first_temp, first_manifest) = fixture();
    let (second_temp, second_manifest) = fixture();
    let first = graph(first_temp.path(), &first_manifest);
    let second = graph(second_temp.path(), &second_manifest);
    assert_eq!(first, second);
    assert_eq!(first.packages.len(), 2);
}

#[test]
fn async_virtual_filesystem_boundary_produces_the_same_graph() {
    let (temp, manifest) = fixture();
    let sync = graph(temp.path(), &manifest);
    let async_graph = block_on(async {
        let composition = build_project_composition_graph_async(&manifest)
            .await
            .unwrap();
        build_project_path_graph_async(&composition, temp.path(), BTreeSet::new())
            .await
            .unwrap()
    });
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
    let (temp, manifest) = fixture();
    let root_manifest = fs::read_to_string(&manifest)
        .unwrap()
        .replace(r#"version = "1.0.0" }"#, r#"version = ">=2.0.0" }"#);
    fs::write(&manifest, root_manifest).unwrap();
    let composition = build_project_composition_graph(&manifest).unwrap();
    let error = build_project_path_graph(&composition, temp.path(), BTreeSet::new()).unwrap_err();
    assert!(error.to_string().contains("requires >=2.0.0"));
}
