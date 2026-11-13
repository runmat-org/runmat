//! Resolve local package source ownership from Cargo's workspace metadata.
//! Numerical stage owners seed the closure; unrelated workspace crates do not.
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    process::Command,
};

const ROOTS: &[&str] = &[
    "runmat-meshing-execution",
    "runmat-geometry-io",
    "runmat-geometry-ops",
    "runmat-execution-runner",
    "runmat-filesystem",
    "runmat-cpu-execution",
];

pub fn metadata(workspace: &Path) -> io::Result<Option<Value>> {
    if !workspace
        .join("crates/runmat-meshing/tetrahedron/Cargo.toml")
        .is_file()
    {
        return Ok(None);
    }
    let result = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args([
            "metadata",
            "--format-version=1",
            "--no-deps",
            "--locked",
            "--offline",
            "--manifest-path",
        ])
        .arg(workspace.join("Cargo.toml"))
        .output()?;
    if !result.status.success() {
        return Err(io::Error::other(format!(
            "meshing dependency metadata: {}",
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    serde_json::from_slice(&result.stdout)
        .map(Some)
        .map_err(io::Error::other)
}

/// None disables reuse when the complete local dependency closure is unavailable.
/// Optional/target-specific normal and build dependencies are included
/// conservatively. Registry dependencies remain bound through Cargo.lock.
pub(super) fn roots(workspace: &Path, volume: bool, metadata: &Value) -> Option<Vec<PathBuf>> {
    resolve(workspace, volume, metadata, ROOTS)
}
/// All local package boundaries, including packages outside this stage closure.
/// A parent package must not absorb another package merely through nesting.
pub(super) fn package_roots(workspace: &Path, metadata: &Value) -> Option<BTreeSet<PathBuf>> {
    metadata
        .get("packages")?
        .as_array()?
        .iter()
        .map(|package| {
            let root = Path::new(package.get("manifest_path")?.as_str()?).parent()?;
            root.starts_with(workspace).then(|| root.to_path_buf())
        })
        .collect()
}

fn resolve(
    workspace: &Path,
    volume: bool,
    metadata: &Value,
    roots: &[&str],
) -> Option<Vec<PathBuf>> {
    let packages = metadata.get("packages")?.as_array()?;
    let mut by_path = BTreeMap::new();
    let mut by_name = BTreeMap::new();
    for package in packages {
        let manifest = Path::new(package.get("manifest_path")?.as_str()?);
        let directory = manifest.parent()?.to_path_buf();
        if !directory.starts_with(workspace) {
            return None;
        }
        by_path.insert(directory.clone(), package);
        by_name.insert(package.get("name")?.as_str()?, directory);
    }
    let mut pending = roots
        .iter()
        .map(|name| by_name.get(name).cloned())
        .collect::<Option<Vec<_>>>()?;
    let mut selected = BTreeSet::new();
    while let Some(directory) = pending.pop() {
        let package = by_path.get(&directory)?;
        if !volume && package.get("name")?.as_str()? == "runmat-meshing-tetrahedron" {
            continue;
        }
        if !selected.insert(directory) {
            continue;
        }
        for dependency in package.get("dependencies")?.as_array()? {
            if dependency.get("kind").and_then(Value::as_str) == Some("dev") {
                continue;
            }
            if let Some(path) = dependency.get("path").and_then(Value::as_str) {
                let directory = PathBuf::from(path);
                if !by_path.contains_key(&directory) {
                    return None;
                }
                pending.push(directory);
            }
        }
    }
    // Adapter/storage implementation changes conservatively invalidate the
    // package. Its unrelated physics dependencies are not numerical mesh owners.
    selected.insert(by_name.get("runmat-execution-runner-native")?.clone());
    Some(selected.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn package(root: &Path, name: &str, dependencies: Value) -> Value {
        json!({"name":name,"manifest_path":root.join(name).join("Cargo.toml"),"dependencies":dependencies})
    }
    #[test]
    fn closure_tracks_transitive_build_dependencies_and_excludes_unrelated_physics() {
        let root = Path::new("/workspace");
        let metadata = json!({"packages":[
            package(root,"mesh",json!([{"path":"/workspace/math","kind":null}, {"path":"/workspace/test-only","kind":"dev"}])),
            package(root,"math",json!([{"path":"/workspace/build-helper","kind":"build"}])),
            package(root,"build-helper",json!([])), package(root,"test-only",json!([])),
            package(root,"runmat-execution-runner-native",json!([])), package(root,"unrelated-physics",json!([]))
        ]});
        let result = resolve(root, true, &metadata, &["mesh"]).unwrap();
        assert_eq!(
            result,
            [
                "build-helper",
                "math",
                "mesh",
                "runmat-execution-runner-native"
            ]
            .map(|name| root.join(name))
        );
    }
    #[test]
    fn missing_local_dependencies_and_missing_roots_disable_reuse() {
        let root = Path::new("/workspace");
        let metadata = json!({"packages":[package(root,"mesh",json!([{"path":"/workspace/missing","kind":null}])), package(root,"runmat-execution-runner-native",json!([]))]});
        assert!(resolve(root, true, &metadata, &["mesh"]).is_none());
        assert!(resolve(root, true, &metadata, &["unknown"]).is_none());
    }
    #[test]
    fn tetrahedron_ownership_is_only_in_the_volume_dependency_closure() {
        let root = Path::new("/workspace");
        let metadata = json!({"packages":[package(root,"mesh",json!([{"path":"/workspace/runmat-meshing-tetrahedron","kind":null}])), package(root,"runmat-meshing-tetrahedron",json!([])), package(root,"runmat-execution-runner-native",json!([]))]});
        assert!(!resolve(root, false, &metadata, &["mesh"])
            .unwrap()
            .contains(&root.join("runmat-meshing-tetrahedron")));
        assert!(resolve(root, true, &metadata, &["mesh"])
            .unwrap()
            .contains(&root.join("runmat-meshing-tetrahedron")));
    }
}
