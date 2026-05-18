use runmat_config::{
    build_project_source_index, discover_project_manifest_from, load_project_manifest,
    parse_project_manifest_toml, PROJECT_MANIFEST_FILENAME,
};
use std::fs;
use tempfile::TempDir;

fn write_manifest(dir: &std::path::Path, text: &str) -> std::path::PathBuf {
    let path = dir.join(PROJECT_MANIFEST_FILENAME);
    fs::write(&path, text).expect("write manifest");
    path
}

#[test]
fn parses_and_validates_minimal_manifest() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::create_dir_all(tmp.path().join("dep_a")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"
version = "0.1.0"

[sources]
roots = ["src"]

[dependencies]
dep_a = { path = "dep_a" }

[[entrypoints]]
name = "main"
path = "src/main"
"#,
    );

    let loaded = load_project_manifest(&manifest_path).expect("manifest should validate");
    assert_eq!(loaded.package.name, "demo");
    assert_eq!(loaded.sources.roots.len(), 1);
    assert_eq!(loaded.entrypoints.len(), 1);
}

#[test]
fn reports_missing_required_sections() {
    let parsed = parse_project_manifest_toml(
        r#"
[package]
name = ""

[sources]
roots = []
"#,
    )
    .expect("manifest should parse");

    let err = parsed
        .validate(std::path::Path::new("."))
        .expect_err("validation should reject empty package name and empty source roots");
    assert!(err.to_string().contains("[package].name is required"));
    assert!(err.to_string().contains("[sources].roots is required"));
}

#[test]
fn validation_rejects_missing_source_dir() {
    let tmp = TempDir::new().unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/main"
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("missing source root should fail");
    let msg = err.to_string();
    assert!(msg.contains("source root `src`"));
}

#[test]
fn validation_rejects_unsupported_dependency_fields() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[dependencies]
dep_a = { path = "dep_a", git = "https://example.com/repo.git" }

[[entrypoints]]
name = "main"
path = "src/main"
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("unsupported fields should fail");
    assert!(err.to_string().contains("unsupported dependency fields"));
}

#[test]
fn validation_rejects_duplicate_entrypoint_names() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/main"

[[entrypoints]]
name = "main"
module = "app.server"
function = "run"
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("duplicate entrypoint should fail");
    assert!(err.to_string().contains("duplicate entrypoint name `main`"));
}

#[test]
fn validation_accepts_module_function_entrypoint_target() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
"#,
    );
    let loaded = load_project_manifest(&manifest_path).expect("module/function target valid");
    assert_eq!(loaded.entrypoints[0].name, "server");
}

#[test]
fn discover_project_manifest_walks_upward() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    fs::create_dir_all(root.join("nested/deeper")).unwrap();
    let manifest_path = write_manifest(
        root,
        r#"
[package]
name = "demo"

[sources]
roots = []
"#,
    );

    let discovered = discover_project_manifest_from(&root.join("nested/deeper/file.m"))
        .expect("manifest should be discovered");
    assert_eq!(discovered, manifest_path);
}

#[test]
fn source_index_discovers_pkg_class_and_private_layout() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/+pkg/@Point/private")).unwrap();
    fs::create_dir_all(tmp.path().join("src/utils")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    fs::write(
        tmp.path().join("src/+pkg/value.m"),
        "function y=value(); y=1; end",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src/+pkg/@Point/move.m"),
        "function y=move(); y=1; end",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src/+pkg/@Point/private/helper.m"),
        "function y=helper(); y=1; end",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src/utils/local.m"),
        "function y=local(); y=1; end",
    )
    .unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]
"#,
    );
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    let index = build_project_source_index(tmp.path(), &manifest).expect("source index");

    let qualified: std::collections::HashSet<_> = index
        .files
        .iter()
        .map(|file| file.qualified_name.as_str())
        .collect();
    assert!(qualified.contains("main"));
    assert!(qualified.contains("pkg.value"));
    assert!(qualified.contains("pkg.Point.move"));
    assert!(qualified.contains("pkg.Point.helper"));
    assert!(qualified.contains("utils.local"));

    assert!(index
        .package_dirs
        .iter()
        .any(|dir| dir == std::path::Path::new("src/+pkg")));
    assert!(index
        .class_dirs
        .iter()
        .any(|dir| dir == std::path::Path::new("src/+pkg/@Point")));
    assert!(index
        .private_dirs
        .iter()
        .any(|dir| dir == std::path::Path::new("src/+pkg/@Point/private")));
}

#[test]
fn source_index_reports_missing_source_root() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]
"#,
    );
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    fs::remove_dir_all(tmp.path().join("src")).unwrap();
    let err = build_project_source_index(tmp.path(), &manifest)
        .expect_err("missing source root should be reported");
    assert!(err.to_string().contains("source root does not exist"));
}
