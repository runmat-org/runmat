use runmat_config::{
    build_project_composition_graph, build_project_source_index, discover_project_manifest_from,
    discover_project_symbols_from, discover_project_symbols_from_source_name,
    load_project_manifest, parse_project_manifest_toml, resolve_named_entrypoint_from,
    resolve_project_entrypoint, ResolvedEntrypointTarget, PROJECT_MANIFEST_FILENAME,
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
fn discover_project_symbols_includes_dependency_alias_qualified_names() {
    let tmp = TempDir::new().unwrap();
    let dep_root = tmp.path().join("deps/statslib");
    fs::create_dir_all(&dep_root).unwrap();
    fs::write(
        tmp.path().join(PROJECT_MANIFEST_FILENAME),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
statsdep = { path = "deps/statslib" }
"#,
    )
    .unwrap();
    fs::write(
        dep_root.join(PROJECT_MANIFEST_FILENAME),
        r#"
[package]
name = "statslib"

[sources]
roots = ["."]
"#,
    )
    .unwrap();
    fs::write(
        dep_root.join("summarize.m"),
        "function y = summarize(x); y = x; end",
    )
    .unwrap();

    let discovered = discover_project_symbols_from(&tmp.path().join("main.m"))
        .expect("discover symbols")
        .expect("symbols should be discovered");

    assert!(discovered.symbols.contains("summarize"));
    assert!(discovered.symbols.contains("statslib.summarize"));
    assert!(discovered.symbols.contains("statsdep.summarize"));
}

#[test]
fn discover_project_symbols_from_source_name_uses_cwd_for_plain_name() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join(PROJECT_MANIFEST_FILENAME),
        r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
    )
    .unwrap();
    fs::write(tmp.path().join("main.m"), "x = 1;").unwrap();

    let discovered = discover_project_symbols_from_source_name("main.m", tmp.path())
        .expect("discover symbols")
        .expect("symbols should be discovered");

    assert_eq!(discovered.root_package, "demo");
    assert!(discovered.symbols.contains("main"));
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

#[test]
fn resolve_project_entrypoint_returns_path_target() {
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
"#,
    );
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    let resolved = resolve_project_entrypoint(tmp.path(), &manifest, "main")
        .expect("resolver should succeed")
        .expect("entrypoint should exist");

    assert_eq!(resolved.target, ResolvedEntrypointTarget::Path);
    assert_eq!(
        resolved.source_file.canonicalize().unwrap(),
        tmp.path().join("src/main.m").canonicalize().unwrap()
    );
}

#[test]
fn resolve_project_entrypoint_returns_module_function_target() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/app")).unwrap();
    fs::write(
        tmp.path().join("src/app/server.m"),
        "function y = main(); y = 1; end",
    )
    .unwrap();
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
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    let resolved = resolve_project_entrypoint(tmp.path(), &manifest, "server")
        .expect("resolver should succeed")
        .expect("entrypoint should exist");

    assert_eq!(resolved.target, ResolvedEntrypointTarget::ModuleFunction);
    assert_eq!(resolved.module.as_deref(), Some("app.server"));
    assert_eq!(resolved.function.as_deref(), Some("main"));
    assert_eq!(
        resolved.source_file.canonicalize().unwrap(),
        tmp.path().join("src/app/server.m").canonicalize().unwrap()
    );
}

#[test]
fn resolve_project_entrypoint_reports_missing_module_target() {
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
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    let err = resolve_project_entrypoint(tmp.path(), &manifest, "server")
        .expect_err("missing module file should return explicit error");
    assert!(err
        .to_string()
        .contains("did not resolve under configured source roots"));
}

#[test]
fn resolve_project_entrypoint_supports_class_folder_module_function_target() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/+pkg/@Point")).unwrap();
    fs::write(
        tmp.path().join("src/+pkg/@Point/move.m"),
        "function obj = move(obj); end",
    )
    .unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "point-move"
module = "pkg.Point"
function = "move"
"#,
    );
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    let resolved = resolve_project_entrypoint(tmp.path(), &manifest, "point-move")
        .expect("resolver should succeed")
        .expect("entrypoint should exist");
    assert_eq!(resolved.target, ResolvedEntrypointTarget::ModuleFunction);
    assert_eq!(
        resolved.source_file.canonicalize().unwrap(),
        tmp.path()
            .join("src/+pkg/@Point/move.m")
            .canonicalize()
            .unwrap()
    );
}

#[test]
fn resolve_project_entrypoint_reports_source_index_failure() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/app")).unwrap();
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
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    fs::remove_dir_all(tmp.path().join("src")).unwrap();
    let err = resolve_project_entrypoint(tmp.path(), &manifest, "server")
        .expect_err("missing source root should bubble source index error");
    assert!(
        err.to_string().contains("failed to resolve entrypoint")
            || err.to_string().contains("source root does not exist")
    );
}

#[test]
fn composition_graph_loads_root_and_local_path_dependency() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::create_dir_all(tmp.path().join("dep_a/src")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    fs::write(
        tmp.path().join("dep_a/src/fn_a.m"),
        "function y = fn_a(); y = 1; end",
    )
    .unwrap();
    write_manifest(
        tmp.path(),
        r#"
[package]
name = "root_pkg"

[sources]
roots = ["src"]

[dependencies]
dep_a = { path = "dep_a" }
"#,
    );
    write_manifest(
        &tmp.path().join("dep_a"),
        r#"
[package]
name = "dep_pkg"

[sources]
roots = ["src"]
"#,
    );

    let graph = build_project_composition_graph(&tmp.path().join("runmat.toml"))
        .expect("composition graph should load");
    assert_eq!(graph.root_package, "root_pkg");
    assert_eq!(graph.packages.len(), 2);
    let root = graph
        .packages
        .get("root_pkg")
        .expect("root package should be present");
    assert_eq!(root.dependencies.get("dep_a"), Some(&"dep_pkg".to_string()));
    let dep = graph
        .packages
        .get("dep_pkg")
        .expect("dependency package should be present");
    assert!(dep
        .source_index
        .files
        .iter()
        .any(|file| file.qualified_name == "fn_a"));
}

#[test]
fn composition_graph_reports_missing_dependency_manifest() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::create_dir_all(tmp.path().join("dep_missing")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    write_manifest(
        tmp.path(),
        r#"
[package]
name = "root_pkg"

[sources]
roots = ["src"]

[dependencies]
dep_missing = { path = "dep_missing" }
"#,
    );

    let err = build_project_composition_graph(&tmp.path().join("runmat.toml"))
        .expect_err("missing dependency manifest should fail");
    assert!(err.to_string().contains("points to missing manifest"));
}

#[test]
fn composition_graph_reports_duplicate_package_names() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::create_dir_all(tmp.path().join("dep_a/src")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    fs::write(
        tmp.path().join("dep_a/src/fn_a.m"),
        "function y = fn_a(); y = 1; end",
    )
    .unwrap();
    write_manifest(
        tmp.path(),
        r#"
[package]
name = "dup_pkg"

[sources]
roots = ["src"]

[dependencies]
dep_a = { path = "dep_a" }
"#,
    );
    write_manifest(
        &tmp.path().join("dep_a"),
        r#"
[package]
name = "dup_pkg"

[sources]
roots = ["src"]
"#,
    );

    let err = build_project_composition_graph(&tmp.path().join("runmat.toml"))
        .expect_err("duplicate package names should fail");
    assert!(err.to_string().contains("duplicate package name"));
}

#[test]
fn resolve_named_entrypoint_from_discovers_and_resolves() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/app")).unwrap();
    fs::write(
        tmp.path().join("src/app/server.m"),
        "function y = main(); y = 1; end",
    )
    .unwrap();
    write_manifest(
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

    let discovered = resolve_named_entrypoint_from(&tmp.path().join("src"), "server")
        .expect("resolver should succeed")
        .expect("entrypoint should resolve");
    assert_eq!(discovered.root_package, "demo");
    assert_eq!(
        discovered.entrypoint.target,
        ResolvedEntrypointTarget::ModuleFunction
    );
    assert_eq!(
        discovered.entrypoint.source_file.canonicalize().unwrap(),
        tmp.path().join("src/app/server.m").canonicalize().unwrap()
    );
}

#[test]
fn resolve_named_entrypoint_from_reports_resolution_errors() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    write_manifest(
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

    let err = resolve_named_entrypoint_from(tmp.path(), "server")
        .expect_err("missing module file should return explicit resolve error");
    assert!(err
        .to_string()
        .contains("failed to resolve project entrypoint"));
}
