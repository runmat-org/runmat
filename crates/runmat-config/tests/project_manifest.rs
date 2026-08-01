use runmat_config::project::{
    build_project_source_index, discover_project_manifest_from, load_project_manifest,
    parse_project_manifest_json, parse_project_manifest_toml, resolve_named_entrypoint_from,
    resolve_project_entrypoint, resolve_project_source_input_from, ProjectEntrypointResolveError,
    ProjectManifestLoadError, ProjectSourceIndexError, ResolveProjectSourceInputError,
    ResolvedEntrypointTarget, PROJECT_MANIFEST_FILENAME,
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

[entrypoints.main]
path = "src/main"
"#,
    );

    let loaded = load_project_manifest(&manifest_path).expect("manifest should validate");
    assert_eq!(loaded.package.name, "demo");
    assert_eq!(loaded.sources.roots.len(), 1);
    assert_eq!(loaded.entrypoints.len(), 1);
}

#[test]
fn parses_manifest_with_runtime_section() {
    let parsed = parse_project_manifest_toml(
        r#"
[package]
name = "demo"
version = "0.1.0"

[sources]
roots = ["src"]

[runtime]
verbose = true
"#,
    )
    .expect("manifest with runtime section should parse");

    assert_eq!(parsed.package.name, "demo");
    assert_eq!(parsed.sources.roots, vec![std::path::PathBuf::from("src")]);
}

#[test]
fn parses_manifest_with_desktop_section() {
    let parsed = parse_project_manifest_toml(
        r#"
[package]
name = "demo"
version = "0.1.0"

[sources]
roots = ["src"]

[desktop]
artifact_root = ".artifacts"
notebook_run_mode = "stop_on_error"
"#,
    )
    .expect("manifest with desktop section should parse");

    assert_eq!(parsed.package.name, "demo");
    assert_eq!(parsed.sources.roots, vec![std::path::PathBuf::from("src")]);
}

#[test]
fn parses_manifest_with_test_section() {
    let parsed = parse_project_manifest_toml(
        r#"
[package]
name = "demo"
version = "0.1.0"

[sources]
roots = ["src"]

[test]
roots = ["tests", "integration"]
jobs = 4
isolation = "process"
"#,
    )
    .expect("manifest with test section should parse");

    assert_eq!(parsed.package.name, "demo");
    assert_eq!(parsed.sources.roots, vec![std::path::PathBuf::from("src")]);
}

#[test]
fn parses_json_manifest_with_runtime_test_and_desktop_sections() {
    let parsed = runmat_config::project::parse_project_manifest_json(
        r#"{
            "package": { "name": "demo", "version": "0.1.0" },
            "sources": { "roots": ["src"] },
            "runtime": { "verbose": true },
            "test": {
                "roots": ["tests"],
                "reports": { "junit": "artifacts/junit.xml" }
            },
            "desktop": { "artifact_root": ".artifacts" }
        }"#,
    )
    .expect("JSON manifest with product-owned sections should parse");

    assert_eq!(parsed.package.name, "demo");
    assert_eq!(parsed.sources.roots, vec![std::path::PathBuf::from("src")]);
}

#[test]
fn project_manifest_round_trips_through_canonical_toml_and_json_shapes() {
    let manifest = parse_project_manifest_toml(
        r#"
[package]
name = "demo"
version = "0.1.0"

[sources]
roots = ["src"]

[dependencies]
helper = { path = "deps/helper", version = "0.2.0" }

[entrypoints.main]
path = "src/main"

[entrypoints.serve]
module = "app.server"
function = "start"

[runtime]
verbose = true

[test]
roots = ["tests"]

[desktop]
artifact_root = ".artifacts"
"#,
    )
    .expect("mixed manifest parses");

    let toml = toml::to_string(&manifest).expect("serialize canonical TOML");
    let from_toml = parse_project_manifest_toml(&toml).expect("reparse canonical TOML");
    assert_eq!(from_toml, manifest);
    assert!(toml.contains("[entrypoints.main]"));
    assert!(toml.contains("[entrypoints.serve]"));

    let json = serde_json::to_string(&manifest).expect("serialize canonical JSON");
    let from_json = parse_project_manifest_json(&json).expect("reparse canonical JSON");
    assert_eq!(from_json, manifest);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap()["entrypoints"]["main"]["path"],
        "src/main"
    );
}

#[test]
fn validation_rejects_unsatisfied_runmat_version() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    let manifest_path = write_manifest(
        tmp.path(),
        r#"
[package]
name = "demo"
runmat-version = ">=999.0.0"

[sources]
roots = ["src"]
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("version gate should fail");
    let ProjectManifestLoadError::Validation { source, .. } = err else {
        panic!("expected validation error");
    };
    assert!(source
        .messages
        .iter()
        .any(|msg| msg.contains("[package].runmat-version")));
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
    assert!(err
        .messages
        .iter()
        .any(|msg| msg == "[package].name is required and must be non-empty"));
    assert!(err
        .messages
        .iter()
        .any(|msg| msg == "[sources].roots is required and must be non-empty"));
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

[entrypoints.main]
path = "src/main"
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("missing source root should fail");
    let ProjectManifestLoadError::Validation { source, .. } = err else {
        panic!("expected validation error");
    };
    assert!(source
        .messages
        .iter()
        .any(|msg| msg.contains("source root `src`")));
}

#[test]
fn validation_rejects_multiple_dependency_locators() {
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

[entrypoints.main]
path = "src/main"
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("multiple locators should fail");
    let ProjectManifestLoadError::Validation { source, .. } = err else {
        panic!("expected validation error");
    };
    assert!(source.messages.iter().any(|message| {
        message.contains("dependency `dep_a`")
            && message.contains("must select exactly one locator")
    }));
}

#[test]
fn parsing_rejects_unknown_dependency_fields() {
    let manifest = r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[dependencies]
    dep_a = { path = "dep_a", unsupported = true }
"#;
    let err = parse_project_manifest_toml(manifest).expect_err("unknown fields should fail");
    assert!(err.to_string().contains("unknown field"));
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

[entrypoints.main]
path = "src/main"

[entrypoints.main]
module = "app.server"
function = "run"
"#,
    );
    let err = load_project_manifest(&manifest_path).expect_err("duplicate entrypoint should fail");
    assert!(matches!(err, ProjectManifestLoadError::ParseToml { .. }));
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

[entrypoints.server]
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
fn project_discovery_skips_runtime_only_configuration() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    fs::create_dir_all(root.join("nested")).unwrap();
    write_manifest(
        root,
        r#"
[runtime]
verbose = false

[desktop]
theme = "dark"
"#,
    );

    assert_eq!(
        discover_project_manifest_from(&root.join("nested/script.m")),
        None,
        "a runtime/desktop configuration must not opt the folder into project semantics"
    );
}

#[test]
fn project_discovery_keeps_malformed_project_documents_visible() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();
    let manifest_path = write_manifest(
        root,
        r#"
[package]
name = "demo"
"#,
    );

    assert_eq!(
        discover_project_manifest_from(&root.join("script.m")),
        Some(manifest_path.clone())
    );
    assert!(
        load_project_manifest(&manifest_path).is_err(),
        "declaring a project section must retain strict project validation"
    );
}

#[test]
fn resolve_project_source_input_from_infers_m_extension() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();

    let resolved = resolve_project_source_input_from(tmp.path(), std::path::Path::new("src/main"))
        .expect("resolve source input");
    assert_eq!(resolved, std::path::PathBuf::from("src/main.m"));
}

#[test]
fn resolve_project_source_input_from_resolves_named_entrypoint() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    fs::write(
        tmp.path().join(PROJECT_MANIFEST_FILENAME),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.main]
path = "src/main"
"#,
    )
    .unwrap();

    let resolved = resolve_project_source_input_from(tmp.path(), std::path::Path::new("main"))
        .expect("resolve source input");
    assert_eq!(
        resolved.canonicalize().unwrap(),
        tmp.path().join("src/main.m").canonicalize().unwrap()
    );
}

#[test]
fn root_entrypoint_resolution_does_not_traverse_dependencies() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::create_dir_all(tmp.path().join("deps/unavailable")).unwrap();
    fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
    fs::write(
        tmp.path().join(PROJECT_MANIFEST_FILENAME),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[dependencies]
unavailable = { path = "deps/unavailable" }

[entrypoints.main]
path = "src/main"
"#,
    )
    .unwrap();

    let resolved = resolve_named_entrypoint_from(tmp.path(), "main")
        .expect("entrypoint syntax is root-local")
        .expect("entrypoint exists");
    assert_eq!(
        resolved.entrypoint.source_file,
        tmp.path().join("src/main.m")
    );
}

#[test]
fn resolve_project_source_input_from_returns_plain_candidate_when_name_is_not_entrypoint() {
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

    let resolved = resolve_project_source_input_from(tmp.path(), std::path::Path::new("missing"))
        .expect("non-entrypoint names should pass through unchanged");
    assert_eq!(resolved, std::path::PathBuf::from("missing"));
}

#[test]
fn resolve_project_source_input_from_reports_named_entrypoint_resolution_errors() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();
    fs::write(
        tmp.path().join(PROJECT_MANIFEST_FILENAME),
        r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.server]
module = "app.server"
function = "main"
"#,
    )
    .unwrap();

    let err = resolve_project_source_input_from(tmp.path(), std::path::Path::new("server"))
        .expect_err("invalid named entrypoint target should return resolution error");

    match err {
        ResolveProjectSourceInputError::EntrypointResolve { entrypoint, .. } => {
            assert_eq!(entrypoint, "server");
        }
    }
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
        tmp.path().join("src/+pkg/@Point/Point.m"),
        "classdef Point; end",
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
    assert!(qualified.contains("pkg.Point.Point"));
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
    let constructor = index
        .files
        .iter()
        .find(|file| file.relative_path == std::path::Path::new("+pkg/@Point/Point.m"))
        .expect("class constructor should be indexed");
    assert_eq!(constructor.qualified_name, "pkg.Point.Point");
    assert_eq!(
        constructor.class_qualified_name.as_deref(),
        Some("pkg.Point")
    );
    assert_eq!(
        constructor.class_definition_qualified_name(),
        Some("pkg.Point")
    );
}

#[test]
fn source_index_distinguishes_root_class_constructor_from_member_identity() {
    let tmp = TempDir::new().unwrap();
    fs::create_dir_all(tmp.path().join("src/@Report")).unwrap();
    fs::write(
        tmp.path().join("src/@Report/Report.m"),
        "classdef Report; end",
    )
    .unwrap();
    fs::write(
        tmp.path().join("src/@Report/title.m"),
        "function out=title(); out=1; end",
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

    let constructor = index
        .files
        .iter()
        .find(|file| file.relative_path == std::path::Path::new("@Report/Report.m"))
        .expect("class constructor should be indexed");
    assert_eq!(constructor.qualified_name, "Report.Report");
    assert_eq!(constructor.class_qualified_name.as_deref(), Some("Report"));
    assert_eq!(
        constructor.class_definition_qualified_name(),
        Some("Report")
    );

    let member = index
        .files
        .iter()
        .find(|file| file.relative_path == std::path::Path::new("@Report/title.m"))
        .expect("class member should be indexed");
    assert_eq!(member.qualified_name, "Report.title");
    assert_eq!(member.class_qualified_name.as_deref(), Some("Report"));
    assert_eq!(member.function_qualified_name(), Some("Report.title"));
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
    let ProjectSourceIndexError::InvalidSourceRoot { root } = err else {
        panic!("expected invalid source root error");
    };
    assert_eq!(root, std::path::PathBuf::from("src"));
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

[entrypoints.main]
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

[entrypoints.server]
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

[entrypoints.server]
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

[entrypoints.point-move]
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

[entrypoints.server]
module = "app.server"
function = "main"
"#,
    );
    let manifest = load_project_manifest(&manifest_path).expect("manifest should validate");
    fs::remove_dir_all(tmp.path().join("src")).unwrap();
    let err = resolve_project_entrypoint(tmp.path(), &manifest, "server")
        .expect_err("missing source root should bubble source index error");
    let ProjectEntrypointResolveError::SourceIndex { source, .. } = err else {
        panic!("expected source index resolution error");
    };
    assert!(matches!(
        source,
        ProjectSourceIndexError::InvalidSourceRoot { .. }
    ));
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

[entrypoints.server]
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

[entrypoints.server]
module = "app.server"
function = "main"
"#,
    );

    let err = resolve_named_entrypoint_from(tmp.path(), "server")
        .expect_err("missing module file should return explicit resolve error");
    let runmat_config::project::DiscoverProjectEntrypointError::Resolve {
        entrypoint, source, ..
    } = err
    else {
        panic!("expected entrypoint resolve error");
    };
    assert_eq!(entrypoint, "server");
    assert!(matches!(
        *source,
        ProjectEntrypointResolveError::MissingModuleTarget { .. }
    ));
}
