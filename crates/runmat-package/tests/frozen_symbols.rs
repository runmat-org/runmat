use futures::executor::block_on;
use runmat_package::{
    discover_source_symbols_from_source_name, discover_source_symbols_from_source_name_async,
};
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn manifest(name: &str, dependencies: &str) -> String {
    format!(
        r#"
[package]
name = "{name}"
version = "1.0.0"

[sources]
roots = ["src"]

{dependencies}
"#
    )
}

#[test]
fn direct_dependencies_preserve_graph_and_source_identity() {
    let temp = TempDir::new().unwrap();
    write(
        &temp.path().join("runmat.toml"),
        &manifest(
            "application",
            r#"[dependencies]
tools = { path = "deps/tools", version = "1.0.0" }"#,
        ),
    );
    write(
        &temp.path().join("src/main.m"),
        "result = tools.helper();\n",
    );
    write(
        &temp.path().join("deps/tools/runmat.toml"),
        &manifest("tools-package", ""),
    );
    write(
        &temp.path().join("deps/tools/src/helper.m"),
        "function y = helper(); y = 42; end\n",
    );

    let discovered = discover_source_symbols_from_source_name("src/main.m", temp.path())
        .unwrap()
        .unwrap();
    assert!(discovered.graph_digest.is_some());
    assert!(discovered.source_revision.is_some());
    assert!(discovered.symbols.contains("tools.helper"));
    assert!(discovered.symbols.contains("helper"));
    let definition = discovered
        .definitions
        .iter()
        .find(|definition| definition.name == "helper")
        .unwrap();
    assert_eq!(definition.package_name, "tools-package");
    assert!(definition.package_instance.is_some());
    let source_id = definition.source_id.as_ref().unwrap();
    assert_eq!(
        definition.package_instance.as_ref(),
        Some(&source_id.package_instance)
    );
    assert_eq!(source_id.relative_path.as_str(), "src/helper.m");

    let asynchronous = block_on(discover_source_symbols_from_source_name_async(
        "src/main.m",
        temp.path(),
    ))
    .unwrap()
    .unwrap();
    assert_eq!(asynchronous.graph_digest, discovered.graph_digest);
    assert_eq!(asynchronous.source_revision, discovered.source_revision);
    assert_eq!(asynchronous.symbols, discovered.symbols);
    assert_eq!(asynchronous.definitions, discovered.definitions);
}

#[test]
fn transitive_dependencies_do_not_leak_into_the_root_namespace() {
    let temp = TempDir::new().unwrap();
    write(
        &temp.path().join("runmat.toml"),
        &manifest(
            "application",
            r#"[dependencies]
middle = { path = "deps/middle", version = "1.0.0" }"#,
        ),
    );
    write(&temp.path().join("src/main.m"), "result = middle.api();\n");
    write(
        &temp.path().join("deps/middle/runmat.toml"),
        &manifest(
            "middle-package",
            r#"[dependencies]
leaf = { path = "deps/leaf", version = "1.0.0" }"#,
        ),
    );
    write(
        &temp.path().join("deps/middle/src/api.m"),
        "function y = api(); y = 1; end\n",
    );
    write(
        &temp.path().join("deps/middle/deps/leaf/runmat.toml"),
        &manifest("leaf-package", ""),
    );
    write(
        &temp.path().join("deps/middle/deps/leaf/src/internal.m"),
        "function y = internal(); y = 2; end\n",
    );

    let discovered = discover_source_symbols_from_source_name("src/main.m", temp.path())
        .unwrap()
        .unwrap();
    assert!(discovered.symbols.contains("middle.api"));
    assert!(!discovered.symbols.contains("leaf.internal"));
    assert!(!discovered.symbols.contains("internal"));
    assert!(discovered
        .definitions
        .iter()
        .all(|definition| definition.package_name != "leaf-package"));
}

#[test]
fn ambiguous_and_private_dependency_symbols_require_valid_visibility() {
    let temp = TempDir::new().unwrap();
    write(
        &temp.path().join("runmat.toml"),
        &manifest(
            "application",
            r#"[dependencies]
left = { path = "deps/left", version = "1.0.0" }
right = { path = "deps/right", version = "1.0.0" }"#,
        ),
    );
    write(&temp.path().join("src/main.m"), "x = left.shared();\n");
    for dependency in ["left", "right"] {
        write(
            &temp.path().join(format!("deps/{dependency}/runmat.toml")),
            &manifest(dependency, ""),
        );
        write(
            &temp.path().join(format!("deps/{dependency}/src/shared.m")),
            "function y = shared(); y = 1; end\n",
        );
        write(
            &temp
                .path()
                .join(format!("deps/{dependency}/src/private/secret.m")),
            "function y = secret(); y = 1; end\n",
        );
    }

    let discovered = discover_source_symbols_from_source_name("src/main.m", temp.path())
        .unwrap()
        .unwrap();
    assert!(discovered.symbols.contains("left.shared"));
    assert!(discovered.symbols.contains("right.shared"));
    assert!(!discovered.symbols.contains("shared"));
    assert!(!discovered.symbols.contains("left.secret"));
    assert!(!discovered.symbols.contains("right.secret"));
    assert!(!discovered.symbols.contains("secret"));
}
