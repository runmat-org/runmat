use futures::executor::block_on;
use runmat_config::project::{
    build_project_source_index_async, discover_project_manifest_from_async,
    load_project_manifest_async, resolve_named_entrypoint_from_async,
    resolve_project_source_input_from_async, ResolvedEntrypointTarget,
};
use runmat_filesystem::{provider_override_lock, replace_provider, MemoryFsProvider};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn write(provider: &MemoryFsProvider, path: &str, content: &str) {
    provider
        .write_project_path(path, content.as_bytes())
        .expect("write virtual project file");
}

#[test]
fn async_project_pipeline_uses_the_installed_virtual_filesystem() {
    let _provider_lock = provider_override_lock();
    let provider = MemoryFsProvider::with_current_dir("/workspace");
    write(
        &provider,
        "/workspace/runmat.toml",
        r#"
[package]
name = "app"

[sources]
roots = ["src"]

[dependencies]
helper = { path = "deps/helper" }

[entrypoints.main]
path = "src/main"

[runtime]
verbose = true

[test]
roots = ["tests"]

[desktop]
artifact_root = ".artifacts"
"#,
    );
    write(
        &provider,
        "/workspace/src/main.m",
        "value = helper.add(1, 2);",
    );
    write(
        &provider,
        "/workspace/deps/helper/runmat.toml",
        r#"
[package]
name = "helper"

[sources]
roots = ["src"]
"#,
    );
    write(
        &provider,
        "/workspace/deps/helper/src/+helper/add.m",
        "function out = add(a, b)\nout = a + b;\nend",
    );
    let _provider = replace_provider(Arc::new(provider));

    block_on(async {
        let source = Path::new("/workspace/src/main.m");
        let manifest_path = discover_project_manifest_from_async(source)
            .await
            .expect("discover virtual manifest");
        assert_eq!(manifest_path, PathBuf::from("/workspace/runmat.toml"));

        let manifest = load_project_manifest_async(&manifest_path)
            .await
            .expect("load and validate through virtual filesystem");
        let index = build_project_source_index_async(Path::new("/workspace"), &manifest)
            .await
            .expect("index virtual source roots");
        assert_eq!(index.files.len(), 1);
        assert_eq!(index.files[0].qualified_name, "main");

        let entrypoint = resolve_named_entrypoint_from_async(source, "main")
            .await
            .expect("resolve virtual entrypoint")
            .expect("entrypoint exists");
        assert_eq!(entrypoint.entrypoint.target, ResolvedEntrypointTarget::Path);
        assert_eq!(
            entrypoint.entrypoint.source_file,
            PathBuf::from("/workspace/src/main.m")
        );

        let source_input =
            resolve_project_source_input_from_async(Path::new("/workspace"), Path::new("main"))
                .await
                .expect("resolve virtual source input");
        assert_eq!(source_input, PathBuf::from("/workspace/src/main.m"));
    });
}

#[test]
fn async_manifest_validation_does_not_consult_the_native_filesystem() {
    let _provider_lock = provider_override_lock();
    let provider = MemoryFsProvider::with_current_dir("/browser");
    write(
        &provider,
        "/browser/runmat.json",
        r#"{
            "package": { "name": "browser-app" },
            "sources": { "roots": ["virtual-src"] },
            "entrypoints": {
                "main": { "path": "virtual-src/main" }
            },
            "test": { "roots": ["tests"] }
        }"#,
    );
    write(&provider, "/browser/virtual-src/main.m", "answer = 42;");
    let _provider = replace_provider(Arc::new(provider));

    block_on(async {
        let manifest = load_project_manifest_async(Path::new("/browser/runmat.json"))
            .await
            .expect("virtual-only paths must validate");
        assert_eq!(manifest.package.name, "browser-app");
    });
}
