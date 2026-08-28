#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use runmat_filesystem::{provider_override_lock, replace_provider, MemoryFsProvider};
use runmat_package::build_frozen_project_async;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[test]
fn async_path_project_loader_uses_the_virtual_filesystem() {
    let _provider_lock = provider_override_lock();
    let provider = MemoryFsProvider::with_current_dir("/workspace");
    for (path, contents) in [
        (
            "/workspace/runmat.toml",
            r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
helper = { path = "deps/helper", version = "1.0.0" }
"#,
        ),
        ("/workspace/src/main.m", "result = helper();\n"),
        (
            "/workspace/deps/helper/runmat.toml",
            r#"
[package]
name = "helper"
version = "1.0.0"

[sources]
roots = ["src"]
"#,
        ),
        (
            "/workspace/deps/helper/src/helper.m",
            "function y = helper(); y = 42; end\n",
        ),
    ] {
        provider
            .write_project_path(path, contents.as_bytes())
            .unwrap();
    }
    let _provider = replace_provider(Arc::new(provider));
    let frozen = block_on(build_frozen_project_async(
        Path::new("/workspace/runmat.toml"),
        BTreeSet::new(),
    ))
    .unwrap();
    assert_eq!(frozen.graph.packages.len(), 2);
    assert_eq!(frozen.all_sources().count(), 2);
    assert_eq!(frozen.workspace_root, PathBuf::from("/workspace"));
}
