use futures::{executor::block_on, FutureExt};
use runmat_filesystem::{
    pinned::{InputManifest, PinnedContent, PinnedEntry},
    FsProvider, MemoryFsProvider, NativeFsProvider,
};
use runmat_package::*;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};

struct GitFixture {
    root: PathBuf,
    source: GitSourceId,
}
impl PackageSourceProvider for GitFixture {
    fn acquire_git<'a>(
        &'a self,
        plan: &'a GitAcquisitionPlan,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<GitPackageMount, String>> + 'a>>
    {
        async move {
            validate_git_acquisition(plan, &self.source).map_err(|e| e.to_string())?;
            Ok(GitPackageMount {
                root: self.root.clone(),
                source: self.source.clone(),
            })
        }
        .boxed_local()
    }
}

#[test]
fn captured_workspace_resolves_external_sources_without_reading_live_workspace_files() {
    let _serial = runmat_filesystem::provider_override_lock();
    block_on(async {
        let temp = tempfile::tempdir().unwrap();
        let cache_root = temp.path().join("cache/helper");
        std::fs::create_dir_all(cache_root.join("src")).unwrap();
        std::fs::write(
            cache_root.join("runmat.toml"),
            "[package]\nname='helper'\nversion='1.0.0'\n[sources]\nroots=['src']\n",
        )
        .unwrap();
        std::fs::write(
            cache_root.join("src/helper.m"),
            "function y = helper(); y = 42; end\n",
        )
        .unwrap();
        let git = GitFixture {
            root: cache_root.clone(),
            source: GitSourceId::new(
                "https://example.test/helper.git",
                "0123456789abcdef0123456789abcdef01234567".parse().unwrap(),
                ".".parse().unwrap(),
                ContentDigest::sha256("verified fixture tree"),
            )
            .unwrap(),
        };
        let workspace = Arc::new(MemoryFsProvider::new());
        workspace.create_dir_all(Path::new("/src")).await.unwrap();
        let manifest_text = "[package]\nname='application'\nversion='1.0.0'\n[sources]\nroots=['src']\n[dependencies]\nhelper={git='https://example.test/helper.git',branch='main',version='1.0.0'}\n";
        let source_text = "answer = helper();\n";
        let mut entries = Vec::new();
        for (path, text) in [
            ("/runmat.toml", manifest_text),
            ("/src/main.m", source_text),
        ] {
            let version_id = workspace
                .publish_output(Path::new(path), text.as_bytes(), None)
                .await
                .unwrap();
            entries.push(PinnedEntry {
                path: path.into(),
                content: PinnedContent::File {
                    version_id,
                    size: text.len() as u64,
                },
            });
        }
        // A later file exists in the workspace but is outside the capture.
        workspace
            .write(
                Path::new("/src/late.m"),
                b"error('must not enter the catalog')",
            )
            .await
            .unwrap();
        let options = ProjectResolveOptions {
            target: "x86_64-unknown-linux-gnu".into(),
            default_server_origin: "https://api.runmat.com".into(),
            default_registry_index: "https://api.runmat.com".into(),
            groups: BTreeSet::from([DependencyGroup::Runtime]),
            root_features: BTreeSet::new(),
            host_capabilities: BTreeSet::new(),
            source_intent: SourceAcquisitionIntent::Execute,
            source_policy: SourceAcquisitionPolicy::default(),
        };
        let result = resolve_captured_project_async(
            Path::new("/runmat.toml"),
            InputManifest {
                snapshot_id: "capture".into(),
                entries,
            },
            options,
            &git,
            workspace.clone(),
            Arc::new(NativeFsProvider),
        )
        .await
        .unwrap();
        assert_eq!(result.project.frozen.graph.packages.len(), 2);
        assert!(!result
            .project
            .frozen
            .access_paths
            .values()
            .any(|p| p.ends_with("late.m")));
        assert_eq!(
            result
                .filesystem
                .read(&cache_root.join("src/helper.m"))
                .await
                .unwrap(),
            b"function y = helper(); y = 42; end\n"
        );
        assert!(result
            .filesystem
            .write(&cache_root.join("src/helper.m"), b"change")
            .await
            .is_err());
        assert!(
            workspace.metadata(Path::new("/runmat.lock")).await.is_err(),
            "resolution must not overwrite captured lock state"
        );
        FrozenProjectHandoff::new(result.project.frozen)
            .validate()
            .unwrap();
    });
}
