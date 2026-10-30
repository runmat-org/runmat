//! Named project selection, shared by hosts without interpreting manifest targets.
use std::path::Path;

use runmat_config::project::{
    parse_project_manifest_json, parse_project_manifest_toml, resolve_project_entrypoint_async,
    ProjectManifest, ResolvedProjectEntrypoint,
};
use runmat_package::FrozenProjectHandoff;

/// A manifest whose canonical identity is checked against its frozen graph.
/// Hosts must retain their filesystem provider while resolving and consuming it.
pub struct CapturedProjectEntrypoints {
    pub handoff: FrozenProjectHandoff,
    pub manifest: ProjectManifest,
    pub project_document: runmat_config::runtime::HostConfigDocument,
}

impl CapturedProjectEntrypoints {
    pub async fn capture(start: &Path) -> Result<Self, String> {
        let project = runmat_package::discover_frozen_project_from_async(start, Default::default())
            .await
            .map_err(|error| error.to_string())?
            .ok_or("named entrypoints require a RunMat project")?;
        let handoff = FrozenProjectHandoff::new(project);
        handoff.validate().map_err(|error| error.to_string())?;
        let source = runmat_filesystem::read_to_string_async(&handoff.project.manifest_path)
            .await
            .map_err(|error| error.to_string())?;
        let format =
            runmat_config::document::RunmatConfigFormat::from_path(&handoff.project.manifest_path)
                .map_err(|error| error.to_string())?;
        let manifest = match format {
            runmat_config::document::RunmatConfigFormat::Toml => {
                parse_project_manifest_toml(&source).map_err(|error| error.to_string())?
            }
            runmat_config::document::RunmatConfigFormat::Json => {
                parse_project_manifest_json(&source).map_err(|error| error.to_string())?
            }
        };
        manifest
            .validate_async(
                handoff
                    .project
                    .manifest_path
                    .parent()
                    .ok_or("project manifest has no parent directory")?,
            )
            .await
            .map_err(|error| error.to_string())?;
        handoff.project.validate_root_manifest(&manifest)?;
        Ok(Self {
            handoff,
            manifest,
            project_document: runmat_config::runtime::HostConfigDocument { source, format },
        })
    }

    pub async fn resolve(&self, name: &str) -> Result<ResolvedProjectEntrypoint, String> {
        self.handoff.validate().map_err(|error| error.to_string())?;
        self.handoff
            .project
            .validate_root_manifest(&self.manifest)?;
        let root = self
            .handoff
            .project
            .manifest_path
            .parent()
            .ok_or("project manifest has no parent directory")?;
        resolve_project_entrypoint_async(root, &self.manifest, name)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| format!("project has no named entrypoint `{name}`"))
    }

    /// Issues the same exact primary-source authority as ordinary compilation.
    pub async fn capture_source(
        &self,
        target: &ResolvedProjectEntrypoint,
    ) -> Result<crate::ExecutableSource, String> {
        crate::ExecutableSource::capture_project_primary(&self.handoff.project, &target.source_file)
            .await
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_filesystem::{provider_override_lock, replace_provider, MemoryFsProvider};
    use std::sync::Arc;

    #[test]
    fn named_selection_keeps_manifest_source_and_module_authority() {
        let _lock = provider_override_lock();
        let provider = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
        let manifest = "[package]\nname = 'named'\n[sources]\nroots = ['src']\n[entrypoints.script]\npath = 'src/main.m'\n[entrypoints.call]\nmodule = 'app.server'\nfunction = 'main'\n";
        for (path, content) in [
            ("/workspace/runmat.toml", manifest),
            ("/workspace/src/main.m", "answer = 42;"),
            (
                "/workspace/src/app/server.m",
                "function y = main(x)\ny = x + 1;\nend\n",
            ),
        ] {
            provider
                .write_project_path(path, content.as_bytes())
                .unwrap();
        }
        let _provider = replace_provider(provider.clone());
        futures::executor::block_on(async {
            let mut project = CapturedProjectEntrypoints::capture(Path::new("/workspace"))
                .await
                .unwrap();
            let script = project.resolve("script").await.unwrap();
            assert_eq!(
                project.capture_source(&script).await.unwrap().text,
                "answer = 42;"
            );
            let function = project.resolve("call").await.unwrap();
            assert_eq!(function.module.as_deref(), Some("app.server"));
            assert_eq!(function.function.as_deref(), Some("main"));
            assert!(project.capture_source(&function).await.is_ok());
            // The companion module also declares `main`; it cannot turn the
            // selected main.m script into that function. Both local identity
            // and portable products must select the primary compiled source.
            for (target, kind) in [
                (&script, runmat_execution::ExecutableEntrypointKind::Script),
                (
                    &function,
                    runmat_execution::ExecutableEntrypointKind::Function,
                ),
            ] {
                let mut session = crate::RunMatSession::with_options(false, false).unwrap();
                session
                    .install_project_handoff(project.handoff.clone())
                    .await
                    .unwrap();
                let source = project.capture_source(target).await.unwrap();
                let unit = session.compile_executable_unit(source, None).await.unwrap();
                assert_eq!(
                    unit.entrypoint_identity(target.function.as_deref())
                        .unwrap()
                        .0,
                    kind
                );
                assert_eq!(
                    unit.portable_envelope_for(target.function.as_deref())
                        .unwrap()
                        .manifest
                        .identity
                        .entrypoint_kind,
                    kind
                );
                assert_eq!(unit.entrypoint_identity(None).unwrap().0, kind);
                if kind == runmat_execution::ExecutableEntrypointKind::Script {
                    assert!(
                        unit.entrypoint_identity(Some("main")).is_err(),
                        "a companion function is not a primary-source entrypoint"
                    );
                }
            }
            assert!(project.resolve("absent").await.is_err());
            provider
                .write_project_path("/workspace/src/main.m", b"answer = 99;")
                .unwrap();
            assert!(project
                .capture_source(&script)
                .await
                .unwrap_err()
                .contains("frozen project"));
            project.manifest.entrypoints[0].name = "substituted".into();
            assert!(project
                .resolve("substituted")
                .await
                .unwrap_err()
                .contains("captured manifest differs"));
        });
    }
}
