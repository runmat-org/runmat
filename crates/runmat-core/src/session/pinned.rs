use super::RunMatSession;
use crate::abi::{ExecutionRequest, ExecutionResponse};
use runmat_filesystem::pinned::{InputManifest, PinnedFsProvider};
use std::sync::Arc;

pub(super) struct CapturedProject {
    inputs: InputManifest,
    lock: runmat_package::PackageLock,
    provider: Arc<PinnedFsProvider>,
    handoff: runmat_package::FrozenProjectHandoff,
}

impl RunMatSession {
    /// Install a graph resolved from captured inputs through the host's existing
    /// authenticated acquisition ports. The host retains its cache leases.
    pub fn install_captured_project(
        &mut self,
        project: runmat_package::CapturedResolvedProject,
    ) -> Result<(), runmat_package::FrozenProjectHandoffError> {
        let inputs = project.inputs().clone();
        let handoff = runmat_package::FrozenProjectHandoff::new(project.project.frozen);
        handoff.validate()?;
        self.captured_project = Some(CapturedProject {
            inputs,
            lock: project.project.lock,
            provider: project.filesystem,
            handoff,
        });
        Ok(())
    }

    pub fn clear_captured_project(&mut self) {
        self.captured_project = None;
        self.pinned_inputs = None;
    }

    pub fn has_captured_project(&self, inputs: &InputManifest) -> bool {
        self.captured_project
            .as_ref()
            .is_some_and(|project| &project.inputs == inputs)
    }

    /// Execute against the caller's captured project inventory. Both hosts use
    /// this scope; filesystem reads may never consult newer live input versions.
    pub async fn execute_request_with_inputs(
        &mut self,
        request: ExecutionRequest,
        inputs: Option<InputManifest>,
    ) -> ExecutionResponse {
        let Some(inputs) = inputs else {
            self.pinned_inputs = None;
            self.captured_project = None;
            return self.execute_request(request).await;
        };
        // Notebook cells execute captured source text while their file reads and
        // companion discovery share the same immutable project inventory.
        // Cells in one notebook run share output ownership as well as inputs.
        // Reusing only a snapshot id is insufficient: compare the entire typed
        // manifest so an altered inventory cannot inherit prior write authority.
        let prepared = self
            .captured_project
            .as_ref()
            .filter(|project| project.inputs == inputs);
        let dependency_lock = match prepared
            .map(|project| runmat_package::encode_lock(&project.lock))
            .transpose()
        {
            Ok(lock) => lock,
            Err(error) => return input_error(&request, error.to_string()),
        };
        let captured_handoff = prepared.map(|project| project.handoff.clone());
        let provider = if let Some(project) = prepared {
            Some(Arc::clone(&project.provider))
        } else if let Some((manifest, provider)) = &self.pinned_inputs {
            (manifest == &inputs).then(|| Arc::clone(provider))
        } else {
            None
        };
        let provider = match provider.map(Ok).unwrap_or_else(|| {
            PinnedFsProvider::new(runmat_filesystem::current_provider(), inputs.clone())
                .map(Arc::new)
        }) {
            Ok(provider) => provider,
            Err(error) => {
                return ExecutionResponse {
                    dependency_lock: None,
                    source_context: super::run::unresolved_source_context(&request.source),
                    result: Err(crate::RunError::Runtime(
                        runmat_runtime::build_runtime_error(format!("Invalid run inputs: {error}"))
                            .with_identifier("RunMat:InvalidRunInputs")
                            .build(),
                    )),
                };
            }
        };
        // A pre-existing source catalog was resolved against live content.
        // Discovery for this request must instead read the captured inventory.
        let previous_handoff = self.project_handoff().cloned();
        self.clear_project_handoff();
        if let Some(handoff) = captured_handoff {
            if let Err(error) = self.install_project_handoff(handoff) {
                self.clear_project_handoff();
                if let Some(previous) = previous_handoff {
                    // Report restoration failure as well; never execute against
                    // a partially installed dependency graph.
                    if let Err(restore) = self.install_project_handoff(previous) {
                        return input_error(
                            &request,
                            format!("{error}; restore failed: {restore}"),
                        );
                    }
                }
                return input_error(&request, error.to_string());
            }
        }
        self.pinned_inputs = Some((inputs, Arc::clone(&provider)));
        let filesystem = runmat_filesystem::replace_provider(provider);
        let mut response = self.execute_request(request).await;
        response.dependency_lock = dependency_lock;
        drop(filesystem);
        self.clear_project_handoff();
        if let Some(handoff) = previous_handoff {
            if let Err(error) = self.install_project_handoff(handoff) {
                response.result = Err(crate::RunError::Runtime(
                    runmat_runtime::build_runtime_error(format!(
                        "Cannot restore project context: {error}"
                    ))
                    .with_identifier("RunMat:ProjectContextRestore")
                    .build(),
                ));
            }
        }
        response
    }
}

fn input_error(request: &ExecutionRequest, message: String) -> ExecutionResponse {
    ExecutionResponse {
        dependency_lock: None,
        source_context: super::run::unresolved_source_context(&request.source),
        result: Err(crate::RunError::Runtime(
            runmat_runtime::build_runtime_error(message)
                .with_identifier("RunMat:InvalidRunInputs")
                .build(),
        )),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::abi::{HostExecutionPolicy, SourceInput};
    use runmat_filesystem::{FsProvider, MemoryFsProvider};
    use std::path::Path;

    #[tokio::test]
    async fn captured_project_executes_its_catalog_and_restores_the_live_context() {
        use runmat_filesystem::pinned::{PinnedContent, PinnedEntry};
        use runmat_package::*;
        use std::collections::BTreeSet;
        let _serial = runmat_filesystem::provider_override_lock();
        let base = Arc::new(MemoryFsProvider::new());
        let _restore = runmat_filesystem::replace_provider(base.clone());
        base.create_dir_all(Path::new("/src")).await.unwrap();
        let mut entries = vec![];
        for (path, text) in [
            (
                "/runmat.toml",
                "[package]\nname='captured'\nversion='1.0.0'\n[sources]\nroots=['src']\n",
            ),
            ("/src/helper.m", "function y = helper(); y = 42; end\n"),
            ("/src/main.m", "answer = helper();\n"),
        ] {
            let version_id = base
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
        let inputs = InputManifest {
            snapshot_id: "package-capture".into(),
            entries,
        };
        struct Sources;
        impl PackageSourceProvider for Sources {
            fn acquire_git<'a>(
                &'a self,
                _: &'a GitAcquisitionPlan,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<GitPackageMount, String>> + 'a>,
            > {
                Box::pin(async { Err("fixture must not acquire an external source".into()) })
            }
        }
        let project = resolve_captured_project_async(
            Path::new("/runmat.toml"),
            inputs.clone(),
            ProjectResolveOptions {
                target: "x86_64-unknown-linux-gnu".into(),
                default_server_origin: "https://api.runmat.com".into(),
                default_registry_index: "https://api.runmat.com".into(),
                groups: BTreeSet::from([DependencyGroup::Runtime]),
                root_features: BTreeSet::new(),
                host_capabilities: BTreeSet::new(),
                source_intent: SourceAcquisitionIntent::Execute,
                source_policy: SourceAcquisitionPolicy::default(),
            },
            &Sources,
            base.clone(),
            base.clone(),
        )
        .await
        .unwrap();
        let mut session = RunMatSession::with_options(false, false).unwrap();
        session.install_captured_project(project).unwrap();
        let request = ExecutionRequest::for_source(
            SourceInput::Path("/src/main.m".into()),
            runmat_parser::CompatMode::default(),
            HostExecutionPolicy::default(),
            session.abi_workspace_handle,
        );
        let response = session
            .execute_request_with_inputs(request, Some(inputs.clone()))
            .await;
        let lock = runmat_package::decode_lock(
            response
                .dependency_lock
                .as_deref()
                .expect("captured dependency provenance"),
        )
        .unwrap();
        assert_eq!(
            runmat_package::encode_lock(&lock).unwrap(),
            response.dependency_lock.clone().unwrap()
        );
        assert!(response.result.is_ok(), "{:?}", response.result.err());
        assert!(
            session.project_handoff().is_none(),
            "captured catalog must not leak into ordinary execution"
        );
        assert!(session.has_captured_project(&inputs));
        session.clear_captured_project();
        assert!(!session.has_captured_project(&inputs));
    }

    #[tokio::test]
    async fn notebook_cells_share_output_ownership_only_within_the_same_capture() {
        let _serial = runmat_filesystem::provider_override_lock();
        let base = Arc::new(MemoryFsProvider::new());
        let _restore = runmat_filesystem::replace_provider(base.clone());
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let manifest = InputManifest {
            snapshot_id: "first".into(),
            entries: vec![],
        };
        let request = |session: &RunMatSession| {
            ExecutionRequest::for_source(
                SourceInput::Text {
                    name: "book.md#cell_1".into(),
                    text: "a = 1;".into(),
                },
                runmat_parser::CompatMode::default(),
                HostExecutionPolicy::default(),
                session.abi_workspace_handle,
            )
        };
        assert!(session
            .execute_request_with_inputs(request(&session), Some(manifest.clone()))
            .await
            .result
            .is_ok());
        let first = Arc::clone(&session.pinned_inputs.as_ref().unwrap().1);
        first
            .write(Path::new("/result.txt"), b"cell one")
            .await
            .unwrap();
        assert!(session
            .execute_request_with_inputs(request(&session), Some(manifest.clone()))
            .await
            .result
            .is_ok());
        let second = &session.pinned_inputs.as_ref().unwrap().1;
        assert!(Arc::ptr_eq(&first, second));
        assert_eq!(
            second.read(Path::new("/result.txt")).await.unwrap(),
            b"cell one"
        );
        let next = InputManifest {
            snapshot_id: "second".into(),
            ..manifest
        };
        assert!(session
            .execute_request_with_inputs(request(&session), Some(next))
            .await
            .result
            .is_ok());
        let next = &session.pinned_inputs.as_ref().unwrap().1;
        assert!(!Arc::ptr_eq(&first, next));
        assert!(next.read(Path::new("/result.txt")).await.is_err());
        assert!(next
            .write(Path::new("/result.txt"), b"overwrite")
            .await
            .is_err());
        assert_eq!(
            base.read(Path::new("/result.txt")).await.unwrap(),
            b"cell one"
        );
    }
}
