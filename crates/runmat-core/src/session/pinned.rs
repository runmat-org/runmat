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

    /// Mint a scope from the canonical immutable-input/owned-output cache.
    /// The filesystem owner verifies the exact captured effective base before
    /// this session publishes the cache selection. No provider reference escapes.
    pub fn capture_input_filesystem(
        &mut self,
        original: &runmat_filesystem::CapturedReadScope,
        inputs: InputManifest,
    ) -> Result<runmat_filesystem::CapturedPinnedScope, crate::RunError> {
        let provider = self
            .pinned_provider(&inputs, original)
            .map_err(|error| input_runtime_error(error.to_string()))?;
        let scope = original
            .pinned(Arc::clone(&provider))
            .map_err(|error| input_runtime_error(error.to_string()))?;
        self.pinned_inputs = Some((inputs, provider));
        Ok(scope)
    }

    /// Invoke the same immutable executor with the input/output scope issued
    /// before named resolution and compilation, not a second live filesystem.
    pub async fn invoke_executable_with_inputs(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
        filesystem: &runmat_filesystem::CapturedPinnedScope,
    ) -> Result<runmat_runtime::execution::RootedValueSequence, crate::RunError> {
        filesystem
            .run(self.invoke_executable(unit, invocation, control))
            .await
            .map_err(|error| input_runtime_error(error.to_string()))?
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
        let (captured_filesystem, _) =
            match runmat_filesystem::CapturedReadScope::capture(&[std::path::Path::new(".")], None)
            {
                Ok(scope) => scope,
                Err(error) => return input_error(&request, error.to_string()),
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
        let provider = match self.pinned_provider(&inputs, &captured_filesystem) {
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
        let filesystem = match captured_filesystem.pinned(Arc::clone(&provider)) {
            Ok(scope) => scope,
            Err(error) => return input_error(&request, error.to_string()),
        };
        let original_execution = match captured_filesystem.execution() {
            Ok(scope) => scope,
            Err(error) => return input_error(&request, error.to_string()),
        };
        // A pre-existing source catalog was resolved against live content.
        // Discovery for this request must instead read the captured inventory.
        let previous_handoff = self.project_handoff().cloned();
        if let Err(error) = filesystem
            .run(self.clear_handoff_for_target())
            .await
            .map_err(|error| error.to_string())
            .and_then(|result| result.map_err(|error| error.to_string()))
        {
            return input_error(&request, error.to_string());
        }
        if let Some(handoff) = captured_handoff {
            if let Err(error) = filesystem
                .run(self.install_handoff_for_target(handoff))
                .await
                .map_err(|error| error.to_string())
                .and_then(|result| result.map_err(|error| error.to_string()))
            {
                // The install error is reported; a failed rollback is folded in below.
                let _ = original_execution
                    .run(self.clear_handoff_for_target())
                    .await;
                if let Some(previous) = previous_handoff {
                    // Report restoration failure as well; never execute against
                    // a partially installed dependency graph.
                    if let Err(restore) = original_execution
                        .run(self.install_handoff_for_target(previous))
                        .await
                        .map_err(|error| error.to_string())
                        .and_then(|result| result.map_err(|error| error.to_string()))
                    {
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
        // A scope failure is infrastructure, not a source-span diagnostic.
        // Retain only its display name, never copy an entire source/workspace.
        let failure_context = crate::abi::ExecutionSourceContext {
            name: match &request.source {
                crate::abi::SourceInput::Path(path) => path.clone(),
                crate::abi::SourceInput::Text { name, .. } => name.clone(),
                crate::abi::SourceInput::CapturedPath(source) => source.display_name().to_owned(),
            },
            text: None,
            identity: None,
        };
        let mut response = match filesystem.run(self.execute_request(request)).await {
            Ok(response) => response,
            Err(error) => ExecutionResponse {
                dependency_lock: None,
                source_context: failure_context,
                result: Err(input_runtime_error(error.to_string())),
            },
        };
        response.dependency_lock = dependency_lock;
        if let Err(error) = original_execution
            .run(self.clear_handoff_for_target())
            .await
            .map_err(|error| error.to_string())
            .and_then(|result| result.map_err(|error| error.to_string()))
        {
            response.result = Err(crate::RunError::Runtime(
                runmat_runtime::build_runtime_error(format!(
                    "Cannot release captured project context: {error}"
                ))
                .with_identifier("RunMat:ProjectContextRestore")
                .build(),
            ));
            return response;
        }
        if let Some(handoff) = previous_handoff {
            if let Err(error) = original_execution
                .run(self.install_handoff_for_target(handoff))
                .await
                .map_err(|error| error.to_string())
                .and_then(|result| result.map_err(|error| error.to_string()))
            {
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

    fn pinned_provider(
        &self,
        inputs: &InputManifest,
        original: &runmat_filesystem::CapturedReadScope,
    ) -> std::io::Result<Arc<PinnedFsProvider>> {
        if let Some(project) = self
            .captured_project
            .as_ref()
            .filter(|project| &project.inputs == inputs)
        {
            return Ok(Arc::clone(&project.provider));
        }
        if let Some((manifest, provider)) = &self.pinned_inputs {
            if manifest == inputs {
                return Ok(Arc::clone(provider));
            }
        }
        original.pin_inputs(inputs.clone())
    }
}

fn input_runtime_error(message: String) -> crate::RunError {
    crate::RunError::Runtime(
        runmat_runtime::build_runtime_error(message)
            .with_identifier("RunMat:InvalidRunInputs")
            .build(),
    )
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

/// Project handoff installation is asynchronous on native hosts (resolution may
/// read the filesystem) and synchronous in the browser; captured runs use both.
impl RunMatSession {
    async fn install_handoff_for_target(
        &mut self,
        handoff: runmat_package::FrozenProjectHandoff,
    ) -> Result<(), runmat_package::FrozenProjectHandoffError> {
        #[cfg(not(target_arch = "wasm32"))]
        let installed = self.install_project_handoff(handoff).await;
        #[cfg(target_arch = "wasm32")]
        let installed = self.install_project_handoff(handoff);
        installed.map(|_| ())
    }

    async fn clear_handoff_for_target(
        &mut self,
    ) -> Result<(), runmat_package::FrozenProjectHandoffError> {
        #[cfg(not(target_arch = "wasm32"))]
        let cleared = self.clear_project_handoff().await;
        #[cfg(target_arch = "wasm32")]
        let cleared = self.clear_project_handoff();
        cleared
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::abi::{HostExecutionPolicy, SourceInput};
    use runmat_filesystem::{FsProvider, MemoryFsProvider};
    use std::path::Path;

    // The provider override is process-wide. Keep its canonical lock on the
    // synchronous test stack for the complete current-thread runtime, including
    // all async provider operations and restoration during unwinding.
    fn with_serialized_provider(test: impl std::future::Future<Output = ()>) {
        let _serial = runmat_filesystem::provider_override_lock();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(test);
    }

    #[test]
    fn captured_project_executes_its_catalog_and_restores_the_live_context() {
        with_serialized_provider(captured_project_body());
    }

    async fn captured_project_body() {
        use runmat_filesystem::pinned::{PinnedContent, PinnedEntry};
        use runmat_package::*;
        use std::collections::BTreeSet;
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

    #[test]
    fn notebook_cells_share_output_ownership_only_within_the_same_capture() {
        with_serialized_provider(notebook_cells_body());
    }

    async fn notebook_cells_body() {
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
        let cwd = runmat_filesystem::current_dir().unwrap();
        let (original, _) =
            runmat_filesystem::CapturedReadScope::capture(&[Path::new(".")], None).unwrap();
        let scoped = session
            .capture_input_filesystem(&original, manifest.clone())
            .unwrap();
        scoped
            .run(async {
                runmat_filesystem::create_dir_async("/owned").await.unwrap();
                runmat_filesystem::set_current_dir_async("/owned")
                    .await
                    .unwrap();
                runmat_filesystem::write_async("proof.txt", b"owned scope")
                    .await
                    .unwrap();
            })
            .await
            .unwrap();
        assert_eq!(runmat_filesystem::current_dir().unwrap(), cwd);
        assert!(Arc::ptr_eq(
            &first,
            &session.pinned_inputs.as_ref().unwrap().1
        ));
        assert_eq!(
            first.read(Path::new("/owned/proof.txt")).await.unwrap(),
            b"owned scope"
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
