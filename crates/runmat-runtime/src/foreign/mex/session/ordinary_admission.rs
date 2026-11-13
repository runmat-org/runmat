//! Invocation-local admission for ordinary session source. A selected unit
//! receives this route only from Core's trusted interactive/script entrypoint;
//! sealed artifact workers keep their installed-only or closed route.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{Read as _, Write as _};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use runmat_execution::Digest;
use runmat_process_host::private_bytes::PrivateByteLease;

use super::{artifact_catalog, runtime_error, MexNameSelection, MexRuntimeSession};
use crate::builtins::common::path_search::{FileCandidate, FileCandidateOrigin};
use crate::context::{RuntimeContext, RuntimeExecutionStack};
use crate::foreign::{admit_interop_manifest, ForeignAdapter, ForeignPlatform};
use crate::source_context::SelectedProgramSourceOwner;
use crate::user_functions::{DynamicFunctionClearRequest, NativeNameFact, NativeNameObserver};
use crate::RuntimeError;

#[path = "ordinary_admission/captured.rs"]
mod captured;
use captured::{CapturedOrdinaryMex, OrdinaryMexEvidence};

pub(super) struct AdmittedOrdinaryMexArtifact {
    source: PathBuf,
    declared_name: String,
    callable_names: BTreeSet<String>,
    module_digest: Digest,
    evidence: OrdinaryMexEvidence,
    identity: runmat_types::ForeignArtifactIdentity,
    private_module: PathBuf,
    lease: PrivateByteLease,
}

/// One call's source-bound artifact fact. The cache owns private bytes, while
/// this receipt pairs those bytes with the exact invoking source and current
/// path selection. It is never serialized into a selected executable.
struct OrdinaryMexInvocationReceipt {
    source_owner: SelectedProgramSourceOwner,
    source: PathBuf,
    origin: FileCandidateOrigin,
    path_revision: Option<u64>,
    module_digest: Digest,
    evidence: OrdinaryMexEvidence,
    identity: runmat_types::ForeignArtifactIdentity,
}

impl AdmittedOrdinaryMexArtifact {
    pub(super) fn persist_unverified(&self) -> Result<(), String> {
        self.lease
            .persist_unverified(None, "ordinary MEX host retirement unverified")
    }

    fn selected_by(&self, request: &DynamicFunctionClearRequest) -> bool {
        match request {
            DynamicFunctionClearRequest::All | DynamicFunctionClearRequest::NativeExtensions => {
                true
            }
            DynamicFunctionClearRequest::Named(name) => {
                self.callable_names
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(name))
                    || self
                        .source
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .is_some_and(|stem| stem.eq_ignore_ascii_case(name))
            }
        }
    }
}

impl OrdinaryMexInvocationReceipt {
    fn matches_artifact(&self, artifact: &AdmittedOrdinaryMexArtifact) -> bool {
        self.source == artifact.source
            && self.module_digest == artifact.module_digest
            && self.evidence == artifact.evidence
            && self.identity == artifact.identity
    }
}

impl MexRuntimeSession {
    pub(super) fn ordinary_selected_private_paths(
        &self,
        request: &DynamicFunctionClearRequest,
    ) -> HashSet<PathBuf> {
        self.ordinary_modules
            .borrow()
            .iter()
            .filter(|artifact| artifact.selected_by(request))
            .map(|artifact| artifact.private_module.clone())
            .collect()
    }

    /// Release private bytes only after both native and isolated owners have
    /// acknowledged retirement. mexLock and failed/unverified clear retain the
    /// exact module, including its persistent/static state.
    pub(super) fn prune_retired_ordinary(&self, request: &DynamicFunctionClearRequest) {
        let live_native = self.native_modules.borrow();
        let live_isolated = self.isolated.borrow();
        let active_isolated = self.active_isolated.borrow();
        let pending = self
            .inflight_operations
            .borrow()
            .iter()
            .any(|lease| lease.outcome().is_none());
        self.ordinary_modules.borrow_mut().retain(|artifact| {
            !artifact.selected_by(request)
                || pending
                || live_native.contains(&artifact.private_module)
                || live_isolated.contains_key(&artifact.private_module)
                || active_isolated.contains(&artifact.private_module)
        });
    }

    pub fn ordinary_name_observer(self: &Rc<Self>) -> Rc<NativeNameObserver> {
        let owner = Rc::clone(self);
        Rc::new(move |runtime, name| {
            let owner = Rc::clone(&owner);
            crate::user_functions::admit_native_name_observation_with(
                &runtime,
                move || async move {
                    owner
                        .observe_search_path_name(&name)
                        .await
                        .map_err(|error| {
                            error.into_identified_runtime_error(
                                "MEX function resolution",
                                "RunMat:FunctionResolution",
                            )
                        })
                        .map(|selection| {
                            selection.and_then(|selection| match selection {
                                MexNameSelection::SearchPath(candidate) => {
                                    let (path, origin) = candidate.into_parts();
                                    Some(NativeNameFact {
                                        path,
                                        origin: Some(origin),
                                        installed: false,
                                        installed_artifact: None,
                                    })
                                }
                                MexNameSelection::Installed { .. } => None,
                            })
                        })
                },
            )
        })
    }

    pub async fn load_ordinary_admitted_and_call(
        self: &Rc<Self>,
        name: &str,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
        source_owner: &SelectedProgramSourceOwner,
    ) -> crate::user_functions::DynamicFunctionLoadAttempt {
        if !source_owner.is_bound_in(&runtime) {
            return ControlFlow::Break(Err(runtime_error(
                "MEX:SourceAuthority",
                "ordinary MEX invocation differs from its issued session source",
            )));
        }
        if self.abandoned.get() || self.shutdown_in_progress.get() {
            return ControlFlow::Break(Err(runtime_error(
                "MEX:Lifecycle",
                "MEX modules cannot be invoked while the session is shutting down",
            )));
        }
        if runtime.is_cancelled() {
            return ControlFlow::Break(Err(runtime_error(
                "Cancelled",
                format!("MEX invocation '{name}' was cancelled before admission"),
            )));
        }
        if runtime.execution_stack() != RuntimeExecutionStack::Process {
            return ControlFlow::Break(Err(runtime_error(
                "Foreign:ExecutionStackViolation",
                "ordinary MEX invocation requires the process thread stack",
            )));
        }
        let selected = match self.observe_search_path_name(name).await {
            Ok(Some(MexNameSelection::SearchPath(selected))) => selected,
            Ok(None) => return ControlFlow::Continue(arguments),
            Ok(Some(MexNameSelection::Installed { .. })) => {
                return ControlFlow::Break(Err(runtime_error(
                    "MEX:SourceAuthority",
                    "ordinary source selected an installed module",
                )))
            }
            Err(error) => {
                return ControlFlow::Break(Err(error.into_identified_runtime_error(
                    "MEX function resolution",
                    "RunMat:FunctionResolution",
                )))
            }
        };
        ControlFlow::Break(
            self.load_ordinary_selected_and_call(
                name,
                arguments,
                requested_outputs,
                runtime,
                source_owner,
                selected,
            )
            .await
            .unwrap_or_else(|| {
                Err(runtime_error(
                    "MEX:SearchPathChanged",
                    "selected ordinary MEX target retired during invocation",
                ))
            }),
        )
    }

    async fn load_ordinary_selected_and_call(
        self: &Rc<Self>,
        name: &str,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
        source_owner: &SelectedProgramSourceOwner,
        expected_selection: FileCandidate,
    ) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
        if !source_owner.is_bound_in(&runtime) {
            return Some(Err(runtime_error(
                "MEX:SourceAuthority",
                "ordinary MEX invocation differs from its issued session source",
            )));
        }
        if self.abandoned.get() || self.shutdown_in_progress.get() {
            return Some(Err(runtime_error(
                "MEX:Lifecycle",
                "MEX modules cannot be invoked while the session is shutting down",
            )));
        }
        if runtime.is_cancelled() {
            return Some(Err(runtime_error(
                "Cancelled",
                format!("MEX invocation '{name}' was cancelled before admission"),
            )));
        }
        if runtime.execution_stack() != RuntimeExecutionStack::Process {
            return Some(Err(runtime_error(
                "Foreign:ExecutionStackViolation",
                "ordinary MEX invocation requires the process thread stack",
            )));
        }
        let path_revision = runtime.search_path().map(|path| path.generation());
        let selection = match self.observe_search_path_name(name).await {
            Ok(Some(MexNameSelection::SearchPath(candidate))) => candidate,
            Ok(None) => {
                return Some(Err(runtime_error(
                    "MEX:SearchPathChanged",
                    "selected ordinary MEX target retired before invocation",
                )))
            }
            Ok(Some(MexNameSelection::Installed { .. })) => unreachable!("search-path-only owner"),
            Err(error) => {
                return Some(Err(error.into_identified_runtime_error(
                    "MEX function resolution",
                    "RunMat:FunctionResolution",
                )))
            }
        };
        if selection != expected_selection {
            return Some(Err(runtime_error(
                "MEX:SearchPathChanged",
                "ordinary MEX selection changed before invocation",
            )));
        }
        let source = match std::fs::canonicalize(selection.path()) {
            Ok(source) => source,
            Err(error) => return Some(Err(runtime_error("MEX:Artifact", error.to_string()))),
        };
        let origin = selection.origin().clone();
        let (_, base_name) = crate::builtins::common::path_search::split_package_components(name);
        // Selection is refreshed on every call, but an already admitted
        // module owns immutable private bytes until a successful clear. Path
        // generation and origin are not module-lifetime identities.
        let cached = {
            let modules = self.ordinary_modules.borrow();
            modules
                .iter()
                .find(|artifact| artifact.source == source)
                .map(|artifact| {
                    (
                        artifact.private_module.clone(),
                        artifact.declared_name.clone(),
                        artifact.module_digest,
                        artifact.evidence.clone(),
                        artifact.identity.clone(),
                    )
                })
        };
        let (private_module, declared_name, module_digest, evidence, identity, pending) =
            match cached {
                Some((path, declared_name, module, evidence, identity)) => {
                    (path, declared_name, module, evidence, identity, None)
                }
                None => {
                    let captured = match CapturedOrdinaryMex::capture(
                        &source,
                        self.isolation_policy.borrow().unmanifested,
                    ) {
                        Ok(captured) => captured,
                        Err(error) => return Some(Err(error)),
                    };
                    if captured.name() != base_name {
                        return Some(Err(runtime_error(
                            "MEX:Artifact",
                            "selected MEX module name differs from the invoked callable",
                        )));
                    }
                    let mut descriptor = self.descriptor();
                    descriptor
                        .artifact_identities
                        .insert(captured.identity().clone());
                    if let Err(error) = admit_interop_manifest(
                        &captured.interop_manifest(),
                        &BTreeMap::from([(descriptor.adapter.clone(), descriptor)]),
                        ForeignPlatform::Native,
                    ) {
                        return Some(Err(error));
                    }
                    if let Err(error) = captured.require_ordinary_resources() {
                        return Some(Err(error));
                    }
                    let lease = match PrivateByteLease::new_private("runmat-ordinary-mex-") {
                        Ok(lease) => lease,
                        Err(error) => return Some(Err(runtime_error("MEX:Artifact", error))),
                    };
                    let Some(file_name) = source.file_name() else {
                        return Some(Err(runtime_error(
                            "MEX:Artifact",
                            "MEX module has no filename",
                        )));
                    };
                    let private_module = lease.path().join(file_name);
                    let private_manifest =
                        runmat_mex::MexArtifactManifest::path_for_module(&private_module);
                    let materialized = (|| {
                        write_private(&private_module, captured.bytes())?;
                        if let Some(bytes) = captured.manifest_bytes() {
                            write_private(&private_manifest, bytes)?;
                        }
                        Ok::<(), String>(())
                    })();
                    if let Err(error) = materialized {
                        return Some(Err(runtime_error("MEX:Artifact", error)));
                    }
                    // Compare the materialized copy in fixed-size chunks. A
                    // second capture would retain another full module Vec while
                    // the original admission buffers are still live.
                    if verify_private(&private_module, captured.bytes()).is_err()
                        || captured
                            .manifest_bytes()
                            .is_some_and(|bytes| verify_private(&private_manifest, bytes).is_err())
                    {
                        return Some(Err(runtime_error(
                            "MEX:Artifact",
                            "private MEX bytes differ from the admitted source artifact",
                        )));
                    }
                    let module_digest = Digest::sha256(captured.bytes());
                    let evidence = captured.evidence();
                    let identity = captured.identity().clone();
                    let pending = AdmittedOrdinaryMexArtifact {
                        source: source.clone(),
                        declared_name: captured.name().to_owned(),
                        callable_names: BTreeSet::from([name.to_owned()]),
                        module_digest,
                        evidence: evidence.clone(),
                        identity: identity.clone(),
                        private_module: private_module.clone(),
                        lease,
                    };
                    (
                        private_module,
                        captured.name().to_owned(),
                        module_digest,
                        evidence,
                        identity,
                        Some(pending),
                    )
                }
            };
        if let Err(error) = evidence.admit_policy(self.isolation_policy.borrow().unmanifested) {
            return Some(Err(error));
        }
        // A canonical path can be reached through another selected alias or
        // symlink. The pinned module's callable name, not the selected path
        // alone, remains the admission authority on every cache hit.
        if declared_name != base_name {
            return Some(Err(runtime_error(
                "MEX:Artifact",
                "selected MEX module name differs from the invoked callable",
            )));
        }
        let selected_now = match self.observe_search_path_name(name).await {
            Ok(selected) => selected,
            Err(error) => {
                return Some(Err(error.into_identified_runtime_error(
                    "MEX function resolution",
                    "RunMat:FunctionResolution",
                )))
            }
        };
        let same_selection = matches!(
            selected_now.as_ref(),
            Some(MexNameSelection::SearchPath(current)) if current == &selection
        );
        let same_canonical_source = selected_now
            .as_ref()
            .and_then(|selected| std::fs::canonicalize(selected.path()).ok())
            .as_deref()
            == Some(source.as_path());
        if runtime.search_path().map(|path| path.generation()) != path_revision
            || !same_selection
            || !same_canonical_source
        {
            return Some(Err(runtime_error(
                "MEX:SearchPathChanged",
                "MEX search path changed during artifact selection",
            )));
        }
        let receipt = OrdinaryMexInvocationReceipt {
            source_owner: source_owner.clone(),
            source,
            origin,
            path_revision,
            module_digest,
            evidence,
            identity,
        };
        if !receipt.source_owner.is_bound_in(&runtime)
            || receipt.path_revision != runtime.search_path().map(|path| path.generation())
            || &receipt.origin != selection.origin()
        {
            return Some(Err(runtime_error(
                "MEX:SourceAuthority",
                "ordinary MEX source retired during artifact admission",
            )));
        }
        if runtime.is_cancelled() {
            return Some(Err(runtime_error(
                "Cancelled",
                format!("MEX invocation '{name}' was cancelled after admission"),
            )));
        }
        if let Some(pending) = pending {
            if !receipt.matches_artifact(&pending) {
                return Some(Err(runtime_error(
                    "MEX:Artifact",
                    "ordinary MEX receipt differs from its captured bytes",
                )));
            }
            self.ordinary_modules.borrow_mut().push(pending);
        } else if !self.ordinary_modules.borrow().iter().any(|artifact| {
            artifact.private_module == private_module && receipt.matches_artifact(artifact)
        }) {
            return Some(Err(runtime_error(
                "MEX:Artifact",
                "ordinary MEX receipt lost its private bytes",
            )));
        } else if let Some(artifact) = self
            .ordinary_modules
            .borrow_mut()
            .iter_mut()
            .find(|artifact| artifact.private_module == private_module)
        {
            artifact.callable_names.insert(name.to_owned());
        }
        self.dispatch_admitted_path(private_module, arguments, requested_outputs, runtime)
            .await
    }
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|error| format!("create private MEX artifact: {error}"))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write private MEX artifact: {error}"))
}

fn verify_private(path: &Path, expected: &[u8]) -> Result<(), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("open private MEX artifact for verification: {error}"))?;
    let mut chunk = [0u8; 64 * 1024];
    for expected_chunk in expected.chunks(chunk.len()) {
        file.read_exact(&mut chunk[..expected_chunk.len()])
            .map_err(|error| format!("read private MEX artifact: {error}"))?;
        if &chunk[..expected_chunk.len()] != expected_chunk {
            return Err("private MEX artifact differs from captured bytes".into());
        }
    }
    if file
        .read(&mut chunk[..1])
        .map_err(|error| format!("finish private MEX artifact verification: {error}"))?
        != 0
    {
        return Err("private MEX artifact has trailing bytes".into());
    }
    Ok(())
}
