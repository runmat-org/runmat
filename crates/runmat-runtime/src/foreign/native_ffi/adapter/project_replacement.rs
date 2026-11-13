//! Validate and stage a complete project-owned native interface inventory.

use super::*;
use std::path::Path;

/// Exact frozen bytes expected by the caller's project handoff.
pub struct NativeProjectArtifactExpectation<'a> {
    pub library: &'a Path,
    pub manifest: &'a Path,
    pub library_digest: runmat_execution::Digest,
    pub manifest_digest: runmat_execution::Digest,
}

pub struct PreparedNativeProjectReplacement<'a> {
    owner: &'a NativeFfiAdapter,
    libraries: BTreeMap<String, LibraryEntry>,
    catalog: NativeProjectCatalog,
    identities: BTreeSet<runmat_types::ForeignArtifactIdentity>,
    aliases: BTreeSet<String>,
    manifests: Vec<runmat_types::InteropManifest>,
    baseline_policy: NativeFfiIsolationPolicy,
    next_policy: NativeFfiIsolationPolicy,
    retirement_identity: Rc<()>,
    restart_required: bool,
    next_lease: Option<Option<runmat_process_host::private_bytes::PrivateByteLease>>,
}

pub struct ReadyNativeProjectReplacement<'a> {
    staged: PreparedNativeProjectReplacement<'a>,
    state: std::cell::RefMut<'a, NativeFfiSessionState>,
    catalog: std::cell::RefMut<'a, NativeProjectCatalog>,
    identities: std::cell::RefMut<'a, BTreeSet<String>>,
    _client: std::cell::RefMut<'a, Option<NativeFfiHostClient>>,
    lease: std::cell::RefMut<'a, Option<runmat_process_host::private_bytes::PrivateByteLease>>,
    retirement: Option<VerifiedNativeProjectRetirement<'a>>,
}

impl<'a> PreparedNativeProjectReplacement<'a> {
    pub fn with_private_byte_lease(
        mut self,
        lease: runmat_process_host::private_bytes::PrivateByteLease,
    ) -> Result<Self, RuntimeError> {
        for entry in self.libraries.values_mut() {
            let library = Rc::get_mut(&mut entry.library).ok_or_else(|| {
                invalid_call("prepared native library escaped before private byte admission")
            })?;
            library
                .attach_private_byte_lease(lease.clone())
                .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        }
        self.catalog.bind_private_root(lease.clone())?;
        self.next_lease = Some(Some(lease));
        Ok(self)
    }

    pub fn clear_private_byte_lease(mut self) -> Self {
        self.catalog.lease = None;
        self.next_lease = Some(None);
        self
    }

    pub async fn retire_owner(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<VerifiedNativeProjectRetirement<'a>, RuntimeError> {
        let owner: &'a NativeFfiAdapter = self.owner;
        let identity = Rc::clone(&self.retirement_identity);
        owner.retire_project_session(&identity, runtime).await
    }

    pub fn artifact_identities(
        &self,
    ) -> impl Iterator<Item = &runmat_types::ForeignArtifactIdentity> {
        self.identities.iter()
    }

    pub fn interop_manifests(&self) -> impl Iterator<Item = &runmat_types::InteropManifest> {
        self.manifests.iter()
    }

    pub fn host_policy_change(&self) -> Option<super::super::super::HostPolicyChange<'a>> {
        if !self.restart_required {
            return None;
        }
        let owner: &'a NativeFfiAdapter = self.owner;
        let policy = if self.next_policy.isolate {
            ForeignExecutionPolicy {
                trust: super::super::super::ForeignTrust::Untrusted,
                isolation: super::super::super::ForeignIsolation::IsolatedProcess,
            }
        } else {
            ForeignExecutionPolicy::trusted_in_process()
        };
        Some(super::super::super::HostPolicyChange::Restart {
            host_identity: owner.host_identity.as_str(),
            policy,
        })
    }

    /// Check the final host lifecycle after any required retirement and before
    /// the caller's last fallible publication step.
    pub fn admit_commit(self) -> Result<ReadyNativeProjectReplacement<'a>, RuntimeError> {
        self.admit_commit_with_retirement(None)
    }

    pub fn admit_commit_after_retirement(
        self,
        retirement: VerifiedNativeProjectRetirement<'a>,
    ) -> Result<ReadyNativeProjectReplacement<'a>, RuntimeError> {
        if !std::ptr::eq(self.owner, retirement.owner) {
            return Err(invalid_call(
                "native project retirement belongs to another adapter",
            ));
        }
        self.admit_commit_with_retirement(Some(retirement))
    }

    fn admit_commit_with_retirement(
        self,
        retirement: Option<VerifiedNativeProjectRetirement<'a>>,
    ) -> Result<ReadyNativeProjectReplacement<'a>, RuntimeError> {
        let authority = &self.owner.project_retirement;
        let retirement_matches = match &retirement {
            Some(ticket) => {
                Rc::ptr_eq(ticket.authority.identity(), &self.retirement_identity)
                    && authority.matches_verified(ticket.authority.identity())
            }
            None => authority.matches_idle(&self.retirement_identity),
        };
        if !retirement_matches {
            return Err(invalid_call(
                "native project retirement authority changed after preparation",
            ));
        }
        if !self.catalog.is_empty() && self.catalog.lease.is_none() {
            return Err(invalid_call(
                "isolated native project catalog lacks its private byte owner",
            ));
        }
        self.owner.ensure_project_replacement_available()?;
        if self.owner.isolation_policy.get() != self.baseline_policy {
            return Err(invalid_call(
                "native project isolation policy changed after preparation",
            ));
        }
        let owner = self.owner;
        let state = owner
            .state
            .try_borrow_mut()
            .map_err(|_| invalid_call("native project state is busy during publication"))?;
        if self.next_policy.isolate != self.baseline_policy.isolate
            && state.libraries.values().any(|entry| !entry.project_owned)
        {
            return Err(invalid_call(
                "native isolation cannot change while user libraries remain loaded",
            ));
        }
        if let Some(alias) = self.aliases.iter().find(|alias| {
            state
                .libraries
                .get(*alias)
                .is_some_and(|entry| !entry.project_owned)
        }) {
            return Err(invalid_call(format!(
                "native project interface alias `{alias}` became user-owned after preparation"
            )));
        }
        let catalog = owner
            .project_catalog
            .try_borrow_mut()
            .map_err(|_| invalid_call("native project catalog is busy during publication"))?;
        let identities = owner
            .prepared_artifact_identities
            .try_borrow_mut()
            .map_err(|_| invalid_call("native project identities are busy during publication"))?;
        let client = owner
            .isolated_client
            .try_borrow_mut()
            .map_err(|_| invalid_call("native isolated client is busy during publication"))?;
        if client.is_some() {
            return Err(invalid_call(
                "native project replacement requires confirmed isolated host retirement",
            ));
        }
        let lease = owner
            .private_byte_lease
            .try_borrow_mut()
            .map_err(|_| invalid_call("native private byte lease is busy during publication"))?;
        Ok(ReadyNativeProjectReplacement {
            staged: self,
            state,
            catalog,
            identities,
            _client: client,
            lease,
            retirement,
        })
    }
}

impl<'a> ReadyNativeProjectReplacement<'a> {
    /// The Session publication gate excludes concurrent adapter calls between
    /// `admit_commit` and this infallible owner replacement.
    pub fn commit(self) -> Option<VerifiedNativeProjectRetirement<'a>> {
        let ReadyNativeProjectReplacement {
            staged,
            mut state,
            mut catalog,
            mut identities,
            _client,
            mut lease,
            retirement,
        } = self;
        if let Some(next) = staged.next_lease {
            staged.owner.private_root_origin.set(if next.is_some() {
                NativePrivateRootOrigin::AdmittedRoot
            } else {
                NativePrivateRootOrigin::None
            });
            *lease = next;
        }
        state.libraries.retain(|_, entry| !entry.project_owned);
        state.libraries.extend(staged.libraries);
        *catalog = staged.catalog;
        *identities = if staged.owner.uses_host(staged.next_policy) {
            staged
                .identities
                .into_iter()
                .map(|identity| identity.to_string())
                .collect()
        } else {
            BTreeSet::new()
        };
        staged.owner.isolation_policy.set(staged.next_policy);
        if retirement.is_none() {
            staged
                .owner
                .project_retirement
                .publish_fresh(&staged.retirement_identity);
        }
        retirement
    }
}

pub(super) struct PreparedNativeArtifact {
    pub(super) alias: String,
    pub(super) identity: runmat_types::ForeignArtifactIdentity,
    pub(super) library: Option<LibraryEntry>,
    pub(super) project_load: Option<NativeProjectLoadRequest>,
    pub(super) interop: runmat_types::InteropManifest,
}

impl NativeFfiAdapter {
    /// Exact inventory installed through project replacement, excluding user libraries.
    pub fn has_project_artifacts(&self) -> Result<bool, RuntimeError> {
        let state = self
            .state
            .try_borrow()
            .map_err(|_| invalid_call("native libraries are busy during project preparation"))?;
        let catalog = self.project_catalog.try_borrow().map_err(|_| {
            invalid_call("native project catalog is busy during project preparation")
        })?;
        let identities = self
            .prepared_artifact_identities
            .try_borrow()
            .map_err(|_| {
                invalid_call("native project identities are busy during project preparation")
            })?;
        Ok(state.libraries.values().any(|entry| entry.project_owned)
            || !catalog.is_empty()
            || !identities.is_empty())
    }

    pub fn prepare_project_replacement<'a, 'paths>(
        &'a self,
        artifacts: impl IntoIterator<Item = NativeProjectArtifactExpectation<'paths>>,
    ) -> Result<PreparedNativeProjectReplacement<'a>, RuntimeError> {
        self.prepare_project_replacement_with_policy(self.isolation_policy.get(), artifacts)
    }

    pub fn prepare_project_replacement_with_policy<'a, 'paths>(
        &'a self,
        next_policy: NativeFfiIsolationPolicy,
        artifacts: impl IntoIterator<Item = NativeProjectArtifactExpectation<'paths>>,
    ) -> Result<PreparedNativeProjectReplacement<'a>, RuntimeError> {
        let mut libraries = BTreeMap::new();
        let mut catalog = NativeProjectCatalog::default();
        let mut identities = BTreeSet::new();
        let mut manifests = Vec::new();
        let mut aliases = BTreeSet::new();
        let baseline_policy = self.isolation_policy.get();
        let restart_required = next_policy.isolate != baseline_policy.isolate
            || self.isolated_client.borrow().is_some();
        let retirement_identity = self.project_retirement.snapshot();
        if next_policy.isolate != baseline_policy.isolate {
            self.handles.admit_host_restart(&self.host_identity)?;
        }
        if next_policy.isolate != baseline_policy.isolate
            && self
                .state
                .borrow()
                .libraries
                .values()
                .any(|entry| !entry.project_owned)
        {
            return Err(invalid_call(
                "native isolation cannot change while user libraries remain loaded",
            ));
        }
        for expected in artifacts {
            let artifact = self.prepare_project_artifact_with_policy(
                expected.library,
                expected.manifest,
                Some((expected.library_digest, expected.manifest_digest)),
                next_policy,
                None,
            )?;
            if !aliases.insert(artifact.alias.clone())
                || self
                    .state
                    .borrow()
                    .libraries
                    .get(&artifact.alias)
                    .is_some_and(|entry| !entry.project_owned)
            {
                return Err(invalid_call(format!(
                    "native project interface alias `{}` conflicts with another interface",
                    artifact.alias
                )));
            }
            if !identities.insert(artifact.identity) {
                return Err(invalid_call(
                    "native project interface identity appears more than once",
                ));
            }
            manifests.push(artifact.interop);
            if let Some(entry) = artifact.library {
                libraries.insert(artifact.alias, entry);
            } else if let Some(load) = artifact.project_load {
                catalog.loads.push(load);
            }
        }
        Ok(PreparedNativeProjectReplacement {
            owner: self,
            libraries,
            catalog,
            identities,
            aliases,
            manifests,
            baseline_policy,
            next_policy,
            retirement_identity,
            restart_required,
            next_lease: None,
        })
    }

    fn ensure_project_replacement_available(&self) -> Result<(), RuntimeError> {
        if self.isolated_call_active.get()
            || self.bootstrap_unverified.is_unverified()
            || self.isolated_client.borrow().is_some()
        {
            return Err(invalid_call(
                "native project interfaces cannot be replaced during an active or unretired isolated host",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_project_artifact(
        &self,
        library_path: &Path,
        manifest_path: &Path,
        expected: Option<(runmat_execution::Digest, runmat_execution::Digest)>,
    ) -> Result<PreparedNativeArtifact, RuntimeError> {
        self.prepare_project_artifact_with_policy(
            library_path,
            manifest_path,
            expected,
            self.isolation_policy.get(),
            None,
        )
    }

    pub(super) fn prepare_project_artifact_with_policy(
        &self,
        library_path: &Path,
        manifest_path: &Path,
        expected: Option<(runmat_execution::Digest, runmat_execution::Digest)>,
        policy: NativeFfiIsolationPolicy,
        declared: Option<(&str, &runmat_types::ForeignArtifactIdentity)>,
    ) -> Result<PreparedNativeArtifact, RuntimeError> {
        let manifest_bytes = std::fs::read(manifest_path).map_err(|error| {
            foreign_error(
                ForeignErrorKind::LoadFailed,
                format!(
                    "could not read native interface manifest {}: {error}",
                    manifest_path.display()
                ),
            )
        })?;
        if expected
            .is_some_and(|(_, digest)| runmat_execution::Digest::sha256(&manifest_bytes) != digest)
        {
            return Err(invalid_call(
                "native project manifest changed after resolution",
            ));
        }
        let manifest = NativeInterfaceArtifactManifest::from_canonical_bytes(&manifest_bytes)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let library_bytes = std::fs::read(library_path).map_err(|error| {
            foreign_error(
                ForeignErrorKind::LoadFailed,
                format!(
                    "could not read native project library {}: {error}",
                    library_path.display()
                ),
            )
        })?;
        if expected
            .is_some_and(|(digest, _)| runmat_execution::Digest::sha256(&library_bytes) != digest)
        {
            return Err(invalid_call(
                "native project library changed after resolution",
            ));
        }
        manifest
            .validate_current_library(&library_bytes)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let alias = manifest.interface_name.clone();
        let identity = runmat_types::ForeignArtifactIdentity::new(manifest.identity.to_string())
            .map_err(|error| invalid_call(error.to_string()))?;
        if let Some((declared_alias, declared_identity)) = declared {
            if alias != declared_alias || identity != *declared_identity {
                return Err(invalid_call(
                    "native project load differs from its exact identity",
                ));
            }
            self.validate_available_alias(&alias)?;
        }
        let interop = manifest.interop_manifest();
        if self.uses_host(policy) {
            return Ok(PreparedNativeArtifact {
                alias: alias.clone(),
                identity: identity.clone(),
                library: None,
                project_load: Some(NativeProjectLoadRequest {
                    request_id: 1, // replaced by the child client's request sequence
                    private_root: std::path::PathBuf::new(),
                    library: library_path.to_path_buf(),
                    manifest: manifest_path.to_path_buf(),
                    alias,
                    artifact_identity: identity,
                    library_digest: expected.map_or_else(
                        || runmat_execution::Digest::sha256(&library_bytes),
                        |(digest, _)| digest,
                    ),
                    manifest_digest: expected.map_or_else(
                        || runmat_execution::Digest::sha256(&manifest_bytes),
                        |(_, digest)| digest,
                    ),
                }),
                interop,
            });
        }
        let metadata = manifest
            .materialized_metadata(library_path)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let (library, _) =
            self.prepare_library_entry(library_path, metadata, identity.to_string(), false, true)?;
        Ok(PreparedNativeArtifact {
            alias,
            identity,
            library: Some(library),
            project_load: None,
            interop,
        })
    }
}
