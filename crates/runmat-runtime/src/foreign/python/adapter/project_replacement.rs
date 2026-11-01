//! Prepare a complete Python project wheel replacement before publication.

use super::*;
use crate::foreign::{HostPolicyChange, PreparedHostPolicyChanges};

pub struct PreparedPythonProjectReplacement<'a> {
    owner: &'a PythonAdapter,
    retirement_identity: Rc<()>,
    baseline: PythonRuntimeConfiguration,
    updated: PythonRuntimeConfiguration,
    installed: Option<InstalledPythonArtifacts>,
    project_identities: BTreeSet<runmat_types::ForeignArtifactIdentity>,
    next_lease: Option<Option<runmat_process_host::private_bytes::PrivateByteLease>>,
}

pub struct ReadyPythonProjectReplacementLocal<'a> {
    staged: PreparedPythonProjectReplacement<'a>,
    _retirement: Option<VerifiedPythonProjectRetirement<'a>>,
    configuration: std::cell::RefMut<'a, PythonRuntimeConfiguration>,
    installed: std::cell::RefMut<'a, Option<InstalledPythonArtifacts>>,
    _session: std::cell::RefMut<'a, Option<PythonSession>>,
    _client: std::cell::RefMut<'a, Option<IsolatedPythonClient>>,
    lease: std::cell::RefMut<'a, Option<runmat_process_host::private_bytes::PrivateByteLease>>,
}

pub struct ReadyPythonProjectReplacement<'a> {
    local: ReadyPythonProjectReplacementLocal<'a>,
    host: PreparedHostPolicyChanges<'a>,
}

impl<'a> PreparedPythonProjectReplacement<'a> {
    pub fn with_private_byte_lease(
        mut self,
        lease: runmat_process_host::private_bytes::PrivateByteLease,
    ) -> Self {
        self.next_lease = Some(Some(lease));
        self
    }

    pub fn clear_private_byte_lease(mut self) -> Self {
        self.next_lease = Some(None);
        self
    }

    pub fn artifact_identities(
        &self,
    ) -> impl Iterator<Item = &runmat_types::ForeignArtifactIdentity> {
        self.project_identities.iter()
    }

    pub fn host_policy_change(&self) -> HostPolicyChange<'a> {
        let owner: &'a PythonAdapter = self.owner;
        HostPolicyChange::Restart {
            host_identity: owner.host_identity.as_str(),
            policy: policy(self.updated.execution_mode),
        }
    }

    pub async fn retire_owner(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<VerifiedPythonProjectRetirement<'a>, RuntimeError> {
        let owner: &'a PythonAdapter = self.owner;
        owner
            .retire_project_session(&self.retirement_identity, runtime)
            .await
    }

    /// Acquire the live adapter and registry owners after required retirement.
    pub fn admit_commit(self) -> Result<ReadyPythonProjectReplacement<'a>, RuntimeError> {
        let owner = self.owner;
        let local = self.admit_local()?;
        let host = owner
            .handles
            .prepare_host_policy_changes(&[HostPolicyChange::Restart {
                host_identity: &owner.host_identity,
                policy: policy(local.staged.updated.execution_mode),
            }])?;
        Ok(ReadyPythonProjectReplacement { local, host })
    }

    pub fn admit_local(self) -> Result<ReadyPythonProjectReplacementLocal<'a>, RuntimeError> {
        self.admit_local_inner(None)
    }

    pub fn admit_local_after_retirement(
        self,
        retirement: VerifiedPythonProjectRetirement<'a>,
    ) -> Result<ReadyPythonProjectReplacementLocal<'a>, RuntimeError> {
        self.admit_local_inner(Some(retirement))
    }

    fn admit_local_inner(
        self,
        retirement: Option<VerifiedPythonProjectRetirement<'a>>,
    ) -> Result<ReadyPythonProjectReplacementLocal<'a>, RuntimeError> {
        let owner = self.owner;
        let retirement_matches = retirement.as_ref().map_or_else(
            || {
                owner
                    .project_retirement
                    .matches_idle(&self.retirement_identity)
            },
            |ticket| {
                std::ptr::eq(ticket.owner, owner)
                    && owner
                        .project_retirement
                        .matches_verified(ticket.authority.identity())
                    && Rc::ptr_eq(ticket.authority.identity(), &self.retirement_identity)
            },
        );
        if !retirement_matches {
            return Err(invalid_call(
                "Python project retirement ticket does not match the prepared owner",
            ));
        }
        if owner.retirement_requested.get()
            || owner.isolated_call_active.get()
            || owner.spawn_unverified.is_unverified()
        {
            return Err(invalid_call(
                "Python project artifacts require confirmed host retirement",
            ));
        }
        let session = owner
            .session
            .try_borrow_mut()
            .map_err(|_| invalid_call("Python session is busy during project publication"))?;
        let client = owner
            .isolated_client
            .try_borrow_mut()
            .map_err(|_| invalid_call("Python child is busy during project publication"))?;
        if session.is_some() || client.is_some() {
            return Err(invalid_call(
                "Python project artifacts cannot change after interpreter startup",
            ));
        }
        let configuration = owner
            .configuration
            .try_borrow_mut()
            .map_err(|_| invalid_call("Python configuration is busy during project publication"))?;
        if *configuration != self.baseline {
            return Err(invalid_call(
                "Python configuration changed after project preparation",
            ));
        }
        let installed = owner.installed_artifacts.try_borrow_mut().map_err(|_| {
            invalid_call("Python installed roots are busy during project publication")
        })?;
        let lease = owner.selected_byte_lease.try_borrow_mut().map_err(|_| {
            invalid_call("Python private byte lease is busy during project publication")
        })?;
        Ok(ReadyPythonProjectReplacementLocal {
            staged: self,
            _retirement: retirement,
            configuration,
            installed,
            _session: session,
            _client: client,
            lease,
        })
    }
}

impl<'a> ReadyPythonProjectReplacementLocal<'a> {
    pub fn admit_with_shared_host(
        self,
        host: PreparedHostPolicyChanges<'a>,
    ) -> Result<ReadyPythonProjectReplacement<'a>, RuntimeError> {
        if !host.contains_restart(
            &self.staged.owner.host_identity,
            policy(self.staged.updated.execution_mode),
        ) {
            return Err(invalid_call(
                "shared Python host guard lacks the prepared policy change",
            ));
        }
        Ok(ReadyPythonProjectReplacement { local: self, host })
    }
}

impl<'a> ReadyPythonProjectReplacement<'a> {
    /// No await or foreign call may intervene between admission and commit.
    pub fn commit(self) -> Option<VerifiedPythonProjectRetirement<'a>> {
        let ReadyPythonProjectReplacement { local, host } = self;
        let ReadyPythonProjectReplacementLocal {
            staged,
            _retirement,
            mut configuration,
            mut installed,
            _session,
            _client,
            mut lease,
        } = local;
        host.commit();
        *configuration = staged.updated;
        *installed = staged.installed;
        if let Some(next) = staged.next_lease {
            *lease = next;
        }
        staged.owner.terminated.set(false);
        if _retirement.is_none() {
            staged
                .owner
                .project_retirement
                .publish_fresh(&staged.retirement_identity);
        }
        _retirement
    }
}

impl PythonAdapter {
    /// Installed project wheel roots, excluding explicit configuration paths.
    pub fn has_project_artifacts(&self) -> Result<bool, RuntimeError> {
        self.installed_artifacts
            .try_borrow()
            .map(|installed| installed.is_some())
            .map_err(|_| invalid_call("Python installed roots are busy during project preparation"))
    }

    pub fn prepare_project_replacement(
        &self,
        bundle: Option<&PythonArtifactBundle>,
    ) -> Result<PreparedPythonProjectReplacement<'_>, RuntimeError> {
        self.prepare_project_replacement_with_configuration(None, bundle)
    }

    pub fn prepare_project_replacement_with_configuration(
        &self,
        configuration: Option<PythonRuntimeConfiguration>,
        bundle: Option<&PythonArtifactBundle>,
    ) -> Result<PreparedPythonProjectReplacement<'_>, RuntimeError> {
        if self.isolated_call_active.get() || self.retirement_requested.get() {
            return Err(invalid_call(
                "Python artifacts cannot change during an active call or terminal retirement",
            ));
        }
        self.handles.admit_host_restart(&self.host_identity)?;
        let baseline = self.configuration.borrow().clone();
        let explicit_configuration = configuration.is_some();
        let mut updated = configuration.unwrap_or_else(|| baseline.clone());
        if !explicit_configuration {
            if let Some(prior) = self.installed_artifacts.borrow().as_ref() {
                updated
                    .module_paths
                    .retain(|path| !prior.module_paths().contains(path));
                updated
                    .artifact_identities
                    .retain(|identity| !prior.artifact_identities().contains(identity));
            }
        }
        let mut project_identities = BTreeSet::new();
        let installed = if let Some(bundle) = bundle {
            bundle.validate().map_err(python_artifact_error)?;
            validate_bundle_environment(bundle, &updated)?;
            if explicit_configuration {
                pin_configuration_to_bundle_environment(&mut updated, bundle);
            }
            let installed = bundle.install().map_err(python_artifact_error)?;
            project_identities = installed
                .artifact_identities()
                .iter()
                .map(|identity| {
                    runmat_types::ForeignArtifactIdentity::new(identity.clone())
                        .map_err(|error| invalid_call(error.to_string()))
                })
                .collect::<Result<_, _>>()?;
            updated
                .module_paths
                .extend_from_slice(installed.module_paths());
            updated
                .artifact_identities
                .extend(installed.artifact_identities().iter().cloned());
            Some(installed)
        } else {
            None
        };
        Ok(PreparedPythonProjectReplacement {
            owner: self,
            retirement_identity: self.project_retirement.snapshot(),
            baseline,
            updated,
            installed,
            project_identities,
            next_lease: None,
        })
    }
}
