//! Stage one exact selected closure against the Session's existing adapter owners.

use runmat_runtime::foreign::{
    HostPolicyChange, MexRuntimeSession, NativeProjectArtifactExpectation,
    PreparedHostPolicyChanges, PreparedJavaProjectReplacement, PreparedMexArtifactCatalog,
    PreparedNativeProjectReplacement, PreparedPythonProjectReplacement,
    ReadyJavaProjectReplacement, ReadyMexArtifactCatalog, ReadyMexIsolationPolicy,
    ReadyNativeProjectReplacement, ReadyPythonProjectReplacement, VerifiedJavaProjectRetirement,
    VerifiedNativeProjectRetirement, VerifiedPythonProjectRetirement,
};

use crate::{foreign_unit_host::SelectedForeignOwners, ForeignUnitHost};

use super::{
    install::SelectedForeignInstallError, java_runtime, mex_policy, native_policy,
    selected_install_plan::SelectedInteropInstallPlan,
};

pub(crate) struct PreparedSelectedHostReplacement<'a> {
    runtime: &'a runmat_runtime::foreign::ForeignRuntime,
    mex: &'a MexRuntimeSession,
    native: Option<PreparedNativeProjectReplacement<'a>>,
    java: Option<PreparedJavaProjectReplacement<'a>>,
    python: Option<PreparedPythonProjectReplacement<'a>>,
    catalog: PreparedMexArtifactCatalog,
    mex_policy: Option<runmat_runtime::foreign::MexIsolationPolicy>,
}

pub(crate) struct ReadySelectedHostReplacement<'a> {
    native: Option<ReadyNativeProjectReplacement<'a>>,
    java: Option<ReadyJavaProjectReplacement<'a>>,
    python: Option<ReadyPythonProjectReplacement<'a>>,
    other_host: Option<PreparedHostPolicyChanges<'a>>,
    catalog: ReadyMexArtifactCatalog<'a>,
    mex_policy: Option<ReadyMexIsolationPolicy<'a>>,
}

pub(crate) struct VerifiedSelectedHostRetirement<'a> {
    native: Option<VerifiedNativeProjectRetirement<'a>>,
    java: Option<VerifiedJavaProjectRetirement<'a>>,
    python: Option<VerifiedPythonProjectRetirement<'a>>,
}

#[derive(Clone, Copy)]
enum ReplacementScope {
    RequiredFamilies,
    Clear,
}

#[derive(Clone, Copy)]
struct ProjectOwnedFamilies {
    native: bool,
    java: bool,
    python: bool,
}

impl ProjectOwnedFamilies {
    fn from_owners(
        owners: &SelectedForeignOwners<'_>,
    ) -> Result<Self, SelectedForeignInstallError> {
        Ok(Self {
            native: owners.native.has_project_artifacts()?,
            java: owners.java.has_project_artifacts()?,
            python: owners.python.has_project_artifacts()?,
        })
    }
}

impl<'a> PreparedSelectedHostReplacement<'a> {
    pub(crate) fn prepare_clear(
        host: &'a ForeignUnitHost,
    ) -> Result<Self, SelectedForeignInstallError> {
        let empty = runmat_types::InteropManifest::empty();
        let plan = SelectedInteropInstallPlan {
            private_byte_lease: None,
            manifest: &empty,
            policy: None,
            admission: None,
            native: Vec::new(),
            mex_catalog: PreparedMexArtifactCatalog::prepare(std::iter::empty::<
                runmat_runtime::foreign::MexArtifactExpectation<'_>,
            >())?,
            python: None,
            java: Vec::new(),
            requires_cuda: false,
        };
        Self::prepare_with_scope(host, plan, ReplacementScope::Clear)
    }

    pub(crate) fn prepare(
        host: &'a ForeignUnitHost,
        plan: SelectedInteropInstallPlan<'_>,
    ) -> Result<Self, SelectedForeignInstallError> {
        Self::prepare_with_scope(host, plan, ReplacementScope::RequiredFamilies)
    }

    fn prepare_with_scope(
        host: &'a ForeignUnitHost,
        plan: SelectedInteropInstallPlan<'_>,
        scope: ReplacementScope,
    ) -> Result<Self, SelectedForeignInstallError> {
        let owners = host.selected_owners();
        let prior = ProjectOwnedFamilies::from_owners(&owners)?;
        let SelectedForeignOwners {
            runtime,
            native,
            java,
            python,
            mex,
        } = owners;
        let SelectedInteropInstallPlan {
            private_byte_lease,
            manifest,
            policy,
            admission,
            native: native_paths,
            mex_catalog,
            python: python_bundle,
            java: java_paths,
            requires_cuda: _,
        } = plan;
        let clearing = matches!(scope, ReplacementScope::Clear);
        let next_native_required = admission
            .as_ref()
            .is_some_and(|admission| admission.native_required);
        let next_java_required = admission
            .as_ref()
            .is_some_and(|admission| admission.java_required);
        let next_python_required = admission
            .as_ref()
            .is_some_and(|admission| admission.python.is_some());
        let native_required = match scope {
            ReplacementScope::Clear => prior.native,
            ReplacementScope::RequiredFamilies => next_native_required || prior.native,
        };
        let java_required = match scope {
            ReplacementScope::Clear => prior.java,
            ReplacementScope::RequiredFamilies => next_java_required || prior.java,
        };
        let python_required = match scope {
            ReplacementScope::Clear => prior.python,
            ReplacementScope::RequiredFamilies => next_python_required || prior.python,
        };
        let next_native_policy = match (policy, admission.as_ref()) {
            (Some(policy), Some(admission)) if admission.native_required => native_policy(policy),
            _ => native.isolation_policy(),
        };
        let native = native_required
            .then(|| {
                native.prepare_project_replacement_with_policy(
                    next_native_policy,
                    native_paths.iter().map(|(library, sidecar)| {
                        NativeProjectArtifactExpectation {
                            library: library.as_path(),
                            manifest: sidecar.as_path(),
                            library_digest: runmat_execution::Digest::from_bytes(
                                *library.content_digest().bytes(),
                            ),
                            manifest_digest: runmat_execution::Digest::from_bytes(
                                *sidecar.content_digest().bytes(),
                            ),
                        }
                    }),
                )
            })
            .transpose()
            .map_err(SelectedForeignInstallError::from)?;
        let native = native
            .map(|staged| {
                if clearing || !next_native_required {
                    Ok(staged.clear_private_byte_lease())
                } else if let Some(lease) = private_byte_lease.as_ref() {
                    staged.with_private_byte_lease(lease.clone())
                } else {
                    Ok(staged)
                }
            })
            .transpose()?;
        let java_configuration = match (policy, admission.as_ref()) {
            (Some(policy), Some(admission)) if admission.java_required => {
                Some(java_runtime(policy))
            }
            _ => None,
        };
        let java_paths = java_paths
            .into_iter()
            .map(|(identity, path)| (identity, path.as_path().to_path_buf()))
            .collect::<Vec<_>>();
        let java = java_required
            .then(|| {
                java.prepare_project_replacement_with_configuration(java_configuration, &java_paths)
            })
            .transpose()
            .map_err(SelectedForeignInstallError::from)?;
        let java = java.map(|staged| {
            if clearing || !next_java_required {
                staged.clear_private_byte_lease()
            } else if let Some(lease) = private_byte_lease.as_ref() {
                staged.with_private_byte_lease(lease.clone())
            } else {
                staged
            }
        });
        let mex_policy = match (policy, admission.as_ref()) {
            (Some(policy), Some(admission)) if admission.mex_required => Some(mex_policy(policy)),
            _ => None,
        };
        let python_configuration = admission.and_then(|admission| admission.python);
        let python = python_required
            .then(|| {
                python.prepare_project_replacement_with_configuration(
                    python_configuration,
                    python_bundle,
                )
            })
            .transpose()
            .map_err(SelectedForeignInstallError::from)?;
        let python = python.map(|staged| {
            if clearing || !next_python_required {
                staged.clear_private_byte_lease()
            } else if let Some(lease) = private_byte_lease.as_ref() {
                staged.with_private_byte_lease(lease.clone())
            } else {
                staged
            }
        });
        runtime
            .admit_with_prepared_selected_replacements(
                manifest,
                native.as_ref(),
                &mex_catalog,
                java.as_ref(),
                python.as_ref(),
            )
            .map_err(SelectedForeignInstallError::from)?;
        Ok(Self {
            runtime,
            mex,
            native,
            java,
            python,
            catalog: mex_catalog,
            mex_policy,
        })
    }

    /// Fence the exact currently published project hosts before any selected
    /// inventory or source changes. Retained owners remain fenced for retry.
    pub(crate) async fn retire_project_hosts(
        &self,
        runtime: Option<&runmat_runtime::context::RuntimeContext>,
    ) -> Result<VerifiedSelectedHostRetirement<'a>, SelectedForeignInstallError> {
        let native = if let Some(staged) = self.native.as_ref() {
            Some(staged.retire_owner(runtime).await?)
        } else {
            None
        };
        let java = self
            .java
            .as_ref()
            .map(PreparedJavaProjectReplacement::retire_owner)
            .transpose()?;
        let python = if let Some(staged) = self.python.as_ref() {
            Some(staged.retire_owner(runtime).await?)
        } else {
            None
        };
        Ok(VerifiedSelectedHostRetirement {
            native,
            java,
            python,
        })
    }

    /// Acquire all adapter cells first, then the one shared host-registry lock.
    /// No await or registry query is allowed until the ready owner commits.
    pub(crate) fn admit_commit(
        self,
    ) -> Result<ReadySelectedHostReplacement<'a>, SelectedForeignInstallError> {
        self.admit_commit_inner(None)
    }

    pub(crate) fn admit_commit_after_retirement(
        self,
        retirement: VerifiedSelectedHostRetirement<'a>,
    ) -> Result<ReadySelectedHostReplacement<'a>, SelectedForeignInstallError> {
        self.admit_commit_inner(Some(retirement))
    }

    fn admit_commit_inner(
        self,
        retirement: Option<VerifiedSelectedHostRetirement<'a>>,
    ) -> Result<ReadySelectedHostReplacement<'a>, SelectedForeignInstallError> {
        let mut changes = Vec::<HostPolicyChange<'a>>::with_capacity(3);
        if let Some(native) = self.native.as_ref() {
            if let Some(change) = native.host_policy_change() {
                changes.push(change);
            }
        }
        if let Some(java) = self.java.as_ref() {
            changes.push(java.host_restart_change());
        }
        if let Some(python) = self.python.as_ref() {
            changes.push(python.host_policy_change());
        }
        let (native_ticket, java_ticket, python_ticket) = retirement
            .map(|retirement| (retirement.native, retirement.java, retirement.python))
            .unwrap_or((None, None, None));
        let native = match (self.native, native_ticket) {
            (Some(staged), Some(ticket)) => Some(staged.admit_commit_after_retirement(ticket)?),
            (Some(staged), None) => Some(staged.admit_commit()?),
            (None, None) => None,
            (None, Some(_)) => unreachable!("Native ticket requires a staged owner"),
        };
        let java = match (self.java, java_ticket) {
            (Some(staged), Some(ticket)) => Some(staged.admit_commit_after_retirement(ticket)?),
            (Some(staged), None) => Some(staged.admit_commit()?),
            (None, None) => None,
            (None, Some(_)) => unreachable!("Java ticket requires a staged owner"),
        };
        let catalog = self.mex.admit_artifact_catalog(self.catalog)?;
        let mex_policy = self
            .mex_policy
            .map(|policy| self.mex.admit_isolation_policy(policy))
            .transpose()?;
        let python = match (self.python, python_ticket) {
            (Some(staged), Some(ticket)) => Some(staged.admit_local_after_retirement(ticket)?),
            (Some(staged), None) => Some(staged.admit_local()?),
            (None, None) => None,
            (None, Some(_)) => unreachable!("Python ticket requires a staged owner"),
        };
        let host = (!changes.is_empty())
            .then(|| self.runtime.handles().prepare_host_policy_changes(&changes))
            .transpose()?;
        let (python, other_host) = match (python, host) {
            (Some(python), Some(host)) => (Some(python.admit_with_shared_host(host)?), None),
            (None, host) => (None, host),
            (Some(_), None) => unreachable!("Python stage always requests host reconfiguration"),
        };
        Ok(ReadySelectedHostReplacement {
            native,
            java,
            python,
            other_host,
            catalog,
            mex_policy,
        })
    }
}

impl<'a> ReadySelectedHostReplacement<'a> {
    pub(crate) fn commit(self) -> VerifiedSelectedHostRetirement<'a> {
        let native = self.native.and_then(ReadyNativeProjectReplacement::commit);
        let java = self.java.and_then(ReadyJavaProjectReplacement::commit);
        let python = self.python.and_then(ReadyPythonProjectReplacement::commit);
        if let Some(host) = self.other_host {
            host.commit();
        }
        if let Some(policy) = self.mex_policy {
            policy.commit();
        }
        self.catalog.commit();
        VerifiedSelectedHostRetirement {
            native,
            java,
            python,
        }
    }
}
