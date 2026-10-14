//! Stage exact project jars separately from configured and dynamic classpaths.

use super::*;
use std::path::PathBuf;

pub(super) struct PreparedJavaConfiguration {
    pub(super) discovery: JavaDiscoveryRequest,
    pub(super) config: JvmConfig,
    pub(super) configured_classpath: Vec<PathBuf>,
    pub(super) classpath: SessionClasspath,
    pub(super) next_revision: u64,
}

pub(super) fn prepare_configuration(
    configuration: JavaRuntimeConfiguration,
    current_revision: u64,
) -> Result<PreparedJavaConfiguration, RuntimeError> {
    let mut discovery = JavaDiscoveryRequest::from_process();
    discovery.explicit_home = configuration.home;
    let config = JvmConfig {
        minimum_major: configuration.minimum_version,
        maximum_major: configuration.maximum_version,
        bootstrap_classpath: Vec::new(),
        options: configuration.options,
    };
    config.validate().map_err(java_runtime_error)?;
    let configured_classpath = configuration.classpath;
    let classpath = SessionClasspath::new(
        config.bootstrap_classpath.clone(),
        configured_classpath.iter().cloned(),
    )
    .map_err(|error| invalid_conversion(error.to_string()))?;
    let next_revision = current_revision
        .checked_add(1)
        .ok_or_else(|| invalid_conversion("Java configuration revision exhausted"))?;
    Ok(PreparedJavaConfiguration {
        discovery,
        config,
        configured_classpath,
        classpath,
        next_revision,
    })
}

pub struct PreparedJavaProjectReplacement<'a> {
    owner: &'a JavaAdapter,
    retirement_identity: Rc<()>,
    classpath: SessionClasspath,
    identities: BTreeSet<JavaArtifactIdentity>,
    baseline_identity: String,
    configured_classpath_revision: u64,
    installation: Option<JvmInstallation>,
    next_configuration: Option<PreparedJavaConfiguration>,
    next_lease: Option<Option<PrivateByteLease>>,
}

pub struct ReadyJavaProjectReplacement<'a> {
    staged: PreparedJavaProjectReplacement<'a>,
    _retirement: Option<VerifiedJavaProjectRetirement<'a>>,
    classpath: std::cell::RefMut<'a, SessionClasspath>,
    identities: std::cell::RefMut<'a, BTreeSet<JavaArtifactIdentity>>,
    configured_classpath: std::cell::RefMut<'a, Vec<PathBuf>>,
    discovery: std::cell::RefMut<'a, JavaDiscoveryRequest>,
    config: std::cell::RefMut<'a, JvmConfig>,
    _session: std::cell::RefMut<'a, Option<Rc<JavaLane>>>,
    lease: std::cell::RefMut<'a, Option<PrivateByteLease>>,
}

impl<'a> PreparedJavaProjectReplacement<'a> {
    pub fn with_private_byte_lease(mut self, lease: PrivateByteLease) -> Self {
        self.next_lease = Some(Some(lease));
        self
    }

    pub fn clear_private_byte_lease(mut self) -> Self {
        self.next_lease = Some(None);
        self
    }

    pub fn artifact_identities(&self) -> impl Iterator<Item = &JavaArtifactIdentity> {
        self.identities.iter()
    }

    pub fn host_restart_change(&self) -> super::super::super::HostPolicyChange<'a> {
        let owner: &'a JavaAdapter = self.owner;
        super::super::super::HostPolicyChange::Restart {
            host_identity: owner.host_identity.as_str(),
            policy: ForeignExecutionPolicy::trusted_in_process(),
        }
    }

    pub fn retire_owner(&self) -> Result<VerifiedJavaProjectRetirement<'a>, RuntimeError> {
        let owner: &'a JavaAdapter = self.owner;
        owner.retire_project_session(&self.retirement_identity)
    }

    /// Recheck live JVM and classpath state after all other preparation, then
    /// hold the owners exclusively through one synchronous publication commit.
    pub fn admit_commit(self) -> Result<ReadyJavaProjectReplacement<'a>, RuntimeError> {
        self.admit_commit_inner(None)
    }

    pub fn admit_commit_after_retirement(
        self,
        retirement: VerifiedJavaProjectRetirement<'a>,
    ) -> Result<ReadyJavaProjectReplacement<'a>, RuntimeError> {
        self.admit_commit_inner(Some(retirement))
    }

    fn admit_commit_inner(
        self,
        retirement: Option<VerifiedJavaProjectRetirement<'a>>,
    ) -> Result<ReadyJavaProjectReplacement<'a>, RuntimeError> {
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
            return Err(invalid_conversion(
                "Java project retirement ticket does not match the prepared owner",
            ));
        }
        let session = owner
            .session
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java session is busy during project publication"))?;
        if session.is_some() || owner.retirement_requested.get() {
            return Err(invalid_conversion(
                "Java project artifacts cannot change after JVM session startup or retirement",
            ));
        }
        let classpath = owner
            .initial_classpath
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java classpath is busy during project publication"))?;
        let configured_classpath = owner.configured_classpath.try_borrow_mut().map_err(|_| {
            invalid_conversion("configured Java paths are busy during project publication")
        })?;
        if owner.configured_classpath_revision.get() != self.configured_classpath_revision
            || classpath.snapshot().identity != self.baseline_identity
        {
            return Err(invalid_conversion(
                "Java classpath changed after project artifact preparation",
            ));
        }
        let identities = owner.artifact_identities.try_borrow_mut().map_err(|_| {
            invalid_conversion("Java artifact identities are busy during project publication")
        })?;
        let discovery = owner
            .discovery
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java discovery is busy during project publication"))?;
        let config = owner.config.try_borrow_mut().map_err(|_| {
            invalid_conversion("Java runtime policy is busy during project publication")
        })?;
        if let Some(installation) = self.installation.as_ref() {
            let policy = self
                .next_configuration
                .as_ref()
                .map_or(&*config, |next| &next.config);
            JvmProcess::validate_existing_compatibility(installation, policy)
                .map_err(java_runtime_error)?;
        }
        let lease = owner.selected_byte_lease.try_borrow_mut().map_err(|_| {
            invalid_conversion("Java private byte lease is busy during project publication")
        })?;
        Ok(ReadyJavaProjectReplacement {
            staged: self,
            _retirement: retirement,
            classpath,
            identities,
            configured_classpath,
            discovery,
            config,
            _session: session,
            lease,
        })
    }
}

impl<'a> ReadyJavaProjectReplacement<'a> {
    /// No await or foreign call may intervene after `admit_commit`.
    pub fn commit(self) -> Option<VerifiedJavaProjectRetirement<'a>> {
        let ReadyJavaProjectReplacement {
            staged,
            _retirement,
            mut classpath,
            mut identities,
            mut configured_classpath,
            mut discovery,
            mut config,
            _session,
            mut lease,
        } = self;
        if let Some(next) = staged.next_lease {
            // Admission proves there is no adapter session; any externally
            // retained classloader owns its byte lease through JavaSession.
            *lease = next;
        }
        *classpath = staged.classpath;
        *identities = staged.identities;
        if let Some(next) = staged.next_configuration {
            *configured_classpath = next.configured_classpath;
            *discovery = next.discovery;
            *config = next.config;
            staged
                .owner
                .configured_classpath_revision
                .set(next.next_revision);
        }
        if _retirement.is_none() {
            staged
                .owner
                .project_retirement
                .publish_fresh(&staged.retirement_identity);
        }
        _retirement
    }
}

impl JavaAdapter {
    /// Project jar identities, excluding the configured and dynamic classpaths.
    pub fn has_project_artifacts(&self) -> Result<bool, RuntimeError> {
        self.artifact_identities
            .try_borrow()
            .map(|identities| !identities.is_empty())
            .map_err(|_| {
                invalid_conversion("Java artifact identities are busy during project preparation")
            })
    }

    pub fn prepare_project_replacement<'a>(
        &'a self,
        artifacts: &[(JavaArtifactIdentity, PathBuf)],
    ) -> Result<PreparedJavaProjectReplacement<'a>, RuntimeError> {
        self.prepare_project_replacement_with_configuration(None, artifacts)
    }

    pub fn prepare_project_replacement_with_configuration<'a>(
        &'a self,
        configuration: Option<JavaRuntimeConfiguration>,
        artifacts: &[(JavaArtifactIdentity, PathBuf)],
    ) -> Result<PreparedJavaProjectReplacement<'a>, RuntimeError> {
        let session = self
            .session
            .try_borrow()
            .map_err(|_| invalid_conversion("Java session is busy during project preparation"))?;
        if self.retirement_requested.get() {
            return Err(invalid_conversion(
                "Java project artifacts cannot be prepared during terminal retirement",
            ));
        }
        drop(session);
        self.handles.admit_host_restart(&self.host_identity)?;
        let current = self
            .initial_classpath
            .try_borrow()
            .map_err(|_| invalid_conversion("Java classpath is busy during project preparation"))?;
        let configured_classpath_revision = self.configured_classpath_revision.get();
        let configured = self.configured_classpath.try_borrow().map_err(|_| {
            invalid_conversion("configured Java paths are busy during project preparation")
        })?;
        let baseline_identity = current.snapshot().identity;
        let next_configuration = configuration
            .map(|configuration| {
                prepare_configuration(configuration, configured_classpath_revision)
            })
            .transpose()?;
        let mut classpath = next_configuration
            .as_ref()
            .map_or_else(|| current.clone(), |next| next.classpath.clone());
        let mut paths = next_configuration.as_ref().map_or_else(
            || configured.iter().cloned().collect::<Vec<_>>(),
            |next| next.configured_classpath.clone(),
        );
        drop(configured);
        drop(current);
        let mut identities = BTreeSet::new();
        for (identity, path) in artifacts {
            identity
                .validate_file(path)
                .map_err(|error| invalid_conversion(error.to_string()))?;
            if !identities.insert(identity.clone()) {
                return Err(invalid_conversion(format!(
                    "Java artifact identity `{identity}` appears more than once"
                )));
            }
            if !paths.contains(path) {
                paths.push(path.clone());
            }
        }
        classpath
            .replace_project(paths)
            .map_err(|error| invalid_conversion(error.to_string()))?;
        let installation = if artifacts.is_empty() && next_configuration.is_none() {
            None
        } else {
            let discovery = next_configuration.as_ref().map_or_else(
                || self.discovery.borrow().clone(),
                |next| next.discovery.clone(),
            );
            let installation = discover_jvm(&discovery).map_err(java_runtime_error)?;
            let config = next_configuration
                .as_ref()
                .map_or_else(|| self.config.borrow().clone(), |next| next.config.clone());
            JvmProcess::validate_existing_compatibility(&installation, &config)
                .map_err(java_runtime_error)?;
            Some(installation)
        };
        Ok(PreparedJavaProjectReplacement {
            owner: self,
            retirement_identity: self.project_retirement.snapshot(),
            classpath,
            identities,
            baseline_identity,
            configured_classpath_revision,
            installation,
            next_configuration,
            next_lease: None,
        })
    }
}
