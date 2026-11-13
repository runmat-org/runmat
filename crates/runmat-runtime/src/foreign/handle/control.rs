//! Registry-owned host control tickets, separate from foreign object resources.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use runmat_types::ForeignAffinity;
use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease, WeakForeignRef};

use super::{ForeignHandleRegistry, ForeignResourceMetadata, RegistryState};
use crate::foreign::{foreign_error, ForeignErrorKind, ForeignExecutionPolicy};
use crate::RuntimeError;

#[derive(Debug, Clone)]
pub(super) struct ControlRecord {
    pub(super) metadata: ForeignResourceMetadata,
    pub(super) lease: WeakForeignRef,
}

/// Holds the exact host record locked between final admission and publication.
pub struct PreparedHostReconfiguration<'a> {
    state: MutexGuard<'a, RegistryState>,
    host_identity: &'a str,
    next_generation: u64,
}

#[derive(Clone, Copy)]
pub enum HostPolicyChange<'a> {
    Set {
        host_identity: &'a str,
        policy: ForeignExecutionPolicy,
    },
    Reconfigure {
        host_identity: &'a str,
        policy: ForeignExecutionPolicy,
    },
    /// A verified interpreter/classloader replacement also fences object IDs.
    Restart {
        host_identity: &'a str,
        policy: ForeignExecutionPolicy,
    },
}

impl HostPolicyChange<'_> {
    fn host_identity(&self) -> &str {
        match self {
            Self::Set { host_identity, .. }
            | Self::Reconfigure { host_identity, .. }
            | Self::Restart { host_identity, .. } => host_identity,
        }
    }
}

/// One registry lock admits and commits all policy changes at a publication gate.
pub struct PreparedHostPolicyChanges<'a> {
    state: MutexGuard<'a, RegistryState>,
    changes: Vec<AdmittedHostPolicyChange<'a>>,
}

enum AdmittedHostPolicyChange<'a> {
    Set {
        host_identity: &'a str,
        policy: ForeignExecutionPolicy,
    },
    Reconfigure {
        host_identity: &'a str,
        policy: ForeignExecutionPolicy,
        next_control_generation: u64,
    },
    Restart {
        host_identity: &'a str,
        policy: ForeignExecutionPolicy,
        next_control_generation: u64,
        next_object_generation: u64,
    },
}

impl AdmittedHostPolicyChange<'_> {
    fn host_identity(&self) -> &str {
        match self {
            Self::Set { host_identity, .. }
            | Self::Reconfigure { host_identity, .. }
            | Self::Restart { host_identity, .. } => host_identity,
        }
    }
}

impl PreparedHostPolicyChanges<'_> {
    pub fn contains_reconfiguration(
        &self,
        host_identity: &str,
        policy: ForeignExecutionPolicy,
    ) -> bool {
        self.changes.iter().any(|change| {
            matches!(
                change,
                AdmittedHostPolicyChange::Reconfigure {
                    host_identity: candidate,
                    policy: candidate_policy,
                    ..
                } if *candidate == host_identity && *candidate_policy == policy
            )
        })
    }

    pub fn contains_restart(&self, host_identity: &str, policy: ForeignExecutionPolicy) -> bool {
        self.changes.iter().any(|change| {
            matches!(
                change,
                AdmittedHostPolicyChange::Restart {
                    host_identity: candidate,
                    policy: candidate_policy,
                    ..
                } if *candidate == host_identity && *candidate_policy == policy
            )
        })
    }

    pub fn commit(mut self) {
        for change in self.changes {
            let host = self
                .state
                .hosts
                .get_mut(change.host_identity())
                .expect("admitted host remains registered under held registry lock");
            match change {
                AdmittedHostPolicyChange::Set { policy, .. } => host.policy = policy,
                AdmittedHostPolicyChange::Reconfigure {
                    policy,
                    next_control_generation,
                    ..
                } => {
                    host.policy = policy;
                    host.controls.clear();
                    host.control_generation = next_control_generation;
                }
                AdmittedHostPolicyChange::Restart {
                    policy,
                    next_control_generation,
                    next_object_generation,
                    ..
                } => {
                    host.policy = policy;
                    host.controls.clear();
                    host.control_generation = next_control_generation;
                    host.generation = next_object_generation;
                }
            }
        }
    }
}

impl PreparedHostReconfiguration<'_> {
    pub fn commit(mut self, policy: ForeignExecutionPolicy) {
        let host = self
            .state
            .hosts
            .get_mut(self.host_identity)
            .expect("prepared host remains registered under its held lock");
        host.policy = policy;
        host.controls.clear();
        host.control_generation = self.next_generation;
    }
}

#[derive(Debug)]
struct ControlLeaseRelease {
    state: Weak<Mutex<RegistryState>>,
}

impl ForeignResourceRelease for ControlLeaseRelease {
    fn release(&self, key: &ForeignResourceKey) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let Ok(mut state) = state.lock() else {
            return;
        };
        if let Some(host) = state.hosts.get_mut(&key.host_identity) {
            if host.control_generation == key.generation {
                host.controls.remove(&key.handle);
            }
        }
    }
}

fn reconfiguration_next(state: &RegistryState, host_identity: &str) -> Result<u64, RuntimeError> {
    let host = state.hosts.get(host_identity).ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign host is not registered",
        )
    })?;
    if !host.resources.is_empty() {
        return Err(foreign_error(
            ForeignErrorKind::InvalidCall,
            "foreign host isolation cannot change while resources are live",
        ));
    }
    host.control_generation.checked_add(1).ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign host control generation exhausted",
        )
    })
}

fn restart_next(state: &RegistryState, host_identity: &str) -> Result<(u64, u64), RuntimeError> {
    let control = reconfiguration_next(state, host_identity)?;
    let host = state
        .hosts
        .get(host_identity)
        .expect("reconfiguration checked host");
    if host.retired {
        return Err(foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign host was permanently retired",
        ));
    }
    let object = host.generation.checked_add(1).ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "foreign host object generation exhausted",
        )
    })?;
    Ok((control, object))
}

pub(super) fn host_policy_available(
    state: &RegistryState,
    host_identity: &str,
) -> Result<(), RuntimeError> {
    let host = state.hosts.get(host_identity).ok_or_else(|| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            format!("foreign host {host_identity} is not registered"),
        )
    })?;
    if !host.resources.is_empty() {
        return Err(foreign_error(
            ForeignErrorKind::InvalidCall,
            "foreign host isolation cannot change while resources are live",
        ));
    }
    Ok(())
}

impl ForeignHandleRegistry {
    pub fn prepare_host_policy_changes<'a>(
        &'a self,
        changes: &[HostPolicyChange<'a>],
    ) -> Result<PreparedHostPolicyChanges<'a>, RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let mut seen = BTreeSet::new();
        let mut prepared = Vec::with_capacity(changes.len());
        for change in changes {
            if !seen.insert(change.host_identity()) {
                return Err(foreign_error(
                    ForeignErrorKind::InvalidCall,
                    "foreign host policy change is duplicated",
                ));
            }
            let prepared_change = match *change {
                HostPolicyChange::Set {
                    host_identity,
                    policy,
                } => {
                    host_policy_available(&state, host_identity)?;
                    AdmittedHostPolicyChange::Set {
                        host_identity,
                        policy,
                    }
                }
                HostPolicyChange::Reconfigure {
                    host_identity,
                    policy,
                } => AdmittedHostPolicyChange::Reconfigure {
                    host_identity,
                    policy,
                    next_control_generation: reconfiguration_next(&state, host_identity)?,
                },
                HostPolicyChange::Restart {
                    host_identity,
                    policy,
                } => {
                    let (control, object) = restart_next(&state, host_identity)?;
                    AdmittedHostPolicyChange::Restart {
                        host_identity,
                        policy,
                        next_control_generation: control,
                        next_object_generation: object,
                    }
                }
            };
            prepared.push(prepared_change);
        }
        Ok(PreparedHostPolicyChanges {
            state,
            changes: prepared,
        })
    }

    /// Side-effect-free admission before retiring the old host. The final
    /// grouped publication guard repeats this check under the registry lock.
    pub fn admit_host_restart(&self, host_identity: &str) -> Result<(), RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        restart_next(&state, host_identity)?;
        Ok(())
    }

    pub fn prepare_host_reconfiguration<'a>(
        &'a self,
        host_identity: &'a str,
    ) -> Result<PreparedHostReconfiguration<'a>, RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let next_generation = reconfiguration_next(&state, host_identity)?;
        Ok(PreparedHostReconfiguration {
            state,
            host_identity,
            next_generation,
        })
    }

    pub fn admit_host_reconfiguration(&self, host_identity: &str) -> Result<(), RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        reconfiguration_next(&state, host_identity)?;
        Ok(())
    }

    /// Commit a pre-start policy change and invalidate old environment
    /// controls atomically. Ordinary foreign objects still forbid the change.
    pub fn reconfigure_host(
        &self,
        host_identity: &str,
        policy: ForeignExecutionPolicy,
    ) -> Result<(), RuntimeError> {
        let mut state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let next = reconfiguration_next(&state, host_identity)?;
        let host = state
            .hosts
            .get_mut(host_identity)
            .expect("validated host remains present under its held lock");
        host.policy = policy;
        host.controls.clear();
        host.control_generation = next;
        Ok(())
    }

    /// A control ticket is not a foreign object: dropping it never sends the
    /// host's object-release callback, and it cannot block pre-start policy.
    pub fn register_host_control(
        &self,
        host_identity: &str,
        metadata: ForeignResourceMetadata,
    ) -> Result<ForeignRef, RuntimeError> {
        let mut state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let host = state.hosts.get_mut(host_identity).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                format!("foreign host {host_identity} is not registered"),
            )
        })?;
        if host.retired {
            return Err(foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host identity has been retired",
            ));
        }
        let handle = host.next_resource;
        host.next_resource = host.next_resource.checked_add(1).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                format!("foreign host {host_identity} exhausted its handle space"),
            )
        })?;
        let reference = ForeignRef::managed(
            ForeignResourceKey {
                host_identity: host_identity.to_owned(),
                handle,
                generation: host.control_generation,
            },
            metadata.type_identity.clone(),
            metadata.ownership,
            metadata.affinity,
            metadata.lifetime,
            Arc::new(ControlLeaseRelease {
                state: Arc::downgrade(&self.state),
            }),
        );
        host.controls.insert(
            handle,
            ControlRecord {
                metadata,
                lease: reference
                    .downgrade()
                    .expect("managed host control has a lease"),
            },
        );
        Ok(reference)
    }

    pub fn resolve_host_control(
        &self,
        reference: &ForeignRef,
        expected: &ForeignResourceMetadata,
    ) -> Result<(), RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let host = state.hosts.get(&reference.host_identity).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host is not registered",
            )
        })?;
        if reference.generation != host.control_generation {
            return Err(foreign_error(
                ForeignErrorKind::StaleHandle,
                "foreign host control belongs to a previous session",
            ));
        }
        let control = host.controls.get(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "foreign host control is not live",
            )
        })?;
        let same_lease = control
            .lease
            .upgrade()
            .is_some_and(|issued| issued.shares_managed_lease(reference));
        if !same_lease
            || control.metadata != *expected
            || reference.type_identity != expected.type_identity
            || reference.ownership != expected.ownership
            || reference.affinity != expected.affinity
            || reference.lifetime != expected.lifetime
        {
            return Err(foreign_error(
                ForeignErrorKind::HandleMetadataMismatch,
                "foreign host control does not match its issuing registry",
            ));
        }
        if matches!(reference.affinity, ForeignAffinity::OriginThread)
            && host.origin_thread != std::thread::current().id()
        {
            return Err(foreign_error(
                ForeignErrorKind::AffinityViolation,
                "foreign host control must be used on its originating thread",
            ));
        }
        Ok(())
    }

    pub fn revoke_host_controls(&self, host_identity: &str) -> Result<(), RuntimeError> {
        let mut state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let host = state.hosts.get_mut(host_identity).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host is not registered",
            )
        })?;
        if !host.resources.is_empty() {
            return Err(foreign_error(
                ForeignErrorKind::InvalidCall,
                "foreign host configuration cannot change while resources are live",
            ));
        }
        let next = host.control_generation.checked_add(1).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host control generation exhausted",
            )
        })?;
        host.controls.clear();
        host.control_generation = next;
        Ok(())
    }
}
