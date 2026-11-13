use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use runmat_types::{
    ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity,
};
use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease};

mod control;
use control::ControlRecord;
pub use control::{HostPolicyChange, PreparedHostPolicyChanges, PreparedHostReconfiguration};

use super::{foreign_error, ForeignErrorKind, ForeignExecutionPolicy};
use crate::RuntimeError;

pub trait ForeignHostRelease: std::fmt::Debug + Send + Sync {
    fn release(&self, key: &ForeignResourceKey);
}

#[derive(Debug, Clone)]
pub struct ForeignHostRegistration {
    pub identity: String,
    pub adapter: String,
    pub session_identity: String,
    pub capabilities: BTreeSet<ForeignCapability>,
    pub policy: ForeignExecutionPolicy,
    pub release: Arc<dyn ForeignHostRelease>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignResourceMetadata {
    pub type_identity: ForeignTypeIdentity,
    pub ownership: ForeignOwnership,
    pub affinity: ForeignAffinity,
    pub lifetime: ForeignLifetime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignResolvedResource {
    pub key: ForeignResourceKey,
    pub adapter: String,
    pub session_identity: String,
    pub metadata: ForeignResourceMetadata,
    pub policy: ForeignExecutionPolicy,
}

#[derive(Debug, Clone)]
struct ResourceRecord {
    metadata: ForeignResourceMetadata,
}

#[derive(Debug)]
struct HostRecord {
    adapter: String,
    session_identity: String,
    generation: u64,
    control_generation: u64,
    next_resource: u64,
    retired: bool,
    capabilities: BTreeSet<ForeignCapability>,
    policy: ForeignExecutionPolicy,
    origin_thread: ThreadId,
    release: Arc<dyn ForeignHostRelease>,
    resources: BTreeMap<u64, ResourceRecord>,
    controls: BTreeMap<u64, ControlRecord>,
}

#[derive(Debug, Default)]
struct RegistryState {
    hosts: BTreeMap<String, HostRecord>,
}

#[derive(Clone, Debug, Default)]
pub struct ForeignHandleRegistry {
    state: Arc<Mutex<RegistryState>>,
}

#[derive(Debug)]
struct RegistryLeaseRelease {
    state: Weak<Mutex<RegistryState>>,
}

impl ForeignResourceRelease for RegistryLeaseRelease {
    fn release(&self, key: &ForeignResourceKey) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let release = {
            let Ok(mut state) = state.lock() else {
                return;
            };
            let Some(host) = state.hosts.get_mut(&key.host_identity) else {
                return;
            };
            if host.generation != key.generation || host.resources.remove(&key.handle).is_none() {
                return;
            }
            Arc::clone(&host.release)
        };
        release.release(key);
    }
}

impl ForeignHandleRegistry {
    pub fn set_host_policy(
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
        control::host_policy_available(&state, host_identity)?;
        let host = state
            .hosts
            .get_mut(host_identity)
            .expect("validated host remains registered under held lock");
        host.policy = policy;
        Ok(())
    }

    pub fn register_host(
        &self,
        registration: ForeignHostRegistration,
    ) -> Result<u64, RuntimeError> {
        if registration.identity.trim().is_empty()
            || registration.adapter.trim().is_empty()
            || registration.session_identity.trim().is_empty()
        {
            return Err(foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host identity, adapter, and session must be non-empty",
            ));
        }
        let mut state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        if state.hosts.contains_key(&registration.identity) {
            return Err(foreign_error(
                ForeignErrorKind::HostAlreadyRegistered,
                format!(
                    "foreign host {} is already registered",
                    registration.identity
                ),
            ));
        }
        state.hosts.insert(
            registration.identity,
            HostRecord {
                adapter: registration.adapter,
                session_identity: registration.session_identity,
                generation: 1,
                control_generation: 1,
                next_resource: 1,
                retired: false,
                capabilities: registration.capabilities,
                policy: registration.policy,
                origin_thread: std::thread::current().id(),
                release: registration.release,
                resources: BTreeMap::new(),
                controls: BTreeMap::new(),
            },
        );
        Ok(1)
    }

    pub fn register_resource(
        &self,
        host_identity: &str,
        metadata: ForeignResourceMetadata,
    ) -> Result<ForeignRef, RuntimeError> {
        let key = {
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
            host.resources.insert(
                handle,
                ResourceRecord {
                    metadata: metadata.clone(),
                },
            );
            ForeignResourceKey {
                host_identity: host_identity.to_string(),
                handle,
                generation: host.generation,
            }
        };
        Ok(ForeignRef::managed(
            key,
            metadata.type_identity,
            metadata.ownership,
            metadata.affinity,
            metadata.lifetime,
            Arc::new(RegistryLeaseRelease {
                state: Arc::downgrade(&self.state),
            }),
        ))
    }

    pub fn resolve(
        &self,
        reference: &ForeignRef,
        required: ForeignCapability,
    ) -> Result<ForeignResolvedResource, RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
        let host = state.hosts.get(&reference.host_identity).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                format!("foreign host {} is not registered", reference.host_identity),
            )
        })?;
        if host.generation != reference.generation {
            return Err(foreign_error(
                ForeignErrorKind::StaleHandle,
                format!(
                    "foreign handle {} belongs to generation {}, current generation is {}",
                    reference.handle, reference.generation, host.generation
                ),
            ));
        }
        if !host.capabilities.contains(&required) {
            return Err(foreign_error(
                ForeignErrorKind::CapabilityUnavailable,
                format!(
                    "foreign host {} lacks {required:?}",
                    reference.host_identity
                ),
            ));
        }
        let resource = host.resources.get(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                format!("foreign handle {} has been released", reference.handle),
            )
        })?;
        if resource.metadata.type_identity != reference.type_identity
            || resource.metadata.ownership != reference.ownership
            || resource.metadata.affinity != reference.affinity
            || resource.metadata.lifetime != reference.lifetime
        {
            return Err(foreign_error(
                ForeignErrorKind::HandleMetadataMismatch,
                "foreign reference metadata does not match the registered resource",
            ));
        }
        if matches!(reference.affinity, ForeignAffinity::OriginThread)
            && host.origin_thread != std::thread::current().id()
        {
            return Err(foreign_error(
                ForeignErrorKind::AffinityViolation,
                "foreign resource must be used on its originating thread",
            ));
        }
        Ok(ForeignResolvedResource {
            key: reference.resource_key(),
            adapter: host.adapter.clone(),
            session_identity: host.session_identity.clone(),
            metadata: resource.metadata.clone(),
            policy: host.policy,
        })
    }

    pub fn restart_host(&self, host_identity: &str) -> Result<u64, RuntimeError> {
        let (release, keys, generation) = {
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
            let previous_generation = host.generation;
            let next_generation = host.generation.checked_add(1).ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    format!("foreign host {host_identity} exhausted its generation space"),
                )
            })?;
            let next_control_generation =
                host.control_generation.checked_add(1).ok_or_else(|| {
                    foreign_error(
                        ForeignErrorKind::HostUnavailable,
                        "foreign host control generation exhausted",
                    )
                })?;
            let keys = host
                .resources
                .keys()
                .map(|handle| ForeignResourceKey {
                    host_identity: host_identity.to_string(),
                    handle: *handle,
                    generation: previous_generation,
                })
                .collect::<Vec<_>>();
            host.resources.clear();
            host.generation = next_generation;
            host.control_generation = next_control_generation;
            host.controls.clear();
            // Control and object generations can differ after reconfiguration.
            // Never recycle an ID: a previous control's public resource key
            // must not equal a newly issued ordinary object's key.
            (Arc::clone(&host.release), keys, host.generation)
        };
        for key in keys {
            release.release(&key);
        }
        Ok(generation)
    }

    /// Abandon a host whose owning asynchronous call was dropped. This must
    /// revoke authority without an async cleanup step or a fallible restart:
    /// a poisoned lock and exhausted generation must never leave old handles
    /// valid after the child has been destroyed.
    pub(crate) fn abandon_host_sync(&self, host_identity: &str) {
        let (release, keys) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(host) = state.hosts.get_mut(host_identity) else {
                return;
            };
            let keys = host
                .resources
                .keys()
                .map(|handle| ForeignResourceKey {
                    host_identity: host_identity.to_owned(),
                    handle: *handle,
                    generation: host.generation,
                })
                .collect::<Vec<_>>();
            let release = Arc::clone(&host.release);
            match (
                host.generation.checked_add(1),
                host.control_generation.checked_add(1),
            ) {
                (Some(generation), Some(control_generation)) => {
                    host.resources.clear();
                    host.controls.clear();
                    host.generation = generation;
                    host.control_generation = control_generation;
                }
                _ => {
                    // Keep this identity occupied forever. Re-registering it
                    // at generation one could alias an old public resource key.
                    host.resources.clear();
                    host.controls.clear();
                    host.retired = true;
                }
            }
            (release, keys)
        };
        for key in keys {
            release.release(&key);
        }
    }

    pub fn unregister_host(&self, host_identity: &str) -> Result<(), RuntimeError> {
        let host = {
            let mut state = self.state.lock().map_err(|_| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    "foreign host registry lock was poisoned",
                )
            })?;
            if state
                .hosts
                .get(host_identity)
                .is_some_and(|host| host.retired)
            {
                return Err(foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    "retired foreign host identity cannot be recycled",
                ));
            }
            state.hosts.remove(host_identity).ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    format!("foreign host {host_identity} is not registered"),
                )
            })?
        };
        for handle in host.resources.keys() {
            host.release.release(&ForeignResourceKey {
                host_identity: host_identity.to_string(),
                handle: *handle,
                generation: host.generation,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[derive(Debug, Default)]
    struct ReleaseCounter(AtomicUsize);

    impl ForeignHostRelease for ReleaseCounter {
        fn release(&self, _key: &ForeignResourceKey) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl ForeignResourceRelease for ReleaseCounter {
        fn release(&self, key: &ForeignResourceKey) {
            ForeignHostRelease::release(self, key);
        }
    }

    fn metadata() -> ForeignResourceMetadata {
        ForeignResourceMetadata {
            type_identity: ForeignTypeIdentity {
                family: "test".into(),
                name: "test.Widget".into(),
                version: 1,
            },
            ownership: ForeignOwnership::Shared,
            affinity: ForeignAffinity::OriginProcess,
            lifetime: ForeignLifetime::Session,
        }
    }

    fn registry(release: Arc<ReleaseCounter>) -> ForeignHandleRegistry {
        let registry = ForeignHandleRegistry::default();
        registry
            .register_host(ForeignHostRegistration {
                identity: "host-a".into(),
                adapter: "test".into(),
                session_identity: "session-a".into(),
                capabilities: BTreeSet::from([ForeignCapability::Invoke, ForeignCapability::Read]),
                policy: ForeignExecutionPolicy::trusted_in_process(),
                release,
            })
            .unwrap();
        registry
    }

    #[test]
    fn clones_release_the_registered_resource_exactly_once() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        let reference = registry.register_resource("host-a", metadata()).unwrap();
        let clone = reference.clone();
        drop(reference);
        assert_eq!(releases.0.load(Ordering::SeqCst), 0);
        drop(clone);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn restart_fences_existing_handles_and_does_not_double_release() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        let reference = registry.register_resource("host-a", metadata()).unwrap();
        assert!(registry
            .resolve(&reference, ForeignCapability::Invoke)
            .is_ok());
        assert_eq!(registry.restart_host("host-a").unwrap(), 2);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
        let error = registry
            .resolve(&reference, ForeignCapability::Invoke)
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Foreign:StaleHandle"));
        drop(reference);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn abandoned_host_fences_both_kinds_of_reference_without_double_release() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        let resource = registry.register_resource("host-a", metadata()).unwrap();
        let control = registry
            .register_host_control("host-a", metadata())
            .unwrap();
        registry.abandon_host_sync("host-a");
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry
                .resolve(&resource, ForeignCapability::Invoke)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        assert_eq!(
            registry
                .resolve_host_control(&control, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        registry
            .register_host_control("host-a", metadata())
            .unwrap();
        registry.abandon_host_sync("host-a");
        drop(resource);
        drop(control);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn abandoned_exhausted_host_keeps_identity_tombstone() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        {
            let mut state = registry.state.lock().unwrap();
            let host = state.hosts.get_mut("host-a").unwrap();
            host.generation = u64::MAX;
            host.control_generation = u64::MAX;
        }
        let resource = registry.register_resource("host-a", metadata()).unwrap();
        let control = registry
            .register_host_control("host-a", metadata())
            .unwrap();
        registry.abandon_host_sync("host-a");
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry
                .resolve(&resource, ForeignCapability::Invoke)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        assert!(registry
            .resolve_host_control(&control, &metadata())
            .is_err());
        assert!(registry.register_resource("host-a", metadata()).is_err());
        assert!(registry
            .register_host_control("host-a", metadata())
            .is_err());
        assert!(registry.unregister_host("host-a").is_err());
        let registration = ForeignHostRegistration {
            identity: "host-a".into(),
            adapter: "test".into(),
            session_identity: "session-b".into(),
            capabilities: BTreeSet::new(),
            policy: ForeignExecutionPolicy::trusted_in_process(),
            release: releases,
        };
        assert_eq!(
            registry
                .register_host(registration)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HostAlreadyRegistered")
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn abandoned_host_revokes_even_after_registry_mutex_poison() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        let resource = registry.register_resource("host-a", metadata()).unwrap();
        let control = registry
            .register_host_control("host-a", metadata())
            .unwrap();
        let state = Arc::clone(&registry.state);
        let _ = std::thread::spawn(move || {
            let _lock = state.lock().unwrap();
            panic!("poison registry for abandonment test");
        })
        .join();
        registry.abandon_host_sync("host-a");
        let state = registry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let host = state.hosts.get("host-a").unwrap();
        assert!(host.resources.is_empty());
        assert!(host.controls.is_empty());
        drop(state);
        drop(resource);
        drop(control);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn metadata_and_capability_tampering_are_rejected() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(releases);
        let reference = registry.register_resource("host-a", metadata()).unwrap();
        let mut forged_type = reference.type_identity.clone();
        forged_type.name = "test.Other".into();
        let forged = runmat_value::ForeignRef::detached(
            reference.resource_key(),
            forged_type,
            reference.ownership,
            reference.affinity,
            reference.lifetime,
        );
        assert_eq!(
            registry
                .resolve(&forged, ForeignCapability::Invoke)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HandleMetadataMismatch")
        );
        assert_eq!(
            registry
                .resolve(&reference, ForeignCapability::Write)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:CapabilityUnavailable")
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn origin_thread_affinity_is_enforced() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(releases);
        let mut resource_metadata = metadata();
        resource_metadata.affinity = ForeignAffinity::OriginThread;
        let reference = registry
            .register_resource("host-a", resource_metadata)
            .unwrap();
        let error = std::thread::spawn(move || {
            registry
                .resolve(&reference, ForeignCapability::Invoke)
                .unwrap_err()
        })
        .join()
        .unwrap();
        assert_eq!(error.identifier(), Some("RunMat:Foreign:AffinityViolation"));
    }

    #[test]
    fn unregister_releases_resources_and_fences_remaining_references() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        let reference = registry.register_resource("host-a", metadata()).unwrap();
        registry.unregister_host("host-a").unwrap();
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry
                .resolve(&reference, ForeignCapability::Invoke)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HostUnavailable")
        );
        drop(reference);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn host_controls_require_the_exact_issuing_lease_and_registry() {
        let releases = Arc::new(ReleaseCounter::default());
        let first = registry(Arc::clone(&releases));
        let second = registry(Arc::clone(&releases));
        let issued = first.register_host_control("host-a", metadata()).unwrap();
        let other = second.register_host_control("host-a", metadata()).unwrap();
        assert_eq!(issued.resource_key(), other.resource_key());
        first
            .resolve_host_control(&issued.clone(), &metadata())
            .unwrap();
        assert_eq!(
            second
                .resolve_host_control(&issued, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HandleMetadataMismatch")
        );
        let detached = ForeignRef::detached(
            issued.resource_key(),
            issued.type_identity.clone(),
            issued.ownership,
            issued.affinity,
            issued.lifetime,
        );
        assert_eq!(
            first
                .resolve_host_control(&detached, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HandleMetadataMismatch")
        );
        let forged = ForeignRef::managed(
            issued.resource_key(),
            issued.type_identity.clone(),
            issued.ownership,
            issued.affinity,
            issued.lifetime,
            Arc::new(ReleaseCounter::default()),
        );
        assert_eq!(
            first
                .resolve_host_control(&forged, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HandleMetadataMismatch")
        );
        drop(issued);
        drop(other);
        drop(forged);
        assert_eq!(releases.0.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn control_revocation_preserves_real_resource_policy_and_restart_fences_both() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        let old = registry
            .register_host_control("host-a", metadata())
            .unwrap();
        registry
            .set_host_policy("host-a", ForeignExecutionPolicy::trusted_in_process())
            .unwrap();
        registry.revoke_host_controls("host-a").unwrap();
        assert_eq!(
            registry
                .resolve_host_control(&old, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        let fresh = registry
            .register_host_control("host-a", metadata())
            .unwrap();
        let resource = registry.register_resource("host-a", metadata()).unwrap();
        assert_eq!(
            registry
                .set_host_policy("host-a", ForeignExecutionPolicy::trusted_in_process())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        registry.restart_host("host-a").unwrap();
        assert_eq!(
            registry
                .resolve_host_control(&fresh, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
        drop(resource);
        drop(fresh);
        drop(old);
        assert_eq!(releases.0.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn grouped_policy_admission_rejects_late_resource_without_partial_change() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(Arc::clone(&releases));
        registry
            .register_host(ForeignHostRegistration {
                identity: "host-b".into(),
                adapter: "test".into(),
                session_identity: "session-b".into(),
                capabilities: BTreeSet::new(),
                policy: ForeignExecutionPolicy::trusted_in_process(),
                release: releases,
            })
            .unwrap();
        let control = registry
            .register_host_control("host-b", metadata())
            .unwrap();
        let resource = registry.register_resource("host-b", metadata()).unwrap();
        let isolated = ForeignExecutionPolicy {
            trust: crate::foreign::ForeignTrust::Untrusted,
            isolation: crate::foreign::ForeignIsolation::IsolatedProcess,
        };
        let changes = [
            HostPolicyChange::Set {
                host_identity: "host-a",
                policy: isolated,
            },
            HostPolicyChange::Reconfigure {
                host_identity: "host-b",
                policy: isolated,
            },
        ];
        assert_eq!(
            registry
                .prepare_host_policy_changes(&changes)
                .err()
                .unwrap()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        {
            let state = registry.state.lock().unwrap();
            assert_eq!(
                state.hosts.get("host-a").unwrap().policy,
                ForeignExecutionPolicy::trusted_in_process()
            );
            assert_eq!(
                state.hosts.get("host-b").unwrap().policy,
                ForeignExecutionPolicy::trusted_in_process()
            );
        }
        drop(resource);
        registry
            .prepare_host_policy_changes(&changes)
            .unwrap()
            .commit();
        assert_eq!(
            registry
                .state
                .lock()
                .unwrap()
                .hosts
                .get("host-a")
                .unwrap()
                .policy,
            isolated
        );
        assert_eq!(
            registry
                .state
                .lock()
                .unwrap()
                .hosts
                .get("host-b")
                .unwrap()
                .policy,
            isolated
        );
        assert_eq!(
            registry
                .resolve_host_control(&control, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
    }

    #[test]
    fn project_restart_rejects_live_object_then_fences_both_generations() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(releases);
        let control = registry
            .register_host_control("host-a", metadata())
            .unwrap();
        let resource = registry.register_resource("host-a", metadata()).unwrap();
        assert_eq!(
            registry
                .admit_host_restart("host-a")
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        drop(resource);
        registry.admit_host_restart("host-a").unwrap();
        let change = HostPolicyChange::Restart {
            host_identity: "host-a",
            policy: ForeignExecutionPolicy::trusted_in_process(),
        };
        let before = {
            let state = registry.state.lock().unwrap();
            let host = state.hosts.get("host-a").unwrap();
            (host.generation, host.control_generation)
        };
        registry
            .prepare_host_policy_changes(&[change])
            .unwrap()
            .commit();
        let state = registry.state.lock().unwrap();
        let host = state.hosts.get("host-a").unwrap();
        assert_eq!(host.generation, before.0 + 1);
        assert_eq!(host.control_generation, before.1 + 1);
        assert!(host.controls.is_empty());
        drop(state);
        assert_eq!(
            registry
                .resolve_host_control(&control, &metadata())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
    }
}
