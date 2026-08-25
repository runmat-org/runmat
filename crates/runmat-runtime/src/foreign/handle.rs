use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use runmat_types::{
    ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity,
};
use runmat_value::{ForeignRef, ForeignResourceKey, ForeignResourceRelease};

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
    next_resource: u64,
    capabilities: BTreeSet<ForeignCapability>,
    policy: ForeignExecutionPolicy,
    origin_thread: ThreadId,
    release: Arc<dyn ForeignHostRelease>,
    resources: BTreeMap<u64, ResourceRecord>,
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
                next_resource: 1,
                capabilities: registration.capabilities,
                policy: registration.policy,
                origin_thread: std::thread::current().id(),
                release: registration.release,
                resources: BTreeMap::new(),
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
            host.generation = host.generation.checked_add(1).ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    format!("foreign host {host_identity} exhausted its generation space"),
                )
            })?;
            host.next_resource = 1;
            (Arc::clone(&host.release), keys, host.generation)
        };
        for key in keys {
            release.release(&key);
        }
        Ok(generation)
    }

    pub fn unregister_host(&self, host_identity: &str) -> Result<(), RuntimeError> {
        let host = self
            .state
            .lock()
            .map_err(|_| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    "foreign host registry lock was poisoned",
                )
            })?
            .hosts
            .remove(host_identity)
            .ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    format!("foreign host {host_identity} is not registered"),
                )
            })?;
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
    fn metadata_and_capability_tampering_are_rejected() {
        let releases = Arc::new(ReleaseCounter::default());
        let registry = registry(releases);
        let reference = registry.register_resource("host-a", metadata()).unwrap();
        let mut forged = reference.clone();
        forged.type_identity.name = "test.Other".into();
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
}
