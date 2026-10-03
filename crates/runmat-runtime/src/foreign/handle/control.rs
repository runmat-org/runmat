//! Registry-owned host control tickets, separate from foreign object resources.

use std::sync::{Arc, Mutex, Weak};

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

impl ForeignHandleRegistry {
    pub fn admit_host_reconfiguration(&self, host_identity: &str) -> Result<(), RuntimeError> {
        let state = self.state.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host registry lock was poisoned",
            )
        })?;
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
        })?;
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
        let host = state.hosts.get_mut(host_identity).ok_or_else(|| {
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
        let next = host.control_generation.checked_add(1).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "foreign host control generation exhausted",
            )
        })?;
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
