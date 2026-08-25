use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForeignResourceKey {
    pub host_identity: String,
    pub handle: u64,
    pub generation: u64,
}

/// Release sink owned by the runtime foreign registry. The value layer knows
/// only how to notify the owner when the final reference disappears.
pub trait ForeignResourceRelease: std::fmt::Debug + Send + Sync {
    fn release(&self, key: &ForeignResourceKey);
}

#[derive(Debug)]
struct ForeignLease {
    key: ForeignResourceKey,
    release: Arc<dyn ForeignResourceRelease>,
}

impl Drop for ForeignLease {
    fn drop(&mut self) {
        self.release.release(&self.key);
    }
}

/// Opaque live foreign-runtime reference. The host registry owns the actual
/// resource; the generation fences stale handles after host/session restart.
#[derive(Debug, Clone)]
pub struct ForeignRef {
    pub host_identity: String,
    pub handle: u64,
    pub generation: u64,
    pub type_identity: ForeignTypeIdentity,
    pub ownership: ForeignOwnership,
    pub affinity: ForeignAffinity,
    pub lifetime: ForeignLifetime,
    lease: Option<Arc<ForeignLease>>,
}

impl ForeignRef {
    pub fn detached(
        key: ForeignResourceKey,
        type_identity: ForeignTypeIdentity,
        ownership: ForeignOwnership,
        affinity: ForeignAffinity,
        lifetime: ForeignLifetime,
    ) -> Self {
        Self::from_parts(key, type_identity, ownership, affinity, lifetime, None)
    }

    pub fn managed(
        key: ForeignResourceKey,
        type_identity: ForeignTypeIdentity,
        ownership: ForeignOwnership,
        affinity: ForeignAffinity,
        lifetime: ForeignLifetime,
        release: Arc<dyn ForeignResourceRelease>,
    ) -> Self {
        let lease = Arc::new(ForeignLease {
            key: key.clone(),
            release,
        });
        Self::from_parts(
            key,
            type_identity,
            ownership,
            affinity,
            lifetime,
            Some(lease),
        )
    }

    fn from_parts(
        key: ForeignResourceKey,
        type_identity: ForeignTypeIdentity,
        ownership: ForeignOwnership,
        affinity: ForeignAffinity,
        lifetime: ForeignLifetime,
        lease: Option<Arc<ForeignLease>>,
    ) -> Self {
        Self {
            host_identity: key.host_identity,
            handle: key.handle,
            generation: key.generation,
            type_identity,
            ownership,
            affinity,
            lifetime,
            lease,
        }
    }

    pub fn resource_key(&self) -> ForeignResourceKey {
        ForeignResourceKey {
            host_identity: self.host_identity.clone(),
            handle: self.handle,
            generation: self.generation,
        }
    }

    pub fn is_managed(&self) -> bool {
        self.lease.is_some()
    }

    pub fn is_same_resource(&self, other: &Self) -> bool {
        self.host_identity == other.host_identity
            && self.handle == other.handle
            && self.generation == other.generation
    }
}

impl PartialEq for ForeignRef {
    fn eq(&self, other: &Self) -> bool {
        self.is_same_resource(other)
            && self.type_identity == other.type_identity
            && self.ownership == other.ownership
            && self.affinity == other.affinity
            && self.lifetime == other.lifetime
    }
}

impl Eq for ForeignRef {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct NoopRelease;

    impl ForeignResourceRelease for NoopRelease {
        fn release(&self, _key: &ForeignResourceKey) {}
    }

    fn reference(generation: u64) -> ForeignRef {
        ForeignRef::managed(
            ForeignResourceKey {
                host_identity: "host-a".into(),
                handle: 17,
                generation,
            },
            ForeignTypeIdentity {
                family: "java".into(),
                name: "java.lang.StringBuilder".into(),
                version: 1,
            },
            ForeignOwnership::Shared,
            ForeignAffinity::OriginProcess,
            ForeignLifetime::Session,
            Arc::new(NoopRelease),
        )
    }

    #[test]
    fn resource_identity_includes_host_handle_and_generation() {
        let current = reference(3);
        assert!(current.is_same_resource(&current.clone()));
        assert!(!current.is_same_resource(&reference(4)));

        let mut different_host = current.clone();
        different_host.host_identity = "host-b".into();
        assert!(!current.is_same_resource(&different_host));
    }
}
