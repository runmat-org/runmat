use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};

/// Opaque live foreign-runtime reference. The host registry owns the actual
/// resource; the generation fences stale handles after host/session restart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignRef {
    pub host_identity: String,
    pub handle: u64,
    pub generation: u64,
    pub type_identity: ForeignTypeIdentity,
    pub ownership: ForeignOwnership,
    pub affinity: ForeignAffinity,
    pub lifetime: ForeignLifetime,
}

impl ForeignRef {
    pub fn is_same_resource(&self, other: &Self) -> bool {
        self.host_identity == other.host_identity
            && self.handle == other.handle
            && self.generation == other.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(generation: u64) -> ForeignRef {
        ForeignRef {
            host_identity: "host-a".into(),
            handle: 17,
            generation,
            type_identity: ForeignTypeIdentity {
                family: "java".into(),
                name: "java.lang.StringBuilder".into(),
                version: 1,
            },
            ownership: ForeignOwnership::Shared,
            affinity: ForeignAffinity::OriginProcess,
            lifetime: ForeignLifetime::Session,
        }
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
