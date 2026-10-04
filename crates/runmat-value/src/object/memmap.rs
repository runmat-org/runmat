use crate::{ForeignRef, NumericDType};

/// Value-owned controls for a live mapped-file receiver. Cloning the object
/// copies these controls while retaining the same opened backing lease.
#[derive(Debug, Clone, PartialEq)]
pub struct MemmapReceiverState {
    pub authority: ForeignRef,
    pub offset: usize,
    pub repeat: Option<usize>,
    pub writable: bool,
    pub fields: Vec<MemmapFieldState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemmapFieldState {
    pub name: Option<String>,
    pub element: MemmapElementType,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemmapElementType {
    Numeric(NumericDType),
    Character,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ForeignResourceKey, ForeignResourceRelease, ObjectInstance, ObjectPersistencePayloadKind,
    };
    use runmat_types::{ForeignAffinity, ForeignLifetime, ForeignOwnership, ForeignTypeIdentity};
    use std::sync::Arc;

    #[derive(Debug)]
    struct NoopRelease;

    impl ForeignResourceRelease for NoopRelease {
        fn release(&self, _: &ForeignResourceKey) {}
    }

    #[test]
    fn cloned_receiver_has_independent_controls_and_shared_sealed_lease() {
        let authority = ForeignRef::managed(
            ForeignResourceKey {
                host_identity: "test.memmap".into(),
                handle: 1,
                generation: 1,
            },
            ForeignTypeIdentity {
                family: "test.memmap".into(),
                name: "memmapfile".into(),
                version: 1,
            },
            ForeignOwnership::Shared,
            ForeignAffinity::OriginProcess,
            ForeignLifetime::Session,
            Arc::new(NoopRelease),
        );
        let original =
            ObjectInstance::new("memmapfile").with_memmap_receiver(MemmapReceiverState {
                authority,
                offset: 2,
                repeat: Some(3),
                writable: false,
                fields: vec![MemmapFieldState {
                    name: None,
                    element: MemmapElementType::Numeric(NumericDType::U16),
                    shape: vec![1, 2],
                }],
            });
        let mut clone = original.clone();
        clone.memmap_receiver_mut().unwrap().writable = true;
        assert!(!original.memmap_receiver().unwrap().writable);
        assert!(clone
            .memmap_receiver()
            .unwrap()
            .authority
            .shares_managed_lease(&original.memmap_receiver().unwrap().authority));
        assert_eq!(
            clone.persistence_payload_kind(),
            ObjectPersistencePayloadKind::MemmapReceiver
        );
    }
}
