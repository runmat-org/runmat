use crate::{ForeignRef, NumericDType, Value};

/// Value-owned controls for a live mapped-file receiver. Cloning the object
/// copies these controls while retaining the same opened backing lease.
#[derive(Debug, Clone, PartialEq)]
pub struct MemmapReceiverState {
    pub authority: MemmapReceiverAuthority,
    pub offset: usize,
    pub repeat: Option<usize>,
    pub writable: bool,
    pub fields: Vec<MemmapFieldState>,
    pub format_presentation: MemmapFormatPresentationState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemmapReceiverAuthority {
    Opened(ForeignRef),
    /// A provider without opened-range support may retain only a detached,
    /// read-only snapshot. This variant can never authorize a file effect.
    Detached(Box<Value>),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemmapFormatPresentationState {
    TypeName,
    Cell { columns: usize },
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
                authority: MemmapReceiverAuthority::Opened(authority),
                offset: 2,
                repeat: Some(3),
                writable: false,
                fields: vec![MemmapFieldState {
                    name: None,
                    element: MemmapElementType::Numeric(NumericDType::U16),
                    shape: vec![1, 2],
                }],
                format_presentation: MemmapFormatPresentationState::TypeName,
            });
        let mut clone = original.clone();
        clone.memmap_receiver_mut().unwrap().writable = true;
        assert!(!original.memmap_receiver().unwrap().writable);
        let MemmapReceiverAuthority::Opened(cloned) = &clone.memmap_receiver().unwrap().authority
        else {
            panic!("opened clone")
        };
        let MemmapReceiverAuthority::Opened(issued) =
            &original.memmap_receiver().unwrap().authority
        else {
            panic!("opened original")
        };
        assert!(cloned.shares_managed_lease(issued));
        assert_eq!(
            clone.persistence_payload_kind(),
            ObjectPersistencePayloadKind::MemmapReceiver
        );
    }
}
