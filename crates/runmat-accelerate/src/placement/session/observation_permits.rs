//! In-flight complete-call observations belong to the selection epoch. A
//! profile restore or matching invalidation revokes them before late receipts
//! can repopulate a replaced profile. Weak registration does not retain calls.

use runmat_execution::{PlacementInvalidation, PlacementSignature};
use runmat_runtime::context::RuntimeWholeWorkObservationPermit;
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Weak,
};

#[derive(Default)]
pub(super) struct ObservationPermits {
    active: RefCell<Vec<(PlacementSignature, Weak<Permit>)>>,
}

struct Permit(AtomicBool);

impl RuntimeWholeWorkObservationPermit for Permit {
    fn is_current(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl ObservationPermits {
    pub(super) fn register(
        &self,
        signature: &PlacementSignature,
    ) -> Arc<dyn RuntimeWholeWorkObservationPermit> {
        let permit = Arc::new(Permit(AtomicBool::new(true)));
        let mut active = self.active.borrow_mut();
        active.retain(|(_, weak)| weak.strong_count() != 0);
        active.push((signature.clone(), Arc::downgrade(&permit)));
        permit
    }

    pub(super) fn invalidate(&self, invalidation: &PlacementInvalidation) {
        self.active.borrow_mut().retain(|(signature, weak)| {
            let Some(permit) = weak.upgrade() else {
                return false;
            };
            if invalidation.matches_signature(signature) {
                permit.0.store(false, Ordering::Release);
                false
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, PlacementExecutionAuthorityRevision, PlacementRevision};

    #[test]
    fn completed_selections_leave_no_session_length_weak_index() {
        let digest = Digest::sha256(b"host-service");
        let signature = PlacementSignature {
            region: None,
            operation: "unzip".into(),
            runtime_facts: digest,
            revision: PlacementRevision {
                program: None,
                catalog: digest,
                compiler: digest,
                authority: PlacementExecutionAuthorityRevision::HostService(digest),
                policy: digest,
            },
        };
        let permits = ObservationPermits::default();
        for _ in 0..128 {
            drop(permits.register(&signature));
            assert_eq!(permits.active.borrow().len(), 1);
        }
        permits.invalidate(&PlacementInvalidation::All);
        assert!(permits.active.borrow().is_empty());
    }
}
