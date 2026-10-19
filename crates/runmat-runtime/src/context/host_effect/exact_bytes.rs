//! Exact-capacity, stage-charged temporary bytes. This is an allocation owner,
//! not a workload peak certificate or a replacement for pre-open admission.

#[cfg(test)]
use super::HostEffectLiveAllocationReservation;
use super::{HostEffectAllocationError, HostEffectLiveAllocationClaim};
use std::alloc::{alloc, dealloc, Layout};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
#[cfg(test)]
use std::sync::Arc;

pub struct ExactHostEffectBytes {
    pointer: NonNull<u8>,
    length: usize,
    capacity: usize,
    _claim: HostEffectLiveAllocationClaim,
}

impl ExactHostEffectBytes {
    #[cfg(test)]
    pub(crate) fn test_buffer(capacity: usize) -> Self {
        #[derive(Debug)]
        struct TestClaim;
        impl runmat_async::ManagedDiagnosticAllocationLease for TestClaim {}
        impl HostEffectLiveAllocationReservation for TestClaim {}
        Self::new(
            capacity,
            HostEffectLiveAllocationClaim::Shared(Arc::new(TestClaim)),
        )
        .expect("test exact byte buffer")
    }

    pub(super) fn new(
        capacity: usize,
        claim: HostEffectLiveAllocationClaim,
    ) -> Result<Self, HostEffectAllocationError> {
        let allocate = |layout| {
            // SAFETY: the checked layout and matching release remain owned by
            // the returned buffer if allocation succeeds.
            unsafe { alloc(layout) }
        };
        // SAFETY: the global allocator supplies a fresh block with precisely
        // the requested layout, or null. This owner releases that same block.
        unsafe { Self::new_with(capacity, claim, allocate) }
    }

    /// # Safety
    /// A non-null allocator result must be a fresh, uniquely owned block from
    /// the global allocator with exactly the supplied layout and alignment.
    /// Ownership transfers to this buffer for `dealloc` on drop.
    unsafe fn new_with(
        capacity: usize,
        claim: HostEffectLiveAllocationClaim,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Self, HostEffectAllocationError> {
        let layout =
            Layout::array::<u8>(capacity).map_err(|_| HostEffectAllocationError::Overflow)?;
        let pointer = if capacity == 0 {
            NonNull::dangling()
        } else {
            NonNull::new(allocate(layout)).ok_or(HostEffectAllocationError::AllocationFailed)?
        };
        Ok(Self {
            pointer,
            length: 0,
            capacity,
            _claim: claim,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn clear(&mut self) {
        self.length = 0;
    }

    pub fn truncate(&mut self, length: usize) -> Result<(), HostEffectAllocationError> {
        if length > self.length {
            return Err(HostEffectAllocationError::InvalidInvocation);
        }
        self.length = length;
        Ok(())
    }

    pub fn append(&mut self, bytes: &[u8]) -> Result<(), HostEffectAllocationError> {
        let next = self
            .length
            .checked_add(bytes.len())
            .ok_or(HostEffectAllocationError::Overflow)?;
        if next > self.capacity {
            return Err(HostEffectAllocationError::ExceedsStageEnvelope);
        }
        if !bytes.is_empty() {
            // SAFETY: `next <= capacity`, the source is a valid slice, and no
            // mutable view into this owner is exposed while appending.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.pointer.as_ptr().add(self.length),
                    bytes.len(),
                );
            }
        }
        self.length = next;
        Ok(())
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `length <= capacity` and only the initialized prefix is
        // exposed. NonNull::dangling is valid for an empty slice.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }

    /// Transfer an exact-capacity UTF-8 allocation to `String` without a
    /// second payload allocation. The caller must retain the returned claim
    /// for at least as long as the string, and release it after the string's
    /// physical allocation. This transfer does not certify the other owners
    /// of a complete diagnostic or invocation.
    pub fn into_utf8_string(
        self,
    ) -> Result<(String, HostEffectLiveAllocationClaim), std::str::Utf8Error> {
        std::str::from_utf8(self.as_bytes())?;
        let owner = ManuallyDrop::new(self);
        // SAFETY: suppressing this owner's Drop transfers the original,
        // global-allocator pointer and exact layout to String. The initialized
        // prefix has been validated as UTF-8 above. A zero-capacity buffer is
        // represented without constructing String from a dangling pointer.
        let string = if owner.capacity == 0 {
            String::new()
        } else {
            unsafe { String::from_raw_parts(owner.pointer.as_ptr(), owner.length, owner.capacity) }
        };
        // SAFETY: ManuallyDrop suppresses field destruction. Read the sole
        // claim owner exactly once and return it alongside the physical string.
        let claim = unsafe { std::ptr::read(&owner._claim) };
        Ok((string, claim))
    }
}

impl Drop for ExactHostEffectBytes {
    fn drop(&mut self) {
        if self.capacity != 0 {
            let layout = Layout::array::<u8>(self.capacity)
                .expect("constructor validated exact byte layout");
            // SAFETY: this is the original pointer and exact allocation layout.
            unsafe { dealloc(self.pointer.as_ptr(), layout) };
        }
        // `_claim` drops after the physical allocation has been released.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::{
        HostEffectLiveAllocationKind, HostEffectStageReservation, ManagedHostEffectLedger,
    };

    #[test]
    fn exact_buffer_capacity_and_drop_follow_one_live_stage_claim() {
        let ledger = ManagedHostEffectLedger::new(8);
        let stage = ledger.reserve_stage(8).expect("eight-byte scan envelope");
        let claim = stage
            .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 8)
            .expect("exact claim");
        let mut buffer = ExactHostEffectBytes::new(8, HostEffectLiveAllocationClaim::Shared(claim))
            .expect("exact allocation");
        assert_eq!(buffer.capacity(), 8);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 8);
        buffer.append(b"ab").expect("first append");
        buffer.append(b"cdefgh").expect("remaining capacity");
        assert_eq!(buffer.as_bytes(), b"abcdefgh");
        assert_eq!(
            buffer.append(b"x"),
            Err(HostEffectAllocationError::ExceedsStageEnvelope)
        );
        assert_eq!(buffer.as_bytes(), b"abcdefgh");
        buffer.truncate(3).expect("valid truncation");
        assert_eq!(buffer.as_bytes(), b"abc");
        buffer.clear();
        assert!(buffer.is_empty());
        drop(buffer);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        stage
            .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 8)
            .expect("released bytes can be claimed again");
    }

    #[test]
    fn empty_and_overflow_layouts_have_no_physical_payload() {
        #[derive(Debug)]
        struct Claim;
        impl runmat_async::ManagedDiagnosticAllocationLease for Claim {}
        impl HostEffectLiveAllocationReservation for Claim {}
        let mut empty =
            ExactHostEffectBytes::new(0, HostEffectLiveAllocationClaim::Shared(Arc::new(Claim)))
                .expect("empty owner");
        empty.append(b"").expect("empty append");
        assert_eq!(empty.as_bytes(), b"");
        assert_eq!(
            empty.append(b"x"),
            Err(HostEffectAllocationError::ExceedsStageEnvelope)
        );
        assert!(matches!(
            ExactHostEffectBytes::new(
                usize::MAX,
                HostEffectLiveAllocationClaim::Shared(Arc::new(Claim))
            ),
            Err(HostEffectAllocationError::Overflow)
        ));
    }

    #[test]
    fn rejected_physical_allocation_releases_its_stage_claim() {
        let ledger = ManagedHostEffectLedger::new(4);
        let stage = ledger.reserve_stage(4).expect("stage");
        let claim = stage
            .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 4)
            .expect("claim");
        assert!(matches!(
            unsafe {
                ExactHostEffectBytes::new_with(
                    4,
                    HostEffectLiveAllocationClaim::Shared(claim),
                    |_| std::ptr::null_mut(),
                )
            },
            Err(HostEffectAllocationError::AllocationFailed)
        ));
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 4);
    }

    #[test]
    fn exact_utf8_transfer_retains_claim_until_after_physical_string_drop() {
        let ledger = ManagedHostEffectLedger::new(6);
        let stage = ledger.reserve_stage(6).expect("stage");
        let claim = stage
            .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 6)
            .expect("claim");
        let mut buffer = ExactHostEffectBytes::new(6, HostEffectLiveAllocationClaim::Shared(claim))
            .expect("exact bytes");
        buffer.append("héllo".as_bytes()).expect("six UTF-8 bytes");
        let (string, claim) = buffer.into_utf8_string().expect("valid UTF-8");
        assert_eq!(string, "héllo");
        assert_eq!(string.capacity(), 6);
        let error = runmat_async::runtime_error(string)
            .with_retained_diagnostic_allocation(claim.into_diagnostic_retention())
            .build();
        assert_eq!(error.message(), "héllo");
        assert_eq!(ledger.snapshot().live_claimed_bytes, 6);
        drop(stage);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 6);
        drop(error);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn invalid_utf8_transfer_drops_payload_and_refunds_claim() {
        let ledger = ManagedHostEffectLedger::new(1);
        let stage = ledger.reserve_stage(1).expect("stage");
        let claim = stage
            .reserve_live_exact(HostEffectLiveAllocationKind::CodecWorkspace, 1)
            .expect("claim");
        let mut buffer = ExactHostEffectBytes::new(1, HostEffectLiveAllocationClaim::Shared(claim))
            .expect("exact bytes");
        buffer.append(&[0xff]).expect("one invalid byte");
        assert!(buffer.into_utf8_string().is_err());
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn zero_capacity_utf8_transfer_uses_empty_string_without_allocator_pointer() {
        #[derive(Debug)]
        struct Claim;
        impl runmat_async::ManagedDiagnosticAllocationLease for Claim {}
        impl HostEffectLiveAllocationReservation for Claim {}
        let buffer =
            ExactHostEffectBytes::new(0, HostEffectLiveAllocationClaim::Shared(Arc::new(Claim)))
                .expect("zero-capacity owner");
        let (string, claim) = buffer.into_utf8_string().expect("empty UTF-8");
        assert!(string.is_empty());
        assert_eq!(string.capacity(), 0);
        drop(string);
        drop(claim);
    }
}
