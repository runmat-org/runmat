//! Known-layout, retained, copy-on-write owner for exact-admitted values.
//! The exact producer charges this Box request before publication.

use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::atomic::{fence, AtomicUsize, Ordering};

struct Inner<T> {
    strong: AtomicUsize,
    value: T,
}

pub(super) struct ExactShared<T> {
    pointer: NonNull<Inner<T>>,
}

impl<T> ExactShared<T> {
    pub(super) const fn requested_bytes() -> usize {
        std::mem::size_of::<Inner<T>>()
    }

    pub(super) fn try_new(value: T) -> Result<Self, T> {
        Self::try_new_with(value, |layout| unsafe { std::alloc::alloc(layout) })
    }

    fn try_new_with(
        value: T,
        allocate: impl FnOnce(std::alloc::Layout) -> *mut u8,
    ) -> Result<Self, T> {
        let layout = std::alloc::Layout::new::<Inner<T>>();
        // SAFETY: the checked, nonzero layout exactly matches Inner<T>.
        let pointer = allocate(layout).cast::<Inner<T>>();
        if pointer.is_null() {
            return Err(value);
        }
        // SAFETY: this fresh allocation is initialized once and uniquely
        // owned by the returned handle until its final reference drops.
        unsafe {
            pointer.write(Inner {
                strong: AtomicUsize::new(1),
                value,
            })
        };
        Ok(Self {
            pointer: NonNull::new(pointer).expect("non-null exact owner"),
        })
    }

    pub(super) fn new(value: T) -> Self {
        Self::try_new(value).unwrap_or_else(|_| {
            std::alloc::handle_alloc_error(std::alloc::Layout::new::<Inner<T>>())
        })
    }

    fn inner(&self) -> &Inner<T> {
        // SAFETY: this handle contributes one strong count, so the allocation
        // remains live throughout the borrow.
        unsafe { self.pointer.as_ref() }
    }

    pub(super) fn get(&self) -> &T {
        &self.inner().value
    }

    pub(super) fn is_unique(&self) -> bool {
        self.inner().strong.load(Ordering::Acquire) == 1
    }

    pub(super) fn get_mut_unique(&mut self) -> &mut T {
        assert!(
            self.is_unique(),
            "exact shared owner must be unique for mutation"
        );
        // SAFETY: &mut self excludes cloning this handle, and strong==1
        // proves no aliasing handle can observe the value.
        unsafe { &mut self.pointer.as_mut().value }
    }

    pub(super) fn ptr_eq(&self, other: &Self) -> bool {
        self.pointer == other.pointer
    }

    pub(super) fn into_inner(self) -> T
    where
        T: Clone,
    {
        if !self.is_unique() {
            return self.get().clone();
        }
        let this = ManuallyDrop::new(self);
        // SAFETY: strong==1, this handle is consumed, and the original
        // allocation was Box. No Drop runs on `this` after transfer.
        let boxed = unsafe { Box::from_raw(this.pointer.as_ptr()) };
        boxed.value
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::*;

    #[test]
    fn denied_exact_owner_allocation_returns_the_original_payload() {
        let payload = vec![3_u64, 5, 8];
        let result = ExactShared::try_new_with(payload, |_| std::ptr::null_mut());
        assert_eq!(result.err(), Some(vec![3, 5, 8]));
    }
}

impl<T> Clone for ExactShared<T> {
    fn clone(&self) -> Self {
        self.inner()
            .strong
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < isize::MAX as usize).then_some(count + 1)
            })
            .expect("exact owner refcount overflow");
        Self {
            pointer: self.pointer,
        }
    }
}

impl<T> Drop for ExactShared<T> {
    fn drop(&mut self) {
        if self.inner().strong.fetch_sub(1, Ordering::Release) == 1 {
            fence(Ordering::Acquire);
            // SAFETY: this was the final strong owner. No other thread can
            // access the allocation, which was originally allocated by Box.
            unsafe { drop(Box::from_raw(self.pointer.as_ptr())) };
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for ExactShared<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

// SAFETY: shared reads require T: Sync, transfer/drop requires T: Send.
// Mutation is exposed only through a unique &mut handle with strong==1.
unsafe impl<T: Send + Sync> Send for ExactShared<T> {}
unsafe impl<T: Send + Sync> Sync for ExactShared<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{atomic::AtomicUsize, Arc, Barrier};

    #[derive(Clone, Debug)]
    struct Payload(Arc<AtomicUsize>);

    impl Drop for Payload {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn concurrent_clones_release_the_known_layout_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = ExactShared::new(Payload(drops.clone()));
        let start = Arc::new(Barrier::new(9));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let clone = owner.clone();
                let start = start.clone();
                scope.spawn(move || {
                    start.wait();
                    assert!(!clone.is_unique());
                    drop(clone);
                });
            }
            start.wait();
        });
        assert!(owner.is_unique());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn into_inner_moves_unique_value_and_clones_only_if_shared() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = ExactShared::new(Payload(drops.clone()));
        let clone = owner.clone();
        let copied = clone.into_inner();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(copied);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let moved = owner.into_inner();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(moved);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
