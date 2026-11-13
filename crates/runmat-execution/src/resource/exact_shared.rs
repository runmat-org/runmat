//! Known-layout, retained, copy-on-write owner for exact-admitted values.
//! Callers charge this allocation before publication and may retain a claim
//! inside the payload for the allocation's complete physical lifetime.

use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::atomic::{fence, AtomicUsize, Ordering};

struct Inner<T> {
    strong: AtomicUsize,
    value: T,
}

pub struct ExactShared<T> {
    pointer: NonNull<Inner<T>>,
}

impl<T> ExactShared<T> {
    pub const fn requested_bytes() -> usize {
        std::mem::size_of::<Inner<T>>()
    }

    pub fn try_new(value: T) -> Result<Self, T> {
        let allocate = |layout| {
            // SAFETY: the exact layout is released by the final owner.
            unsafe { std::alloc::alloc(layout) }
        };
        // SAFETY: `alloc` returns either null or a fresh pointer with the
        // exact requested global-allocator layout.
        unsafe { Self::try_new_with(value, allocate) }
    }

    /// # Safety
    /// A non-null allocator result must be a fresh, uniquely owned block from
    /// the global allocator with exactly the supplied layout and alignment.
    /// Its ownership transfers to this exact shared owner.
    pub unsafe fn try_new_with(
        value: T,
        allocate: impl FnOnce(std::alloc::Layout) -> *mut u8,
    ) -> Result<Self, T> {
        let layout = std::alloc::Layout::new::<Inner<T>>();
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

    pub fn new(value: T) -> Self {
        Self::try_new(value).unwrap_or_else(|_| {
            std::alloc::handle_alloc_error(std::alloc::Layout::new::<Inner<T>>())
        })
    }

    fn inner(&self) -> &Inner<T> {
        // SAFETY: this handle contributes one strong count, so the allocation
        // remains live throughout the borrow.
        unsafe { self.pointer.as_ref() }
    }

    pub fn get(&self) -> &T {
        &self.inner().value
    }

    pub fn is_unique(&self) -> bool {
        self.inner().strong.load(Ordering::Acquire) == 1
    }

    pub fn get_mut_unique(&mut self) -> &mut T {
        assert!(
            self.is_unique(),
            "exact shared owner must be unique for mutation"
        );
        // SAFETY: &mut self excludes cloning this handle, and strong==1
        // proves no aliasing handle can observe the value.
        unsafe { &mut self.pointer.as_mut().value }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.pointer == other.pointer
    }

    /// Transfer one strong reference into an opaque ownership token without
    /// changing its count or allocating. This supports executor interfaces
    /// that retain a data pointer, such as a raw waker.
    pub fn into_raw(self) -> *const () {
        let owner = ManuallyDrop::new(self);
        owner.pointer.as_ptr().cast()
    }

    /// Recover the strong reference transferred by `into_raw`.
    ///
    /// # Safety
    /// `pointer` must be the unmodified token returned by `ExactShared<T>`
    /// with this exact `T`. Each transferred reference may be recovered only
    /// once, and its allocation must still be live. A temporary borrowed
    /// reconstruction must be wrapped in `ManuallyDrop` so it does not consume
    /// the transferred reference. The token is not a pointer to `T` and must
    /// not be dereferenced or used to infer an external owner's layout.
    pub unsafe fn from_raw(pointer: *const ()) -> Self {
        Self {
            pointer: NonNull::new(pointer.cast_mut().cast::<Inner<T>>())
                .expect("exact shared ownership token must be non-null"),
        }
    }

    /// Move a uniquely owned payload out without cloning its allocation
    /// claims. A shared payload remains owned by the returned handle.
    pub fn try_unwrap(self) -> Result<T, Self> {
        if !self.is_unique() {
            return Err(self);
        }
        let this = ManuallyDrop::new(self);
        // SAFETY: this consumed handle is the sole strong owner. There is
        // no weak-reference API that could create another owner here. The
        // Box releases the exact header while moving its payload intact.
        let boxed = unsafe { Box::from_raw(this.pointer.as_ptr()) };
        Ok(boxed.value)
    }

    pub fn into_inner(self) -> T
    where
        T: Clone,
    {
        match self.try_unwrap() {
            Ok(value) => value,
            Err(shared) => shared.get().clone(),
        }
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
            // SAFETY: this is the final strong owner, so no alias can access
            // the allocation. Move the payload out exactly once and release
            // the header before dropping a claim-bearing payload.
            let value =
                unsafe { std::ptr::read(std::ptr::addr_of!((*self.pointer.as_ptr()).value)) };
            unsafe {
                std::alloc::dealloc(
                    self.pointer.as_ptr().cast::<u8>(),
                    std::alloc::Layout::new::<Inner<T>>(),
                )
            };
            drop(value);
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
    fn denied_allocation_returns_the_original_payload() {
        let payload = vec![3_u64, 5, 8];
        // SAFETY: null never transfers an allocation into the owner.
        let result = unsafe { ExactShared::try_new_with(payload, |_| std::ptr::null_mut()) };
        assert_eq!(result.err(), Some(vec![3, 5, 8]));
    }

    #[test]
    fn denied_owner_releases_a_claim_bearing_payload_on_payload_drop() {
        struct Claim(Arc<AtomicUsize>);
        impl Drop for Claim {
            fn drop(&mut self) {
                self.0.fetch_sub(7, Ordering::SeqCst);
            }
        }
        let live = Arc::new(AtomicUsize::new(7));
        // SAFETY: null never transfers an allocation into the owner.
        let payload =
            unsafe { ExactShared::try_new_with(Claim(live.clone()), |_| std::ptr::null_mut()) }
                .err()
                .expect("failed owner returns the claim-bearing payload");
        assert_eq!(live.load(Ordering::SeqCst), 7);
        drop(payload);
        assert_eq!(live.load(Ordering::SeqCst), 0);

        // The payload deliberately has no Clone implementation: extracting
        // its claim must move it, and a shared extraction must preserve it.
        live.store(7, Ordering::SeqCst);
        let owner = ExactShared::new(Claim(live.clone()));
        let token = owner.into_raw();
        assert_eq!(live.load(Ordering::SeqCst), 7);
        // SAFETY: this is the original token with the same non-Clone Claim
        // type. ManuallyDrop borrows its count; only the later recovery owns
        // that transferred reference.
        let borrowed = ManuallyDrop::new(unsafe { ExactShared::<Claim>::from_raw(token) });
        let alias_from_token = ExactShared::clone(&borrowed);
        // SAFETY: the token's original transferred count is still live and
        // has not been consumed by the borrowed reconstruction.
        let owner = unsafe { ExactShared::<Claim>::from_raw(token) };
        assert!(owner.ptr_eq(&alias_from_token));
        drop(alias_from_token);
        let alias = owner.clone();
        let owner = owner.try_unwrap().err().expect("still shared");
        assert_eq!(live.load(Ordering::SeqCst), 7);
        drop(alias);
        let claim = owner.try_unwrap().ok().expect("sole remaining owner");
        assert_eq!(live.load(Ordering::SeqCst), 7);
        drop(claim);
        assert_eq!(live.load(Ordering::SeqCst), 0);
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
