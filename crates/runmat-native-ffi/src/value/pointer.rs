use std::ffi::c_void;
use std::ptr::NonNull;

use crate::{NativeType, PointerOwnership};

/// Opaque address returned by a native library. Dereferencing remains inside
/// the invocation adapter and requires matching normalized type metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePointer {
    address: NonNull<c_void>,
    pub pointee: NativeType,
    pub ownership: PointerOwnership,
}

impl NativePointer {
    /// # Safety
    ///
    /// `address` must remain valid for the declared ownership and pointee type.
    /// The caller must arrange any matching release operation.
    pub unsafe fn from_address(
        address: NonNull<c_void>,
        pointee: NativeType,
        ownership: PointerOwnership,
    ) -> Self {
        Self {
            address,
            pointee,
            ownership,
        }
    }

    pub(crate) fn address(&self) -> NonNull<c_void> {
        self.address
    }
}
