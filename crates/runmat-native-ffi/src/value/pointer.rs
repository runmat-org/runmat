use std::ffi::c_void;
use std::ptr::NonNull;

use crate::{NativeType, PointerOwnership};

/// Opaque address returned by a native library. Dereferencing remains inside
/// the invocation adapter and requires matching normalized type metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePointer {
    address: Option<NonNull<c_void>>,
    pub pointee: NativeType,
    pub ownership: PointerOwnership,
}

impl NativePointer {
    /// # Safety
    ///
    /// `address` must remain valid for the declared lifetime and pointee type.
    /// The caller must uphold the ownership contract and arrange a matching
    /// release operation when that contract requires one.
    pub unsafe fn from_address(
        address: Option<NonNull<c_void>>,
        pointee: NativeType,
        ownership: PointerOwnership,
    ) -> Self {
        Self {
            address,
            pointee,
            ownership,
        }
    }

    pub fn is_null(&self) -> bool {
        self.address.is_none()
    }

    pub(crate) fn address(&self) -> *mut c_void {
        self.address.map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }
}
