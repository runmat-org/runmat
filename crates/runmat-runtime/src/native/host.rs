use std::ffi::c_void;

use super::{
    NativeCall, NativeExecuteSiteFn, NativeExit, NativeSafepoint, NativeSourceMapEntry,
    NativeValueRef, NATIVE_ABI_VERSION,
};

/// Uniform generated-function entrypoint.
///
/// The caller owns `exit`; writing through an output pointer avoids platform-
/// specific aggregate return conventions. `NativeHostStatus::OK` means `exit`
/// was initialized and must then pass `NativeCall::validate_exit`. Panics,
/// exceptions, and platform unwinding must never cross this boundary; semantic
/// control transfer is represented exclusively by `NativeExit`.
pub type NativeEntryPoint =
    unsafe extern "C" fn(call: *mut NativeCall, exit: *mut NativeExit) -> NativeHostStatus;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeHostStatus(pub u32);

impl NativeHostStatus {
    pub const OK: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const STALE_VALUE: Self = Self(2);
    pub const HOST_FAILURE: Self = Self(3);

    pub const fn is_known(self) -> bool {
        self.0 <= Self::HOST_FAILURE.0
    }
}

pub type NativeRetainValueFn =
    unsafe extern "C" fn(context: *mut c_void, value: NativeValueRef) -> NativeHostStatus;
pub type NativeReleaseValueFn =
    unsafe extern "C" fn(context: *mut c_void, value: NativeValueRef) -> NativeHostStatus;
pub type NativeSlowCallFn = unsafe extern "C" fn(
    context: *mut c_void,
    call: *mut NativeCall,
    exit: *mut NativeExit,
) -> NativeHostStatus;
pub type NativeSafepointFn = unsafe extern "C" fn(
    context: *mut c_void,
    safepoint: *const NativeSafepoint,
    exit: *mut NativeExit,
) -> NativeHostStatus;
pub type NativeSourceLookupFn = unsafe extern "C" fn(
    context: *mut c_void,
    source: u32,
    output: *mut NativeSourceMapEntry,
) -> NativeHostStatus;

/// Runtime helper table supplied to generated code.
///
/// This schema revision requires the exact table size and every callback.
/// Later append-only revisions must negotiate a new ABI version rather than
/// silently interpreting a truncated prefix.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NativeHostVTable {
    pub abi_version: u32,
    pub struct_size: u32,
    pub context: *mut c_void,
    pub retain_value: Option<NativeRetainValueFn>,
    pub release_value: Option<NativeReleaseValueFn>,
    pub slow_call: Option<NativeSlowCallFn>,
    pub poll_safepoint: Option<NativeSafepointFn>,
    pub source_lookup: Option<NativeSourceLookupFn>,
    pub execute_site: Option<NativeExecuteSiteFn>,
}

impl NativeHostVTable {
    pub fn validate(&self) -> Result<(), super::NativeAbiError> {
        if self.abi_version != NATIVE_ABI_VERSION.encoded() {
            return Err(super::NativeAbiError::new(
                "native.host.abi_version",
                "unsupported native ABI version",
            ));
        }
        if self.struct_size as usize != std::mem::size_of::<Self>() {
            return Err(super::NativeAbiError::new(
                "native.host.struct_size",
                "host table size does not match this ABI revision",
            ));
        }
        if self.retain_value.is_none()
            || self.release_value.is_none()
            || self.slow_call.is_none()
            || self.poll_safepoint.is_none()
            || self.source_lookup.is_none()
            || self.execute_site.is_none()
        {
            return Err(super::NativeAbiError::new(
                "native.host.callbacks",
                "all native host callbacks are required",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn retain(_: *mut c_void, _: NativeValueRef) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn release(_: *mut c_void, _: NativeValueRef) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn slow(
        _: *mut c_void,
        _: *mut NativeCall,
        _: *mut NativeExit,
    ) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn safepoint(
        _: *mut c_void,
        _: *const NativeSafepoint,
        _: *mut NativeExit,
    ) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn source(
        _: *mut c_void,
        _: u32,
        _: *mut NativeSourceMapEntry,
    ) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    unsafe extern "C" fn site(
        _: *mut c_void,
        _: *mut NativeCall,
        _: *const crate::native::NativeSiteRequest,
        _: *mut crate::native::NativeSiteOutcome,
        _: *mut NativeExit,
    ) -> NativeHostStatus {
        NativeHostStatus::OK
    }

    fn table() -> NativeHostVTable {
        NativeHostVTable {
            abi_version: NATIVE_ABI_VERSION.encoded(),
            struct_size: std::mem::size_of::<NativeHostVTable>() as u32,
            context: std::ptr::null_mut(),
            retain_value: Some(retain),
            release_value: Some(release),
            slow_call: Some(slow),
            poll_safepoint: Some(safepoint),
            source_lookup: Some(source),
            execute_site: Some(site),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn host_table_requires_version_size_and_every_service() {
        assert_eq!(table().validate(), Ok(()));
        assert!(NativeHostStatus::OK.is_known());
        assert!(!NativeHostStatus(u32::MAX).is_known());

        let mut missing = table();
        missing.slow_call = None;
        assert_eq!(
            missing.validate().unwrap_err().field,
            "native.host.callbacks"
        );

        let mut future = table();
        future.abi_version += 1;
        assert_eq!(
            future.validate().unwrap_err().field,
            "native.host.abi_version"
        );
    }
}
