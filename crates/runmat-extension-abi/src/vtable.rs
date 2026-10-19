use core::ffi::c_void;

use crate::{
    RunMatAbiVersion, RunMatBufferLease, RunMatBufferLeaseHandle, RunMatBufferView,
    RunMatCancellationToken, RunMatExtensionCall, RunMatExtensionCapabilities,
    RunMatExtensionInstance, RunMatExtensionResult, RunMatStatusCode, RunMatUtf8View,
    RunMatValueHandle, RunMatValueKind,
};

pub type RunMatRetainValueFn =
    unsafe extern "C" fn(context: *mut c_void, value: RunMatValueHandle) -> RunMatStatusCode;
pub type RunMatReleaseValueFn =
    unsafe extern "C" fn(context: *mut c_void, value: RunMatValueHandle) -> RunMatStatusCode;
pub type RunMatValueKindFn = unsafe extern "C" fn(
    context: *mut c_void,
    value: RunMatValueHandle,
    kind: *mut RunMatValueKind,
) -> RunMatStatusCode;
pub type RunMatBorrowBufferFn = unsafe extern "C" fn(
    context: *mut c_void,
    value: RunMatValueHandle,
    view: *mut RunMatBufferView,
) -> RunMatStatusCode;
pub type RunMatBorrowBufferLeaseFn = unsafe extern "C" fn(
    context: *mut c_void,
    value: RunMatValueHandle,
    lease: *mut RunMatBufferLease,
) -> RunMatStatusCode;
pub type RunMatReleaseBufferFn =
    unsafe extern "C" fn(context: *mut c_void, lease: RunMatBufferLeaseHandle) -> RunMatStatusCode;
pub type RunMatInvokeCallbackFn = unsafe extern "C" fn(
    context: *mut c_void,
    name: RunMatUtf8View,
    call: *const RunMatExtensionCall,
    result: *mut RunMatExtensionResult,
) -> RunMatStatusCode;
pub type RunMatIsCancelledFn = unsafe extern "C" fn(
    context: *mut c_void,
    cancellation: *const RunMatCancellationToken,
) -> bool;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RunMatHostVTable {
    pub abi_version: RunMatAbiVersion,
    pub struct_size: usize,
    pub capabilities: RunMatExtensionCapabilities,
    pub context: *mut c_void,
    pub retain_value: Option<RunMatRetainValueFn>,
    pub release_value: Option<RunMatReleaseValueFn>,
    pub value_kind: Option<RunMatValueKindFn>,
    pub borrow_buffer: Option<RunMatBorrowBufferFn>,
    pub invoke_callback: Option<RunMatInvokeCallbackFn>,
    pub is_cancelled: Option<RunMatIsCancelledFn>,
    pub borrow_buffer_lease: Option<RunMatBorrowBufferLeaseFn>,
    pub release_buffer: Option<RunMatReleaseBufferFn>,
}

pub type RunMatExtensionInitializeFn = unsafe extern "C" fn(
    host: *const RunMatHostVTable,
    instance: *mut *mut RunMatExtensionInstance,
) -> RunMatStatusCode;
pub type RunMatExtensionInvokeFn = unsafe extern "C" fn(
    instance: *mut RunMatExtensionInstance,
    symbol: RunMatUtf8View,
    call: *const RunMatExtensionCall,
    result: *mut RunMatExtensionResult,
) -> RunMatStatusCode;
pub type RunMatExtensionShutdownFn = unsafe extern "C" fn(instance: *mut RunMatExtensionInstance);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RunMatExtensionVTable {
    pub abi_version: RunMatAbiVersion,
    pub struct_size: usize,
    pub required_host_capabilities: RunMatExtensionCapabilities,
    pub provided_capabilities: RunMatExtensionCapabilities,
    pub initialize: Option<RunMatExtensionInitializeFn>,
    pub invoke: Option<RunMatExtensionInvokeFn>,
    pub shutdown: Option<RunMatExtensionShutdownFn>,
}

impl core::fmt::Debug for RunMatHostVTable {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RunMatHostVTable")
            .field("abi_version", &self.abi_version)
            .field("struct_size", &self.struct_size)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for RunMatExtensionVTable {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RunMatExtensionVTable")
            .field("abi_version", &self.abi_version)
            .field("struct_size", &self.struct_size)
            .field(
                "required_host_capabilities",
                &self.required_host_capabilities,
            )
            .field("provided_capabilities", &self.provided_capabilities)
            .finish_non_exhaustive()
    }
}
