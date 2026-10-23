//! Shared native deployment carriers. These are declarations and identities,
//! not verified-byte, provider-instance, qualification, or execution rights.
mod clock;
mod local_inputs;
mod policy;
mod product;
mod service_issuer;

pub use clock::{
    native_clock_service_abi, validate_native_clock_inputs, NativeClockCpuInputs,
    NativeClockInputs, NativeClockInvocationIdentity, NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
};
pub use local_inputs::{
    validate_native_local_inputs, NativeEnvironmentInputs, NativeFilesystemEntry,
    NativeFilesystemInputs, NativeLocalServiceInputs, NativeRandomInputs,
    NATIVE_LOCAL_SERVICE_INPUTS_SCHEMA_VERSION,
};

pub use policy::{
    native_host_callback_service_abi, native_hosted_dynamic_dispatch_abi, native_local_service_abi,
    native_network_service_abi, native_source_evaluation_service_abi, native_ui_service_abi,
    NativeDeploymentProfile, NativeDeploymentServiceBehavior, NativeDeploymentServiceKind,
};
pub use product::{
    NativeProductMode, NativeSelectedProductBinding, NativeSelectedServiceBinding, TargetPackId,
    NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
};
pub use service_issuer::VerifiedNativeServiceIssuer;
