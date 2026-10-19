//! Shared native deployment carriers. These are declarations and identities,
//! not verified-byte, provider-instance, qualification, or execution rights.
mod policy;
mod product;
mod service_issuer;

pub use policy::{
    native_source_evaluation_service_abi, NativeDeploymentProfile, NativeDeploymentServiceBehavior,
    NativeDeploymentServiceKind,
};
pub use product::{
    NativeProductMode, NativeSelectedProductBinding, NativeSelectedServiceBinding, TargetPackId,
    NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
};
pub use service_issuer::VerifiedNativeServiceIssuer;
