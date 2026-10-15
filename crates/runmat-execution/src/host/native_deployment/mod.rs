//! Shared native deployment carriers. These are declarations and identities,
//! not verified-byte, provider-instance, qualification, or execution rights.
mod policy;
mod product;

pub use policy::{
    NativeDeploymentProfile, NativeDeploymentServiceBehavior, NativeDeploymentServiceKind,
};
pub use product::{
    NativeProductMode, NativeSelectedProductBinding, NativeSelectedServiceBinding, TargetPackId,
    NATIVE_SELECTED_PRODUCT_SCHEMA_VERSION,
};
