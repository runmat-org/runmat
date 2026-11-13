mod canonical;
mod driver;
mod inputs;
mod invoke;
pub(crate) mod native_input;
pub(crate) mod paired;
pub(crate) mod persisted;
mod plan;
pub(crate) mod publication;
mod response;
mod selected;
mod section_gc;
mod toolchain;

pub use crate::tree::{HostedNativeTree, NativeBuildTree, NativeStorageGrant, NativeTreeBlobRef};
pub use canonical::{CanonicalSelectedLinkPlan, CanonicalSelectedLinkReceipt, SelectedLinkOption};
pub use driver::{discover_linker, LinkerDriver, LinkerFamily};
pub use inputs::HostedNativeBuildTrees;
pub use invoke::{link_standalone, LinkRequest, LinkedProgram, VerifiedSelectedImage};
pub use paired::{link_selected_native_pair, LinkedNativePair, PairedLinkRequests};
pub use persisted::{PersistedSelectedImage, PERSISTED_SELECTED_IMAGE_SCHEMA_VERSION};
pub use plan::{
    build_link_plan, LinkPlan, NativeProgramLinkIdentity, NATIVE_LINK_PLAN_SCHEMA_VERSION,
};
pub use selected::link_selected_standalone;
pub use toolchain::{
    ExternalNativeLinkAdmission, NativeHostedDependency, NativeLinkAdapter,
    NativeLinkBuildIdentity, NativeLinkExecutionHost,
};
