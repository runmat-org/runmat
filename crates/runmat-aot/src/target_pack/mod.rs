//! Build a canonical selected pack from actual target-built bytes. Producing
//! inventory never issues a caller capability, numerical qualification, or
//! toolchain execution admission.

mod produce;
mod tree;

pub use produce::{
    build_content_addressed_target_pack, build_hosted_target_pack, ContentAddressedNativeArchive,
    ContentAddressedPackBuildInputs, HostedPackDeclaration, HostedPackNativeInput,
    HostedPackService, ProducedTargetPack,
};
pub use tree::{
    build_hosted_tree, load_physical_hosted_tree, read_physical_native_archive,
    read_physical_tree_component, ContentAddressedMaterialization, HostedTreeEntry,
    ProducedHostedTree,
};
