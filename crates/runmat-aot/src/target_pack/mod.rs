//! Build a canonical selected pack from actual target-built bytes. Producing
//! inventory never issues a caller capability, numerical qualification, or
//! toolchain execution admission.

mod produce;
mod tree;

pub use produce::{
    build_hosted_target_pack, HostedPackDeclaration, HostedPackNativeInput, HostedPackService,
    ProducedTargetPack,
};
pub use tree::{build_hosted_tree, HostedTreeEntry, ProducedHostedTree};
