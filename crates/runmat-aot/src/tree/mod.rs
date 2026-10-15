mod codec;
mod metadata;
mod resolve;
mod storage;
pub(crate) mod physical;
pub use codec::{NativeBuildTree, NativeTreeBlobRef};
pub(crate) use metadata::{admit_json, canonical_json};
#[cfg(test)]
pub(crate) mod tests;
pub use resolve::HostedNativeTree;
pub(crate) use resolve::{resolve_build_tree, NativeFile, ResolvedNativeTree};
pub use storage::NativeStorageGrant;
