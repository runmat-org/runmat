//! Host-only orchestration for RunMat native AOT products.

pub mod archive;
pub mod compile;
mod error;
pub mod link;
pub mod installation;
pub mod target_pack;
mod tree;
mod object;
mod selected_program;

pub use error::{AotError, AotResult};
pub use object::{emit_native_object, emit_selected_native_object, NativeObjectOptions};
pub use selected_program::{
    compile_selected_standalone, compile_selected_target_pack_standalone, SelectedLinkRequest,
    SelectedTargetPack,
};
mod artifact;
pub use artifact::{materialize_native_object_artifact, read_native_object_artifact};
