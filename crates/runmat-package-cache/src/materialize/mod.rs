mod assemble;
mod mount;
mod promote;
mod state;
mod verify;

pub use assemble::begin;
pub use mount::MountDescriptor;
pub use promote::promote;
pub use state::{MaterializationRecord, MaterializationState};
pub use verify::{mark_corrupt, verify};
