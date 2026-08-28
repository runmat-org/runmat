mod canonical;
mod limits;
mod manifest_codec;
mod reader;
mod writer;

pub use limits::ArchiveLimits;
pub use reader::read_bundle;
pub use writer::write_bundle;

const MAGIC: &[u8] = b"RUNMAT-EXECUTION-BUNDLE\x01";
