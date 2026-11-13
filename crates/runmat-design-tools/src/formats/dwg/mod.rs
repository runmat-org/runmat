//! DWG, clean-room from the Open Design Alliance's public "Open Design Specification for .dwg
//! files" (version 5.4.1). Target: R2018 (`AC1032`), whose file container is the R2004 one
//! (section pages, page and section maps, LZ77 compression) and whose objects use the R2010+
//! bit coding.
//!
//! - [`bits`]: the bit-coded value types (B, BB, 3B, BS, BL, BLL, BD, DD, RC, RS, RL, RD, MC, MS,
//!   H, TU, BE, BT, CMC, OT) for writing and reading.
//! - [`crc`]: the 16-bit object CRC and the 32-bit CRC of R2004+ headers.
//! - [`lz77`]: the R2004 compression format (decompression in full; compression as literal
//!   runs, which every reader decompresses).
//! - [`container`]: the R2004+ file container: the file header and its masked header data,
//!   system pages (page map and section map) and masked data pages, written and read back.
//! - [`stream`]: the R2010+ object frame (data, string and handle streams, CRC).
//! - [`header`], [`sections`]: AcDb:Header and the other data sections.
//! - [`tables`], [`entities`]: the database objects and the entities.
//! - [`write`], [`read`]: a [`Sheet`](crate::formats::sheet::Sheet) to DWG bytes and back.

pub mod bits;
pub mod container;
pub mod crc;
pub mod entities;
pub mod header;
pub mod lz77;
pub mod read;
pub mod sections;
pub mod stream;
pub mod tables;
pub mod write;

pub use read::read;
pub use write::write;
