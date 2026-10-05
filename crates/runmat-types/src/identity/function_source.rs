use serde::{Deserialize, Serialize};

/// Source-level function role carried with compiled function identity.
///
/// `Unknown` is the only safe interpretation of an older artifact that did
/// not preserve this distinction; it must not be treated as a local function.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum FunctionSourceRole {
    #[default]
    Unknown = 0,
    Primary = 1,
    Local = 2,
    Nested = 3,
    Anonymous = 4,
    Method = 5,
    Synthetic = 6,
}
