use serde::{Deserialize, Serialize};

/// Source-level function role carried with compiled function identity.
///
/// `Unknown` is the only safe interpretation of an older artifact that did
/// not preserve this distinction; it must not be treated as a local function.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FunctionSourceRole {
    #[default]
    Unknown,
    Primary,
    Local,
    Nested,
    Anonymous,
    Method,
    Synthetic,
}
