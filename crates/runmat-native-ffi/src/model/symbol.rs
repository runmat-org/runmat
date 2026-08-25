use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallingConvention {
    C,
    System,
    Stdcall,
    Fastcall,
    Thiscall,
    Vectorcall,
}

impl Default for CallingConvention {
    fn default() -> Self {
        Self::C
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SymbolAlias {
    pub public_name: String,
    pub exported_name: String,
}
