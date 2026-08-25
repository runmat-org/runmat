use serde::{Deserialize, Serialize};

use super::SymbolPrototype;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeLibrary {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub symbols: Vec<SymbolPrototype>,
}
