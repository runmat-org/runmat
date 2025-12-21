//! Symbol table and symbol management
//!
//! Symbols are interned strings with unique IDs for efficient comparison.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Unique identifier for an interned symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SymbolId(pub u32);

/// Symbol attributes that affect simplification behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SymbolAttrs {
    /// Symbol represents a real number
    pub real: bool,
    /// Symbol represents a positive number
    pub positive: bool,
    /// Symbol represents an integer
    pub integer: bool,
    /// Symbol represents a non-negative number
    pub nonnegative: bool,
}

/// A symbolic variable with interned name and optional attributes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol {
    pub id: SymbolId,
    pub name: String,
    pub attrs: SymbolAttrs,
}

impl Symbol {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let id = SYMBOL_TABLE.with_global(|table| table.intern(&name));
        Symbol {
            id,
            name,
            attrs: SymbolAttrs::default(),
        }
    }

    pub fn with_attrs(name: impl Into<String>, attrs: SymbolAttrs) -> Self {
        let name = name.into();
        let id = SYMBOL_TABLE.with_global(|table| table.intern(&name));
        Symbol { id, name, attrs }
    }

    /// Check if this symbol should be treated as real-valued
    pub fn is_real(&self) -> bool {
        self.attrs.real
    }

    /// Check if this symbol should be treated as positive
    pub fn is_positive(&self) -> bool {
        self.attrs.positive
    }

    /// Check if this symbol should be treated as an integer
    pub fn is_integer(&self) -> bool {
        self.attrs.integer
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Global symbol table for interning symbol names
pub struct SymbolTable {
    name_to_id: HashMap<String, SymbolId>,
    id_to_name: Vec<String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            name_to_id: HashMap::new(),
            id_to_name: Vec::new(),
        }
    }

    /// Intern a symbol name, returning its unique ID
    pub fn intern(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = SymbolId(self.id_to_name.len() as u32);
        self.id_to_name.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Look up a symbol name by ID
    pub fn get_name(&self, id: SymbolId) -> Option<&str> {
        self.id_to_name.get(id.0 as usize).map(|s| s.as_str())
    }

    /// Look up a symbol ID by name
    pub fn get_id(&self, name: &str) -> Option<SymbolId> {
        self.name_to_id.get(name).copied()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe global symbol table wrapper
struct GlobalSymbolTable {
    inner: OnceLock<Mutex<SymbolTable>>,
}

impl GlobalSymbolTable {
    const fn new() -> Self {
        GlobalSymbolTable {
            inner: OnceLock::new(),
        }
    }

    fn with_global<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut SymbolTable) -> R,
    {
        let mutex = self.inner.get_or_init(|| Mutex::new(SymbolTable::new()));
        let mut guard = mutex.lock().unwrap();
        f(&mut guard)
    }
}

static SYMBOL_TABLE: GlobalSymbolTable = GlobalSymbolTable::new();

/// Get a symbol name by ID from the global table
#[allow(dead_code)]
pub fn get_symbol_name(id: SymbolId) -> Option<String> {
    SYMBOL_TABLE.with_global(|table| table.get_name(id).map(|s| s.to_string()))
}

/// Intern a symbol in the global table
#[allow(dead_code)]
pub fn intern_symbol(name: &str) -> SymbolId {
    SYMBOL_TABLE.with_global(|table| table.intern(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_interning() {
        let x1 = Symbol::new("x");
        let x2 = Symbol::new("x");
        let y = Symbol::new("y");

        assert_eq!(x1.id, x2.id);
        assert_ne!(x1.id, y.id);
    }

    #[test]
    fn test_symbol_attrs() {
        let x = Symbol::with_attrs(
            "x",
            SymbolAttrs {
                real: true,
                positive: true,
                ..Default::default()
            },
        );
        assert!(x.is_real());
        assert!(x.is_positive());
        assert!(!x.is_integer());
    }
}
