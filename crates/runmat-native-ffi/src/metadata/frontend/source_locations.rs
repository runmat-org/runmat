//! Resolve Clang's physical-file elision in JSON wire order, before Value
//! sorts object keys. Include traces and presumed filenames are not file changes.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};

const RESOLVED_FILE: &str = "__runmat_source_file";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SourceFileId(usize);

pub(super) struct HeaderAst {
    pub root: Value,
    pub sources: SourceFiles,
}

pub(super) struct SourceFiles {
    headers: BTreeSet<PathBuf>,
    files: BTreeMap<String, SourceFileId>,
    declared: BTreeSet<SourceFileId>,
    current: Option<SourceFileId>,
}

impl SourceFiles {
    fn observe(&mut self, filename: &str) {
        let id = if let Some(id) = self.files.get(filename) {
            *id
        } else {
            let id = SourceFileId(self.files.len());
            let path = Path::new(filename);
            let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if self.headers.contains(&canonical) {
                self.declared.insert(id);
            }
            self.files.insert(filename.to_owned(), id);
            id
        };
        self.current = Some(id);
    }

    pub(super) fn owns_declaration(&self, node: &Value) -> bool {
        fn id(location: &Value) -> Option<SourceFileId> {
            // A macro-generated declaration belongs to its expansion site.
            let location = location.get("expansionLoc").unwrap_or(location);
            location
                .get(RESOLVED_FILE)?
                .as_u64()
                .and_then(|id| usize::try_from(id).ok())
                .map(SourceFileId)
        }
        node.get("loc")
            .and_then(id)
            .or_else(|| node.pointer("/range/begin").and_then(id))
            .is_some_and(|id| self.declared.contains(&id))
    }
}

pub(super) fn decode(bytes: &[u8], headers: BTreeSet<PathBuf>) -> serde_json::Result<HeaderAst> {
    let mut sources = SourceFiles {
        headers,
        files: BTreeMap::new(),
        declared: BTreeSet::new(),
        current: None,
    };
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let root = AstValue {
        sources: &mut sources,
        role: Role::Node,
    }
    .deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(HeaderAst { root, sources })
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Role {
    Node,
    Location,
    Range,
    IncludeTrace,
}

impl Role {
    fn child(self, key: &str) -> Self {
        match (self, key) {
            (Self::IncludeTrace, _) | (_, "includedFrom") => Self::IncludeTrace,
            (Self::Location, "spellingLoc" | "expansionLoc") => Self::Location,
            (Self::Range, "begin" | "end") | (_, "loc") => Self::Location,
            (_, "range") => Self::Range,
            _ => Self::Node,
        }
    }
}

struct AstValue<'a> {
    sources: &'a mut SourceFiles,
    role: Role,
}

impl<'de> DeserializeSeed<'de> for AstValue<'_> {
    type Value = Value;
    fn deserialize<D: serde::Deserializer<'de>>(self, decoder: D) -> Result<Value, D::Error> {
        decoder.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for AstValue<'_> {
    type Value = Value;
    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a Clang JSON AST value")
    }
    fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }
    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Value, E> {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite AST JSON number"))
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }
    fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: serde::de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(AstValue {
            sources: &mut *self.sources,
            role: self.role,
        })? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        use serde::de::Error;
        let mut value = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if key == RESOLVED_FILE || value.contains_key(&key) {
                return Err(A::Error::custom("duplicate or reserved Clang AST field"));
            }
            let child = map.next_value_seed(AstValue {
                sources: &mut *self.sources,
                role: self.role.child(&key),
            })?;
            if self.role == Role::Location && key == "file" {
                self.sources.observe(
                    child
                        .as_str()
                        .ok_or_else(|| A::Error::custom("AST source filename is not text"))?,
                );
            }
            value.insert(key, child);
        }
        if self.role == Role::Location && value.contains_key("offset") {
            let id = self
                .sources
                .current
                .ok_or_else(|| A::Error::custom("elided AST source file has no predecessor"))?;
            let id = u64::try_from(id.0).map_err(A::Error::custom)?;
            value.insert(RESOLVED_FILE.into(), Value::Number(id.into()));
        }
        Ok(Value::Object(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_elision_ignores_include_traces_and_presumed_names() {
        let ast = decode(br#"{"inner":[
          {"loc":{"offset":1,"file":"public.h","includedFrom":{"file":"main.h"}}},
          {"loc":{"offset":2},"range":{"begin":{"offset":2},"end":{"offset":5}}},
          {"loc":{"spellingLoc":{"offset":1,"file":"macro.h"},"expansionLoc":{"offset":8,"file":"public.h"}}},
          {"loc":{"offset":1,"file":"other.h","presumedFile":"public.h"}}
        ]}"#, BTreeSet::from([PathBuf::from("public.h")])).unwrap();
        let nodes = ast.root["inner"].as_array().unwrap();
        assert!(ast.sources.owns_declaration(&nodes[0]));
        assert!(ast.sources.owns_declaration(&nodes[1]));
        assert!(ast.sources.owns_declaration(&nodes[2]));
        assert!(!ast.sources.owns_declaration(&nodes[3]));
    }
}
