//! The design section of `runmat.lock`, as a pure data structure: a stable UUID per item or
//! occurrence path, plus the paths it was known by (rename aliases). File I/O comes later.
//!
//! Merge rule: the union of entries; the same path with two UUIDs, or one UUID at two paths, is
//! a conflict.

use crate::ids::{FeatureId, FilePath, ItemUuid, PartName};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use thiserror::Error;

/// An item path: a file plus segments within it (`["feature", "body"]`, `["part", "gasket"]`,
/// or an occurrence path `["occurrence", "wing.left", "box", "rib.r12"]`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ItemPath {
    pub file: FilePath,
    pub item: Vec<String>,
}

impl ItemPath {
    pub fn feature(file: &FilePath, feature: &FeatureId) -> Self {
        Self {
            file: file.clone(),
            item: vec!["feature".into(), feature.0.clone()],
        }
    }
    pub fn part(file: &FilePath, part: &PartName) -> Self {
        Self {
            file: file.clone(),
            item: vec!["part".into(), part.0.clone()],
        }
    }
    pub fn occurrence(file: &FilePath, path: &[&str]) -> Self {
        let mut item = vec!["occurrence".to_owned()];
        item.extend(path.iter().map(|s| s.to_string()));
        Self {
            file: file.clone(),
            item,
        }
    }
}

impl fmt::Display for ItemPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.file, self.item.join("."))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockEntry {
    pub uuid: ItemUuid,
    /// Earlier paths of this item, oldest first is not kept; the set is what resolves.
    #[serde(default)]
    pub aliases: BTreeSet<ItemPath>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockSection {
    /// Keyed by the item's current path, so serialisation is sorted and stable. Serialised as
    /// a list of `{ path, uuid, aliases }` rows (JSON maps need string keys).
    #[serde(with = "entry_rows")]
    pub entries: BTreeMap<ItemPath, LockEntry>,
}

mod entry_rows {
    use super::{ItemPath, LockEntry};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    #[derive(Serialize, Deserialize)]
    struct Row {
        path: ItemPath,
        #[serde(flatten)]
        entry: LockEntry,
    }

    pub fn serialize<S: Serializer>(
        m: &BTreeMap<ItemPath, LockEntry>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let rows: Vec<Row> = m
            .iter()
            .map(|(p, e)| Row {
                path: p.clone(),
                entry: e.clone(),
            })
            .collect();
        rows.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<ItemPath, LockEntry>, D::Error> {
        let rows = Vec::<Row>::deserialize(d)?;
        Ok(rows.into_iter().map(|r| (r.path, r.entry)).collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum LockError {
    #[error("`{path}` has two ids: {a} and {b}")]
    PathConflict {
        path: ItemPath,
        a: ItemUuid,
        b: ItemUuid,
    },
    #[error("id {uuid} is at two paths: `{a}` and `{b}`")]
    UuidConflict {
        uuid: ItemUuid,
        a: ItemPath,
        b: ItemPath,
    },
    #[error("`{path}` is not in the lock")]
    Unknown { path: ItemPath },
    #[error("`{path}` already exists")]
    Exists { path: ItemPath },
}

/// Mints new item ids.
pub trait UuidMinter {
    fn mint(&mut self, path: &ItemPath) -> ItemUuid;
}

/// Deterministic ids from a project seed and the first path an item had. Tests and offline
/// sessions use it; the server may mint random ones. Either way an id never changes once minted.
#[derive(Clone, Debug)]
pub struct SeededMinter {
    pub seed: String,
}

impl UuidMinter for SeededMinter {
    fn mint(&mut self, path: &ItemPath) -> ItemUuid {
        let mut h = Sha256::new();
        h.update(b"runmat-cad/lock/v1\0");
        h.update(self.seed.as_bytes());
        h.update([0]);
        h.update(path.to_string().as_bytes());
        let d: [u8; 32] = h.finalize().into();
        let mut b = [0u8; 16];
        b.copy_from_slice(&d[..16]);
        ItemUuid(u128::from_be_bytes(b))
    }
}

/// The result of looking a path up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockLookup {
    pub uuid: ItemUuid,
    pub current: ItemPath,
    pub via_alias: bool,
}

impl LockSection {
    /// The id for `path`, minting one if the item is new.
    pub fn ensure(&mut self, path: &ItemPath, minter: &mut dyn UuidMinter) -> ItemUuid {
        if let Some(e) = self.entries.get(path) {
            return e.uuid;
        }
        let uuid = minter.mint(path);
        self.entries.insert(
            path.clone(),
            LockEntry {
                uuid,
                aliases: BTreeSet::new(),
            },
        );
        uuid
    }

    pub fn lookup(&self, path: &ItemPath) -> Option<LockLookup> {
        if let Some(e) = self.entries.get(path) {
            return Some(LockLookup {
                uuid: e.uuid,
                current: path.clone(),
                via_alias: false,
            });
        }
        self.entries
            .iter()
            .find(|(_, e)| e.aliases.contains(path))
            .map(|(p, e)| LockLookup {
                uuid: e.uuid,
                current: p.clone(),
                via_alias: true,
            })
    }

    /// Record a rename: the id moves to `to`, and `from` becomes an alias.
    pub fn rename(&mut self, from: &ItemPath, to: &ItemPath) -> Result<ItemUuid, LockError> {
        if self.entries.contains_key(to) {
            return Err(LockError::Exists { path: to.clone() });
        }
        let mut entry = self
            .entries
            .remove(from)
            .ok_or_else(|| LockError::Unknown { path: from.clone() })?;
        entry.aliases.insert(from.clone());
        entry.aliases.remove(to);
        let uuid = entry.uuid;
        self.entries.insert(to.clone(), entry);
        Ok(uuid)
    }

    /// The current feature id for a feature known by an older name in `file`.
    pub fn feature_alias(&self, file: &FilePath, old: &FeatureId) -> Option<FeatureId> {
        let l = self.lookup(&ItemPath::feature(file, old))?;
        if !l.via_alias
            || l.current.file != *file
            || l.current.item.len() != 2
            || l.current.item[0] != "feature"
        {
            return None;
        }
        Some(FeatureId::new(l.current.item[1].clone()))
    }

    pub fn merge(&self, other: &LockSection) -> Result<LockSection, LockError> {
        let mut out = self.clone();
        for (path, e) in &other.entries {
            match out.entries.get_mut(path) {
                Some(mine) if mine.uuid != e.uuid => {
                    return Err(LockError::PathConflict {
                        path: path.clone(),
                        a: mine.uuid,
                        b: e.uuid,
                    })
                }
                Some(mine) => mine.aliases.extend(e.aliases.iter().cloned()),
                None => {
                    out.entries.insert(path.clone(), e.clone());
                }
            }
        }
        let mut by_uuid: BTreeMap<ItemUuid, &ItemPath> = BTreeMap::new();
        for (p, e) in &out.entries {
            if let Some(prev) = by_uuid.insert(e.uuid, p) {
                return Err(LockError::UuidConflict {
                    uuid: e.uuid,
                    a: prev.clone(),
                    b: p.clone(),
                });
            }
        }
        Ok(out)
    }
}
