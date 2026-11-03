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
    /// The file itself (its rename alias).
    pub fn file(file: &FilePath) -> Self {
        Self {
            file: file.clone(),
            item: vec!["file".into()],
        }
    }
    pub fn publish(file: &FilePath, name: &crate::ids::PublishName) -> Self {
        Self {
            file: file.clone(),
            item: vec!["publish".into(), name.0.clone()],
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

/// Keyed by the item's current path, so serialisation is sorted and stable. Serialised as a
/// list of `{ path, uuid, aliases }` rows (JSON maps need string keys). Two derived indexes
/// (alias → current path, uuid → current path) are rebuilt on load and kept in step by every
/// mutation, so lookups stay logarithmic at hundreds of thousands of occurrence paths. They are
/// never serialised: the entries are the one source of truth.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "LockRepr", into = "LockRepr")]
pub struct LockSection {
    entries: BTreeMap<ItemPath, LockEntry>,
    alias_index: BTreeMap<ItemPath, ItemPath>,
    uuid_index: BTreeMap<ItemUuid, ItemPath>,
}

#[derive(Serialize, Deserialize)]
struct LockRepr {
    #[serde(with = "entry_rows")]
    entries: BTreeMap<ItemPath, LockEntry>,
}

impl From<LockRepr> for LockSection {
    fn from(r: LockRepr) -> Self {
        let mut s = LockSection::default();
        for (p, e) in r.entries {
            s.insert_entry(p, e);
        }
        s
    }
}

impl From<LockSection> for LockRepr {
    fn from(s: LockSection) -> Self {
        LockRepr { entries: s.entries }
    }
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
    pub fn entries(&self) -> &BTreeMap<ItemPath, LockEntry> {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert or replace an entry as is (no conflict checks; [`LockSection::merge`] checks).
    pub fn insert_entry(&mut self, path: ItemPath, entry: LockEntry) {
        if let Some(old) = self.entries.remove(&path) {
            self.unindex(&path, &old);
        }
        for a in &entry.aliases {
            self.alias_index.insert(a.clone(), path.clone());
        }
        self.uuid_index.insert(entry.uuid, path.clone());
        self.entries.insert(path, entry);
    }

    fn unindex(&mut self, path: &ItemPath, entry: &LockEntry) {
        for a in &entry.aliases {
            if self.alias_index.get(a) == Some(path) {
                self.alias_index.remove(a);
            }
        }
        if self.uuid_index.get(&entry.uuid) == Some(path) {
            self.uuid_index.remove(&entry.uuid);
        }
    }

    /// The current path of an id.
    pub fn path_of(&self, uuid: ItemUuid) -> Option<&ItemPath> {
        self.uuid_index.get(&uuid)
    }

    /// The id for `path`, minting one if the item is new.
    pub fn ensure(&mut self, path: &ItemPath, minter: &mut dyn UuidMinter) -> ItemUuid {
        if let Some(e) = self.entries.get(path) {
            return e.uuid;
        }
        let uuid = minter.mint(path);
        self.insert_entry(
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
        let current = self.alias_index.get(path)?;
        let e = self.entries.get(current)?;
        Some(LockLookup {
            uuid: e.uuid,
            current: current.clone(),
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
        self.unindex(from, &entry);
        entry.aliases.insert(from.clone());
        entry.aliases.remove(to);
        let uuid = entry.uuid;
        self.insert_entry(to.clone(), entry);
        Ok(uuid)
    }

    /// Move every item of `from` to `to` (a file rename or move), keeping ids and recording
    /// each old path as an alias. The file's own entry is created if it had none.
    pub fn rename_file(
        &mut self,
        from: &FilePath,
        to: &FilePath,
        minter: &mut dyn UuidMinter,
    ) -> Result<usize, LockError> {
        self.ensure(&ItemPath::file(from), minter);
        let moving: Vec<ItemPath> = self
            .entries
            .keys()
            .filter(|p| &p.file == from)
            .cloned()
            .collect();
        for p in &moving {
            let target = ItemPath {
                file: to.clone(),
                item: p.item.clone(),
            };
            self.rename(p, &target)?;
        }
        Ok(moving.len())
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
            match out.entries.get(path) {
                Some(mine) if mine.uuid != e.uuid => {
                    return Err(LockError::PathConflict {
                        path: path.clone(),
                        a: mine.uuid,
                        b: e.uuid,
                    })
                }
                Some(mine) => {
                    let mut merged = mine.clone();
                    merged.aliases.extend(e.aliases.iter().cloned());
                    out.insert_entry(path.clone(), merged);
                }
                None => out.insert_entry(path.clone(), e.clone()),
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
