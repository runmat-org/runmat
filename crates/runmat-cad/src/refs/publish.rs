//! Publish and consume: the only legal form of cross-file reference.
//!
//! Decision 11 ("free by gesture, strict in storage"): picking geometry in another file is
//! allowed, and becomes two model edits: a `publish` entry added to that file (unless one
//! already exposes the same thing) and a [`Reference::Published`] here. The reference is live
//! within a partition and pinned to the latest released revision across partitions. Files the
//! owner marked closed refuse auto-publish.

use crate::ids::{FilePath, PartitionName, PublishName, RevisionId};
use crate::model::datum::DatumValue;
use crate::model::items::{PublishEntry, PublishTarget};
use crate::model::PartDocument;
use crate::refs::lock::{ItemPath, LockError, LockSection, UuidMinter};
use crate::refs::name::{EntityName, SourceTag, TopoKind};
use crate::refs::reference::{Reference, XrefContext};
use crate::value::Quantity;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// A published entity as another file sees it: its kind and names in the publishing file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForeignEntity {
    pub kind: TopoKind,
    pub names: Vec<EntityName>,
}

/// What a published name resolves to in the publishing file, at a revision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForeignTarget {
    pub file: FilePath,
    /// `None`: the live working state.
    pub revision: Option<RevisionId>,
    pub name: PublishName,
    #[serde(default)]
    pub entities: Vec<ForeignEntity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datum: Option<DatumValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Quantity>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "xref")]
pub enum XrefProblem {
    #[error("no project catalog is available to resolve cross-file references")]
    NoCatalog,
    #[error("`{file}` is not in the project")]
    UnknownFile { file: FilePath },
    #[error("`{file}` publishes no `{name}`")]
    NotPublished { file: FilePath, name: PublishName },
    #[error("`{file}` has no revision `{revision}`")]
    RevisionMissing {
        file: FilePath,
        revision: RevisionId,
    },
    #[error(
        "`{file}` is in another partition; references to it must be pinned to a released revision"
    )]
    LiveAcrossPartition {
        file: FilePath,
        latest_release: Option<RevisionId>,
    },
    #[error("`{name}` in `{file}` no longer resolves there")]
    PublisherBroken { file: FilePath, name: PublishName },
}

/// The project's view of other files' published names, by revision.
pub trait PublishedCatalog {
    /// Whether the project has this file.
    fn knows(&self, file: &FilePath) -> bool;
    fn partition(&self, file: &FilePath) -> Option<PartitionName>;
    fn latest_release(&self, file: &FilePath) -> Option<RevisionId>;
    /// Resolve `name` in `file`, live (`revision: None`) or at a revision.
    fn resolve_published(
        &self,
        file: &FilePath,
        name: &PublishName,
        revision: Option<&RevisionId>,
    ) -> Result<ForeignTarget, XrefProblem>;
}

/// A file's resolved publish table.
pub type PublishTable = BTreeMap<PublishName, Result<ForeignTarget, XrefProblem>>;

#[derive(Clone, Debug, Default)]
pub struct CatalogFile {
    pub partition: Option<PartitionName>,
    pub live: PublishTable,
    /// Released revisions, oldest first.
    pub releases: Vec<(RevisionId, PublishTable)>,
}

/// An in-memory catalog: the server index's role, for tests and local sessions.
#[derive(Clone, Debug, Default)]
pub struct MemoryCatalog {
    pub files: BTreeMap<FilePath, CatalogFile>,
}

impl MemoryCatalog {
    pub fn set_partition(&mut self, file: &FilePath, partition: &str) {
        self.files.entry(file.clone()).or_default().partition = Some(PartitionName::new(partition));
    }
    pub fn set_live(&mut self, file: &FilePath, table: PublishTable) {
        self.files.entry(file.clone()).or_default().live = table;
    }
    pub fn release(&mut self, file: &FilePath, revision: &str, table: PublishTable) {
        let rev = RevisionId::new(revision);
        let table = table
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.map(|t| ForeignTarget {
                        revision: Some(rev.clone()),
                        ..t
                    }),
                )
            })
            .collect();
        self.files
            .entry(file.clone())
            .or_default()
            .releases
            .push((rev, table));
    }
}

impl PublishedCatalog for MemoryCatalog {
    fn knows(&self, file: &FilePath) -> bool {
        self.files.contains_key(file)
    }

    fn partition(&self, file: &FilePath) -> Option<PartitionName> {
        self.files.get(file).and_then(|f| f.partition.clone())
    }

    fn latest_release(&self, file: &FilePath) -> Option<RevisionId> {
        self.files
            .get(file)
            .and_then(|f| f.releases.last().map(|(r, _)| r.clone()))
    }

    fn resolve_published(
        &self,
        file: &FilePath,
        name: &PublishName,
        revision: Option<&RevisionId>,
    ) -> Result<ForeignTarget, XrefProblem> {
        let f = self
            .files
            .get(file)
            .ok_or_else(|| XrefProblem::UnknownFile { file: file.clone() })?;
        let table = match revision {
            None => &f.live,
            Some(r) => {
                &f.releases
                    .iter()
                    .find(|(id, _)| id == r)
                    .ok_or_else(|| XrefProblem::RevisionMissing {
                        file: file.clone(),
                        revision: r.clone(),
                    })?
                    .1
            }
        };
        match table.get(name) {
            None => Err(XrefProblem::NotPublished {
                file: file.clone(),
                name: name.clone(),
            }),
            Some(r) => r.clone(),
        }
    }
}

/// A pick made in another file: the canonical reference in that file's terms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForeignPick {
    pub file: FilePath,
    pub kind: TopoKind,
    pub reference: Reference,
}

/// The two edits an auto-publish produces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoPublishEdit {
    pub publishing_file: FilePath,
    /// The entry to add there; `None` when an existing entry already publishes the pick.
    pub add: Option<PublishEntry>,
    /// The reference to write in the consuming file.
    pub reference: Reference,
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum AutoPublishError {
    #[error("`{file}` is closed: its owner turned off auto-publish")]
    Closed { file: FilePath },
    #[error("the pick is in the consuming file itself; reference it directly")]
    SameFile,
    #[error("`{file}` is in another partition and has no released revision to pin to")]
    NoRelease { file: FilePath },
    #[error("the picked document is `{found}`, not `{expected}`")]
    WrongDocument { expected: FilePath, found: FilePath },
}

/// Turn a pick in `publisher` into the publish entry to add there and the reference to use in
/// `consumer`.
pub fn auto_publish(
    publisher: &PartDocument,
    pick: &ForeignPick,
    consumer: &FilePath,
    catalog: &dyn PublishedCatalog,
) -> Result<AutoPublishEdit, AutoPublishError> {
    if publisher.path != pick.file {
        return Err(AutoPublishError::WrongDocument {
            expected: pick.file.clone(),
            found: publisher.path.clone(),
        });
    }
    if &pick.file == consumer {
        return Err(AutoPublishError::SameFile);
    }
    let context = if catalog.partition(&pick.file) == catalog.partition(consumer) {
        XrefContext::Live
    } else {
        match catalog.latest_release(&pick.file) {
            Some(revision) => XrefContext::Pinned { revision },
            None => {
                return Err(AutoPublishError::NoRelease {
                    file: pick.file.clone(),
                })
            }
        }
    };
    // Reuse an entry that already publishes exactly this.
    for p in &publisher.publishes {
        if let PublishTarget::Entities { kind, references } = &p.target {
            if *kind == pick.kind && references.len() == 1 && references[0] == pick.reference {
                return Ok(AutoPublishEdit {
                    publishing_file: pick.file.clone(),
                    add: None,
                    reference: Reference::Published {
                        file: Some(pick.file.clone()),
                        name: p.name.clone(),
                        context,
                    },
                });
            }
        }
    }
    if publisher.closed {
        return Err(AutoPublishError::Closed {
            file: pick.file.clone(),
        });
    }
    let base = suggested_name(&pick.reference);
    let mut name = base.clone();
    let mut n = 2;
    while publisher.publishes.iter().any(|p| p.name.0 == name) {
        name = format!("{}_{}", base, n);
        n += 1;
    }
    let name = PublishName::new(name);
    Ok(AutoPublishEdit {
        publishing_file: pick.file.clone(),
        add: Some(PublishEntry {
            name: name.clone(),
            target: PublishTarget::Entities {
                kind: pick.kind,
                references: vec![pick.reference.clone()],
            },
            auto: true,
        }),
        reference: Reference::Published {
            file: Some(pick.file.clone()),
            name,
            context,
        },
    })
}

/// Apply the publishing side of an auto-publish edit.
pub fn apply_publish(publisher: &mut PartDocument, edit: &AutoPublishEdit) {
    if let Some(entry) = &edit.add {
        if publisher.publish(&entry.name).is_none() {
            publisher.publishes.push(entry.clone());
        }
    }
}

/// Apply an auto-publish edit and give the new entry its stable id in the lock, so later
/// renames of it resolve for every consumer through the alias.
pub fn apply_publish_locked(
    publisher: &mut PartDocument,
    edit: &AutoPublishEdit,
    lock: &mut LockSection,
    minter: &mut dyn UuidMinter,
) {
    apply_publish(publisher, edit);
    if let Reference::Published { name, .. } = &edit.reference {
        lock.ensure(&ItemPath::publish(&publisher.path, name), minter);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum PublishRenameError {
    #[error("`{file}` publishes no `{name}`")]
    Unknown { file: FilePath, name: PublishName },
    #[error("`{file}` already publishes `{name}`")]
    Exists { file: FilePath, name: PublishName },
    #[error(transparent)]
    Lock(#[from] LockError),
}

/// Rename a publish entry (part of the file's API): the entry is renamed in place and the
/// lock records the old name as an alias, so consumers keep resolving and are told to update
/// (`cad.xref.via_alias` with a rewrite repair).
pub fn rename_publish(
    publisher: &mut PartDocument,
    from: &PublishName,
    to: &PublishName,
    lock: &mut LockSection,
    minter: &mut dyn UuidMinter,
) -> Result<(), PublishRenameError> {
    if publisher.publish(to).is_some() {
        return Err(PublishRenameError::Exists {
            file: publisher.path.clone(),
            name: to.clone(),
        });
    }
    let Some(entry) = publisher.publishes.iter_mut().find(|p| &p.name == from) else {
        return Err(PublishRenameError::Unknown {
            file: publisher.path.clone(),
            name: from.clone(),
        });
    };
    entry.name = to.clone();
    let old = ItemPath::publish(&publisher.path, from);
    lock.ensure(&old, minter);
    lock.rename(&old, &ItemPath::publish(&publisher.path, to))?;
    Ok(())
}

fn suggested_name(r: &Reference) -> String {
    fn name_part(n: &EntityName) -> String {
        match n {
            EntityName::Generated(g) => {
                let mut s = format!("{}_{}", g.feature, g.role.as_str());
                match &g.tag {
                    SourceTag::None => {}
                    SourceTag::Curve { entity, .. } => s.push_str(&format!("_{}", entity)),
                    SourceTag::Key { key } => s.push_str(&format!("_{}", key)),
                    SourceTag::Side { side } => s.push_str(match side {
                        crate::refs::name::PlaneSide::Negative => "_negative",
                        crate::refs::name::PlaneSide::Positive => "_positive",
                    }),
                    SourceTag::Edge { .. } | SourceTag::Face { .. } => s.push_str("_of"),
                }
                s
            }
            EntityName::Copy {
                base,
                pattern,
                instance,
                ..
            } => format!("{}_{}_{}", pattern, instance, base.short()),
            EntityName::CadFile { name, .. } => name.clone(),
        }
    }
    let raw = match r {
        Reference::Name { name, .. } => name_part(name),
        Reference::Piece { base, .. } => format!("{}_piece", suggested_name(base)),
        Reference::EdgesBetween { a, b } => {
            format!("{}_{}_edge", suggested_name(a), suggested_name(b))
        }
        Reference::VertexOf { .. } => "vertex".to_owned(),
        Reference::Published { name, .. } => name.0.clone(),
        Reference::WithFallback { primary, .. } => suggested_name(primary),
    };
    let mut out: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    out.trim_matches('_').to_owned()
}
