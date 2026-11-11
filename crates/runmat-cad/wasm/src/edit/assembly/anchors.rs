//! What a mate connector anchors to: a face, edge or vertex of a component, or one of its part's
//! datums (a plane, axis, point, frame or mate connector), published from that part so the mate
//! follows the part's edits.
//!
//! A pick's entity is written into its part file as a `publish` entry (a face, edge or vertex as
//! `entities(kind = …, references = [<its stable name>])`, a datum as `datum(feature.<id>)`), and
//! the connector anchors to that published name. Reordering features or changing dimensions
//! renumbers faces; the stable name still finds the same face, so the mate keeps its geometry.
//! An entity whose display carries no stable name keeps the frame the pick had, frozen in the
//! part's coordinates.
//!
//! [`Publisher`] gathers the entries one command adds and writes each part file once.

use super::model;
use crate::picks::Picks;
use crate::runtime::Files;
use crate::xref_repairs::FileEdit;
use runmat_cad::ids::{FeatureId, PublishName};
use runmat_cad::model::datum::DatumRef;
use runmat_cad::model::items::{PublishEntry, PublishTarget};
use runmat_cad::model::PartDocument;
use runmat_cad::refs::{Reference, TopoKind};
use runmat_cad::syntax::edit::{edit_part, EditFailure, SourceChange};
use runmat_cad::syntax::parse_part;
use runmat_design_syntax::edit::TextEdit;
use runmat_types::Span;
use std::collections::BTreeMap;

/// What one pick anchors its connector to.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Anchor {
    /// The whole component: its origin.
    Origin,
    /// A face, edge or vertex by its reference text in part file `file`.
    Entity {
        file: String,
        kind: TopoKind,
        reference: String,
    },
    /// A datum feature of part file `file`.
    Datum { file: String, feature: String },
    /// An entity with no stable name: the frame the pick had (origin, X, Z), in part coordinates.
    Frozen {
        origin: [f64; 3],
        x: [f64; 3],
        z: [f64; 3],
    },
    /// A connector the assembly declares already.
    Connector(String),
}

/// The publish entries one command adds, by part file.
pub(crate) struct Publisher<'a> {
    files: &'a Files,
    documents: BTreeMap<String, PartDocument>,
    pending: BTreeMap<String, Vec<PublishEntry>>,
}

impl<'a> Publisher<'a> {
    pub(crate) fn new(files: &'a Files) -> Self {
        Self {
            files,
            documents: BTreeMap::new(),
            pending: BTreeMap::new(),
        }
    }

    fn text(&self, file: &str) -> Result<&'a String, EditFailure> {
        self.files
            .get(file)
            .ok_or_else(|| model(format!("{file} isn't open to publish the mate's face from")))
    }

    fn document(&mut self, file: &str) -> Result<&PartDocument, EditFailure> {
        if !self.documents.contains_key(file) {
            let text = self.text(file)?;
            let parsed = parse_part(file, text);
            let document = parsed
                .document
                .ok_or(EditFailure::Invalid(parsed.diagnostics))?;
            self.documents.insert(file.to_owned(), document);
        }
        Ok(&self.documents[file])
    }

    /// The name `file` publishes `target` under: an entry that publishes exactly it already, or
    /// a new one named `<stem><n>` (`<stem>` itself first when `bare`).
    fn publish(
        &mut self,
        file: &str,
        target: PublishTarget,
        stem: &str,
        bare: bool,
    ) -> Result<PublishName, EditFailure> {
        let document = self.document(file)?;
        let written = document.publishes.clone();
        let pending = self.pending.entry(file.to_owned()).or_default();
        if let Some(p) = written
            .iter()
            .chain(pending.iter())
            .find(|p| p.target == target)
        {
            return Ok(p.name.clone());
        }
        let taken = |n: &str| {
            written
                .iter()
                .chain(pending.iter())
                .any(|p| p.name.as_str() == n)
        };
        let name = bare
            .then(|| stem.to_owned())
            .filter(|n| !taken(n))
            .unwrap_or_else(|| {
                (1..)
                    .map(|k| format!("{stem}{k}"))
                    .find(|n| !taken(n))
                    .expect("a free name")
            });
        let name = PublishName::new(name);
        pending.push(PublishEntry {
            name: name.clone(),
            target,
            auto: false,
        });
        Ok(name)
    }

    /// The published name `anchor` reads, publishing it from its part when it isn't yet; `None`
    /// for an anchor that names no publish (origin, frozen, connector).
    pub(super) fn name_of(&mut self, anchor: &Anchor) -> Result<Option<PublishName>, EditFailure> {
        Ok(Some(match anchor {
            Anchor::Entity {
                file,
                kind,
                reference,
            } => {
                let text = self.text(file)?;
                let document = self.document(file)?.clone();
                let value = Picks::new(&document, file, text).reference(reference)?;
                let reference: Reference = serde_json::from_value(value)
                    .map_err(|e| model(format!("the picked entity's name: {e}")))?;
                let stem = match kind {
                    TopoKind::Face => "mate_face",
                    TopoKind::Edge => "mate_edge",
                    TopoKind::Vertex => "mate_vertex",
                    _ => "mate_body",
                };
                self.publish(
                    file,
                    PublishTarget::Entities {
                        kind: *kind,
                        references: vec![reference],
                    },
                    stem,
                    false,
                )?
            }
            Anchor::Datum { file, feature } => {
                let target = PublishTarget::Datum {
                    datum: DatumRef::Feature {
                        feature: FeatureId(feature.clone()),
                    },
                };
                self.publish(file, target, feature, true)?
            }
            Anchor::Origin | Anchor::Frozen { .. } | Anchor::Connector(_) => return Ok(None),
        }))
    }

    /// Writes the gathered entries, one change per part file. A file another step of the same
    /// command changed already (a path mate's path) takes the entries on top of that change.
    pub(crate) fn finish(self, others: &mut Vec<FileEdit>) -> Result<(), EditFailure> {
        for (file, entries) in self.pending {
            if entries.is_empty() {
                continue;
            }
            let add = |document: &mut PartDocument| {
                document.publishes.extend(entries.iter().cloned());
                Ok(())
            };
            match others.iter_mut().find(|(p, _, _)| *p == file) {
                Some((_, old, change)) => {
                    let next = edit_part(&change.source, &file, add)?;
                    *change = SourceChange {
                        edits: vec![TextEdit::replace(
                            Span {
                                start: 0,
                                end: old.len(),
                            },
                            next.source.clone(),
                        )],
                        source: next.source,
                        reprinted: change.reprinted || next.reprinted,
                    };
                }
                None => {
                    let old = self.files.get(&file).cloned().unwrap_or_default();
                    let change = edit_part(&old, &file, add)?;
                    others.push((file, old, change));
                }
            }
        }
        Ok(())
    }
}
