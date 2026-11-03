//! References: what a feature, region or publish entry writes to point at topology.

use crate::ids::{FeatureId, FilePath, PublishName, RevisionId, SketchEntityId};
use crate::refs::name::{EntityName, GenerationName, InstanceKey, Role, SourceTag, TopoKind};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Live or pinned, for references into another file.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "context")]
pub enum XrefContext {
    /// Follow the other file's working state. Allowed within a partition only.
    Live,
    /// Use the other file as of this released revision. Updating a pin is a reviewable change.
    Pinned { revision: RevisionId },
}

/// The stored part of an explicit geometric fallback: what the entity looked like when it was
/// picked. It can only match entities made by `maker`, and only within `tolerance`; it never
/// picks the nearest of several.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometricFallback {
    pub maker: FeatureId,
    pub signature: GeometricSignature,
    pub tolerance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometricSignature {
    pub kind: TopoKind,
    /// Outward normal for planar faces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normal: Option<[f64; 3]>,
    pub centroid: [f64; 3],
    /// Area for faces, length for edges, zero for vertices.
    pub measure: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "ref")]
pub enum Reference {
    /// A face or body by name (generation name, pattern copy or CAD-file name).
    Name { kind: TopoKind, name: EntityName },
    /// Layer 3: the pieces of `base` that touch every reference in `bounded_by` and none in
    /// `not_bounded_by` (for faces: sharing an edge; for edges: meeting at a vertex).
    Piece {
        base: Box<Reference>,
        #[serde(default)]
        bounded_by: Vec<Reference>,
        #[serde(default)]
        not_bounded_by: Vec<Reference>,
    },
    /// Layer 3: every edge between a face of `a` and a face of `b`.
    EdgesBetween {
        a: Box<Reference>,
        b: Box<Reference>,
    },
    /// Layer 3: every vertex that touches each of these (faces or edges).
    VertexOf { of: Vec<Reference> },
    /// A published name: of this file (`file: None`) or of another file. Cross-file references
    /// can take no other form.
    Published {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file: Option<FilePath>,
        name: PublishName,
        context: XrefContext,
    },
    /// Layer 5: `primary`, falling back to a geometric match only when the resolver's policy
    /// allows it, and always with a diagnostic.
    WithFallback {
        primary: Box<Reference>,
        fallback: GeometricFallback,
    },
}

impl Reference {
    pub fn face(name: EntityName) -> Self {
        Reference::Name {
            kind: TopoKind::Face,
            name,
        }
    }

    pub fn body(name: EntityName) -> Self {
        Reference::Name {
            kind: TopoKind::Body,
            name,
        }
    }

    /// `feature.<id>.face.<role>(<tag>)`.
    pub fn gen_face(feature: &str, role: Role, tag: SourceTag) -> Self {
        Reference::face(EntityName::generated(feature, role, tag))
    }

    /// `feature.<id>.face.side(<curve>)` for a profile curve in `sketch`.
    pub fn side(feature: &str, sketch: &str, curve: &str) -> Self {
        Self::gen_face(
            feature,
            Role::Side,
            SourceTag::Curve {
                sketch: FeatureId::new(sketch),
                entity: SketchEntityId::new(curve),
            },
        )
    }

    pub fn start_cap(feature: &str) -> Self {
        Self::gen_face(feature, Role::StartCap, SourceTag::None)
    }

    pub fn end_cap(feature: &str) -> Self {
        Self::gen_face(feature, Role::EndCap, SourceTag::None)
    }

    /// The body a feature made: `feature.<id>.body`.
    pub fn feature_body(feature: &str) -> Self {
        Reference::body(EntityName::generated(feature, Role::Body, SourceTag::None))
    }

    pub fn edges_between(a: Reference, b: Reference) -> Self {
        Reference::EdgesBetween {
            a: Box::new(a),
            b: Box::new(b),
        }
    }

    pub fn vertex_of(of: Vec<Reference>) -> Self {
        Reference::VertexOf { of }
    }

    pub fn piece(base: Reference, bounded_by: Vec<Reference>) -> Self {
        Reference::Piece {
            base: Box::new(base),
            bounded_by,
            not_bounded_by: Vec::new(),
        }
    }

    pub fn copy_of(base: Reference, pattern: &str, instance: InstanceKey) -> Self {
        match base {
            Reference::Name { kind, name } => Reference::Name {
                kind,
                name: EntityName::Copy {
                    base: Box::new(name),
                    pattern: FeatureId::new(pattern),
                    instance,
                },
            },
            other => other,
        }
    }

    pub fn published(file: Option<&str>, name: &str, context: XrefContext) -> Self {
        Reference::Published {
            file: file.map(FilePath::new),
            name: PublishName::new(name),
            context,
        }
    }

    /// The topology kind the reference yields, where it is fixed by its form.
    pub fn kind(&self) -> Option<TopoKind> {
        match self {
            Reference::Name { kind, .. } => Some(*kind),
            Reference::Piece { base, .. } => base.kind(),
            Reference::EdgesBetween { .. } => Some(TopoKind::Edge),
            Reference::VertexOf { .. } => Some(TopoKind::Vertex),
            Reference::Published { .. } => None,
            Reference::WithFallback { primary, .. } => primary.kind(),
        }
    }

    /// Every feature of this file the reference names.
    pub fn makers(&self) -> Vec<FeatureId> {
        let mut out = Vec::new();
        self.collect_makers(&mut out);
        out.sort();
        out.dedup();
        out
    }

    fn collect_makers(&self, out: &mut Vec<FeatureId>) {
        match self {
            Reference::Name { name, .. } => out.extend(name.makers()),
            Reference::Piece {
                base,
                bounded_by,
                not_bounded_by,
            } => {
                base.collect_makers(out);
                bounded_by
                    .iter()
                    .chain(not_bounded_by)
                    .for_each(|r| r.collect_makers(out));
            }
            Reference::EdgesBetween { a, b } => {
                a.collect_makers(out);
                b.collect_makers(out);
            }
            Reference::VertexOf { of } => of.iter().for_each(|r| r.collect_makers(out)),
            Reference::Published { .. } => {}
            Reference::WithFallback { primary, fallback } => {
                primary.collect_makers(out);
                out.push(fallback.maker.clone());
            }
        }
    }

    /// The same reference with feature `from` renamed to `to` (an alias rewrite).
    pub fn renamed(&self, from: &FeatureId, to: &FeatureId) -> Reference {
        match self {
            Reference::Name { kind, name } => Reference::Name {
                kind: *kind,
                name: name.renamed(from, to),
            },
            Reference::Piece {
                base,
                bounded_by,
                not_bounded_by,
            } => Reference::Piece {
                base: Box::new(base.renamed(from, to)),
                bounded_by: bounded_by.iter().map(|r| r.renamed(from, to)).collect(),
                not_bounded_by: not_bounded_by.iter().map(|r| r.renamed(from, to)).collect(),
            },
            Reference::EdgesBetween { a, b } => Reference::EdgesBetween {
                a: Box::new(a.renamed(from, to)),
                b: Box::new(b.renamed(from, to)),
            },
            Reference::VertexOf { of } => Reference::VertexOf {
                of: of.iter().map(|r| r.renamed(from, to)).collect(),
            },
            Reference::Published { .. } => self.clone(),
            Reference::WithFallback { primary, fallback } => Reference::WithFallback {
                primary: Box::new(primary.renamed(from, to)),
                fallback: GeometricFallback {
                    maker: if &fallback.maker == from {
                        to.clone()
                    } else {
                        fallback.maker.clone()
                    },
                    ..fallback.clone()
                },
            },
        }
    }
}

impl From<GenerationName> for EntityName {
    fn from(g: GenerationName) -> Self {
        EntityName::Generated(g)
    }
}

impl fmt::Display for Reference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reference::Name { kind, name } => match name {
                EntityName::Generated(g) if g.role == Role::Body => {
                    write!(f, "feature.{}.body", g.feature)
                }
                _ => write!(f, "{}:{}", kind, name),
            },
            Reference::Piece {
                base,
                bounded_by,
                not_bounded_by,
            } => {
                write!(f, "piece({}", base)?;
                for r in bounded_by {
                    write!(f, ", bounded_by = {}", r)?;
                }
                for r in not_bounded_by {
                    write!(f, ", not_bounded_by = {}", r)?;
                }
                f.write_str(")")
            }
            Reference::EdgesBetween { a, b } => write!(f, "edges_between({}, {})", a, b),
            Reference::VertexOf { of } => {
                f.write_str("vertex_of(")?;
                for (i, r) in of.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", r)?;
                }
                f.write_str(")")
            }
            Reference::Published {
                file,
                name,
                context,
            } => {
                match file {
                    Some(file) => write!(f, "{:?}.publish.{}", file.as_str(), name)?,
                    None => write!(f, "publish.{}", name)?,
                }
                if let XrefContext::Pinned { revision } = context {
                    write!(f, "@{}", revision)?;
                }
                Ok(())
            }
            Reference::WithFallback { primary, fallback } => {
                write!(
                    f,
                    "{} (geometric fallback within {})",
                    primary, fallback.maker
                )
            }
        }
    }
}
