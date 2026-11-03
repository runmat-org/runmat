//! Structured entity names (layers 1, 2 and 4 of synthesis §6).
//!
//! A name is a typed record built from what the kernel's history says made an entity: the
//! feature, the role and a source tag taken from the feature's own definition (a sketch curve,
//! a hole key, a split side, the edge a blend replaced). Names are never parsed from text. The
//! `Display` forms (`extrude1.side(l1)`) are for people, messages and agents only.

use crate::ids::{FeatureId, SketchEntityId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopoKind {
    Body,
    Face,
    Edge,
    Vertex,
}

impl fmt::Display for TopoKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TopoKind::Body => "body",
            TopoKind::Face => "face",
            TopoKind::Edge => "edge",
            TopoKind::Vertex => "vertex",
        })
    }
}

/// What a feature made an entity as. The role table is per feature family; two faces of one
/// feature never share a role and tag unless the feature numbers them with an ordinal from its
/// own definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A face swept from a profile curve (extrude, revolve, sweep, loft).
    Side,
    /// The cap at the profile plane (extrude, sweep start, revolve start).
    StartCap,
    /// The far cap.
    EndCap,
    /// A fillet face, tagged by the edge it replaced.
    Blend,
    /// A chamfer face, tagged by the edge it replaced.
    Chamfer,
    /// A shell's inner wall, tagged by the face it offsets.
    ShellInner,
    /// A hole's wall, tagged by the hole key.
    HoleWall,
    /// A blind hole's bottom.
    HoleBottom,
    /// A counterbore's floor or a countersink's cone.
    HoleCounterbore,
    /// A face made where a split tool cut a body, tagged by side.
    SplitFace,
    /// A body a feature made.
    Body,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Side => "side",
            Role::StartCap => "start",
            Role::EndCap => "end",
            Role::Blend => "blend",
            Role::Chamfer => "chamfer",
            Role::ShellInner => "inner",
            Role::HoleWall => "wall",
            Role::HoleBottom => "bottom",
            Role::HoleCounterbore => "counterbore",
            Role::SplitFace => "split_face",
            Role::Body => "body",
        }
    }
}

/// Which side of a split tool (by the tool's normal), from the feature's definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaneSide {
    Negative,
    Positive,
}

/// A pattern instance's semantic key. Never an array index into kernel output: linear and grid
/// keys are the step counts along the layout's directions, table keys are row keys.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "instance")]
pub enum InstanceKey {
    Linear {
        step: u32,
    },
    Grid {
        first: u32,
        second: u32,
    },
    Circular {
        step: u32,
    },
    Row {
        key: String,
    },
    /// The single image of a mirror.
    Mirror,
}

impl fmt::Display for InstanceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstanceKey::Linear { step } => write!(f, "{}", step),
            InstanceKey::Grid { first, second } => write!(f, "{},{}", first, second),
            InstanceKey::Circular { step } => write!(f, "@{}", step),
            InstanceKey::Row { key } => write!(f, "{:?}", key),
            InstanceKey::Mirror => f.write_str("mirror"),
        }
    }
}

/// What, from the feature's definition, distinguishes entities of the same role.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "tag")]
pub enum SourceTag {
    None,
    /// A sketch curve of the profile.
    Curve {
        sketch: FeatureId,
        entity: SketchEntityId,
    },
    /// A semantic key from the feature (a hole position key).
    Key {
        key: String,
    },
    /// A split side.
    Side {
        side: PlaneSide,
    },
    /// The edge a blend or chamfer replaced: the edge between these two faces (sorted).
    Edge {
        between: Box<[EntityName; 2]>,
    },
    /// The face a shell wall offsets.
    Face {
        face: Box<EntityName>,
    },
}

/// Layer 1: a generation name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GenerationName {
    pub feature: FeatureId,
    pub role: Role,
    pub tag: SourceTag,
    /// A tie-break from the feature's own definition (curve parameter order of seam pieces,
    /// distance along a direction). Never the kernel's output order. In a reference, `None`
    /// matches every ordinal: the whole logical entity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u32>,
}

/// A face or body name. Edges and vertices are named relationally (by the faces around them),
/// so they need no stored name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "form")]
pub enum EntityName {
    Generated(GenerationName),
    /// A pattern or mirror image of `base`.
    Copy {
        base: Box<EntityName>,
        pattern: FeatureId,
        instance: InstanceKey,
    },
    /// Layer 4: a name read from an imported CAD file (STEP/AP242), matched to canonical ids.
    CadFile {
        import: FeatureId,
        name: String,
    },
}

impl EntityName {
    pub fn generated(feature: &str, role: Role, tag: SourceTag) -> Self {
        EntityName::Generated(GenerationName {
            feature: FeatureId::new(feature),
            role,
            tag,
            ordinal: None,
        })
    }

    /// The feature that made the entity (the outermost maker).
    pub fn maker(&self) -> &FeatureId {
        match self {
            EntityName::Generated(g) => &g.feature,
            EntityName::Copy { pattern, .. } => pattern,
            EntityName::CadFile { import, .. } => import,
        }
    }

    /// Every feature the name mentions, outermost first.
    pub fn makers(&self) -> Vec<FeatureId> {
        let mut out = Vec::new();
        self.collect_makers(&mut out);
        let mut seen = BTreeSet::new();
        out.retain(|f| seen.insert(f.clone()));
        out
    }

    fn collect_makers(&self, out: &mut Vec<FeatureId>) {
        match self {
            EntityName::Generated(g) => {
                out.push(g.feature.clone());
                g.tag.collect_makers(out);
            }
            EntityName::Copy { base, pattern, .. } => {
                out.push(pattern.clone());
                base.collect_makers(out);
            }
            EntityName::CadFile { import, .. } => out.push(import.clone()),
        }
    }

    /// Whether `self`, used as a query, matches the stored name `candidate`. A query without an
    /// ordinal matches every ordinal of the same feature, role and tag.
    pub fn matches(&self, candidate: &EntityName) -> bool {
        match (self, candidate) {
            (EntityName::Generated(q), EntityName::Generated(c)) => {
                q.feature == c.feature
                    && q.role == c.role
                    && q.tag == c.tag
                    && (q.ordinal.is_none() || q.ordinal == c.ordinal)
            }
            (
                EntityName::Copy {
                    base: qb,
                    pattern: qp,
                    instance: qi,
                },
                EntityName::Copy {
                    base: cb,
                    pattern: cp,
                    instance: ci,
                },
            ) => qp == cp && qi == ci && qb.matches(cb),
            _ => self == candidate,
        }
    }

    /// Replace every occurrence of feature `from` with `to` (rename through an alias).
    pub fn renamed(&self, from: &FeatureId, to: &FeatureId) -> EntityName {
        let swap = |f: &FeatureId| if f == from { to.clone() } else { f.clone() };
        match self {
            EntityName::Generated(g) => EntityName::Generated(GenerationName {
                feature: swap(&g.feature),
                role: g.role,
                tag: g.tag.renamed(from, to),
                ordinal: g.ordinal,
            }),
            EntityName::Copy {
                base,
                pattern,
                instance,
            } => EntityName::Copy {
                base: Box::new(base.renamed(from, to)),
                pattern: swap(pattern),
                instance: instance.clone(),
            },
            EntityName::CadFile { import, name } => EntityName::CadFile {
                import: swap(import),
                name: name.clone(),
            },
        }
    }
}

impl SourceTag {
    fn collect_makers(&self, out: &mut Vec<FeatureId>) {
        match self {
            SourceTag::Curve { sketch, .. } => out.push(sketch.clone()),
            SourceTag::Edge { between } => {
                between[0].collect_makers(out);
                between[1].collect_makers(out);
            }
            SourceTag::Face { face } => face.collect_makers(out),
            SourceTag::None | SourceTag::Key { .. } | SourceTag::Side { .. } => {}
        }
    }

    fn renamed(&self, from: &FeatureId, to: &FeatureId) -> SourceTag {
        match self {
            SourceTag::Curve { sketch, entity } => SourceTag::Curve {
                sketch: if sketch == from {
                    to.clone()
                } else {
                    sketch.clone()
                },
                entity: entity.clone(),
            },
            SourceTag::Edge { between } => SourceTag::Edge {
                between: Box::new([between[0].renamed(from, to), between[1].renamed(from, to)]),
            },
            SourceTag::Face { face } => SourceTag::Face {
                face: Box::new(face.renamed(from, to)),
            },
            other => other.clone(),
        }
    }
}

impl fmt::Display for SourceTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SourceTag::None => Ok(()),
            SourceTag::Curve { entity, .. } => write!(f, "{}", entity),
            SourceTag::Key { key } => write!(f, "{}", key),
            SourceTag::Side { side } => f.write_str(match side {
                PlaneSide::Negative => "negative",
                PlaneSide::Positive => "positive",
            }),
            SourceTag::Edge { between } => {
                write!(f, "edges_between({}, {})", between[0], between[1])
            }
            SourceTag::Face { face } => write!(f, "{}", face),
        }
    }
}

impl fmt::Display for EntityName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityName::Generated(g) => {
                write!(f, "{}.{}", g.feature, g.role.as_str())?;
                if g.tag != SourceTag::None || g.ordinal.is_some() {
                    write!(f, "({}", g.tag)?;
                    if let Some(o) = g.ordinal {
                        if g.tag != SourceTag::None {
                            f.write_str(", ")?;
                        }
                        write!(f, "#{}", o)?;
                    }
                    f.write_str(")")?;
                }
                Ok(())
            }
            EntityName::Copy {
                base,
                pattern,
                instance,
            } => write!(f, "{}[{}].{}", pattern, instance, base),
            EntityName::CadFile { import, name } => write!(f, "{}.name({:?})", import, name),
        }
    }
}
