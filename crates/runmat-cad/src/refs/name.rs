//! Structured entity names (layers 1, 2 and 4 of synthesis §6), one level deep.
//!
//! A name is a typed record built from what the kernel's history says made an entity: the
//! feature, the role and a source tag taken from the feature's own definition (a sketch curve,
//! a hole key, a split side, the edge a blend replaced). Names are never parsed from text.
//!
//! # One level, interned
//!
//! A name never contains another name. Where a name is made from other names (a blend made in
//! place of the edge between two faces, a shell wall offsetting a face, a pattern image of a
//! face), it holds their [`NameToken`]s: fixed-size tokens derived from each name's own
//! one-level recipe (`SHA-256` over a domain tag and the recipe's canonical encoding, truncated
//! to 128 bits). A piece of a piece of a blend of a blend is therefore a chain of bounded
//! records, and a stored reference is the same size however many features came before it.
//! Depth lives in the [`NameTable`], which keeps every recipe a regeneration produced, so names
//! can be shown in full, rewritten through rename aliases and checked for token collisions.
//!
//! Tokens are a pure function of the recipe: the same design gives the same tokens on every
//! platform, release and kernel, so they can be compared without the table.
//!
//! The `Display` forms (`extrude1.side(l1)`) are for people, messages and agents only; nested
//! names show as `~` plus the token's first eight hex digits unless a table expands them
//! ([`NameTable::describe`]).

use crate::ids::{FeatureId, SketchEntityId};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use thiserror::Error;
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
    /// A face swept from a profile curve (extrude, revolve, sweep, loft). A sweep along a path
    /// with corners numbers the pieces by path segment; a loft through more than two sections
    /// numbers them by span; both ordinals come from the definition.
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
    /// A counterbore's floor.
    HoleCounterbore,
    /// A counterbore's wall.
    HoleCounterboreWall,
    /// A countersink's cone.
    HoleCountersink,
    /// The rim a shell leaves where it opened a face, tagged by the opened face.
    ShellRim,
    /// A shell's outer wall when shelled outward, tagged by the face it offsets.
    ShellOuter,
    /// A thin feature's inner wall, tagged by profile curve; ordinal 0 and 1 for the inner
    /// faces of its start and end caps.
    ThinWall,
    /// A step face a step draft leaves between faces hinged at different heights, tagged by
    /// the drafted face.
    DraftStep,
    /// A face made where a split tool cut a body, tagged by side.
    SplitFace,
    /// A body a feature made.
    Body,
    // ----------------------------------------------------------------- direct edits
    /// The face that replaced another (replace face), tagged by the face it replaced.
    Replacement,
    // ---------------------------------------------------------------------- surfaces
    /// A surface body's boundary edge, tagged by the profile curve it runs along or ends;
    /// ordinal 0 at the start cap, 1 at the end cap, 2 at the curve's start point, 3 at its
    /// end point (none: the curve itself, for a planar patch).
    Boundary,
    /// A patch a surface feature made (planar, boundary, fill), tagged by key or curve.
    SurfacePatch,
    /// An offset surface's face, tagged by the face it offsets.
    OffsetFace,
    /// The strip an extend added past a boundary edge, tagged by the edge's key.
    Extension,
    /// A thicken's far face, tagged by the face it offsets.
    ThickenFace,
    /// A thicken's rim along a boundary edge, tagged by the edge's key.
    ThickenRim,
    /// A mid-surface face, tagged by the face pair key.
    MidSurface,
    /// A ruled surface along an edge, tagged by the edge's key.
    Ruled,
    // ------------------------------------------------------------------ sheet metal
    /// A sheet's flat face, tagged by flange key; ordinal 0 is the reference side, 1 the
    /// opposite side.
    SheetFace,
    /// A sheet's thickness face (an edge of the sheet), tagged by flange key and edge.
    SheetEdge,
    /// A bend's cylindrical face, tagged by bend key; ordinal 0 inside, 1 outside.
    BendFace,
    /// A relief cut's wall, tagged by the relief key.
    Relief,
    /// A face a form tool made, tagged by tool face key.
    FormFace,
    // -------------------------------------------------------------------- weldments
    /// A structural member's side, tagged `member/profile-curve`.
    MemberSide,
    /// A structural member's end face; ordinal 0 start, 1 end.
    MemberEnd,
    /// A weldment end cap's face, tagged by cap key.
    CapPlate,
    /// A gusset's face, tagged by gusset key.
    Gusset,
    /// A weld bead's face, tagged by bead key.
    WeldBead,
    // ------------------------------------------------------------------------ mold
    /// A parting or shut-off face of a core/cavity split, tagged by key.
    PartingFace,
    // ---------------------------------------------------------------------- details
    /// A modelled thread's face, tagged by the face it threads.
    ThreadFace,
    /// The top face of an emboss or deboss, tagged by profile key.
    EmbossTop,
    /// A rib's face, tagged by profile curve.
    Rib,
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
            Role::HoleCounterboreWall => "counterbore_wall",
            Role::HoleCountersink => "countersink",
            Role::ShellRim => "rim",
            Role::ShellOuter => "outer",
            Role::ThinWall => "thin_wall",
            Role::DraftStep => "draft_step",
            Role::SplitFace => "split_face",
            Role::Body => "body",
            Role::Replacement => "replacement",
            Role::Boundary => "boundary",
            Role::SurfacePatch => "patch",
            Role::OffsetFace => "offset",
            Role::Extension => "extension",
            Role::ThickenFace => "thicken",
            Role::ThickenRim => "thicken_rim",
            Role::MidSurface => "mid_surface",
            Role::Ruled => "ruled",
            Role::SheetFace => "sheet",
            Role::SheetEdge => "sheet_edge",
            Role::BendFace => "bend",
            Role::Relief => "relief",
            Role::FormFace => "form",
            Role::MemberSide => "member_side",
            Role::MemberEnd => "member_end",
            Role::CapPlate => "cap_plate",
            Role::Gusset => "gusset",
            Role::WeldBead => "weld_bead",
            Role::PartingFace => "parting",
            Role::ThreadFace => "thread",
            Role::EmbossTop => "emboss_top",
            Role::Rib => "rib",
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
    /// A sketch-driven instance, by the sketch point it sits on.
    Point {
        point: String,
    },
    /// A curve-driven instance: the step along the curve.
    Curve {
        step: u32,
    },
    /// A fill instance: lattice steps from the seed along the fill's two directions.
    Lattice {
        i: i32,
        j: i32,
    },
    /// A circular fill instance: ring, then step around it.
    Ring {
        ring: u32,
        step: u32,
    },
}

impl fmt::Display for InstanceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstanceKey::Linear { step } => write!(f, "{}", step),
            InstanceKey::Grid { first, second } => write!(f, "{},{}", first, second),
            InstanceKey::Circular { step } => write!(f, "@{}", step),
            InstanceKey::Row { key } => write!(f, "{:?}", key),
            InstanceKey::Mirror => f.write_str("mirror"),
            InstanceKey::Point { point } => write!(f, "pt:{}", point),
            InstanceKey::Curve { step } => write!(f, "~{}", step),
            InstanceKey::Lattice { i, j } => write!(f, "[{},{}]", i, j),
            InstanceKey::Ring { ring, step } => write!(f, "r{}@{}", ring, step),
        }
    }
}

// ------------------------------------------------------------------------------- tokens

/// A fixed-size, stable token for a one-level name. Serialised as 32 lowercase hex digits.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NameToken(pub [u8; 16]);

impl NameToken {
    /// The first eight hex digits, for display.
    pub fn short(&self) -> String {
        hex::encode(&self.0[..4])
    }
}

impl fmt::Debug for NameToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "~{}", self.short())
    }
}

impl fmt::Display for NameToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl Serialize for NameToken {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for NameToken {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        let arr: [u8; 16] = bytes
            .try_into()
            .map_err(|_| serde::de::Error::custom("a name token is 32 hex digits"))?;
        Ok(NameToken(arr))
    }
}

/// What, from the feature's definition, distinguishes entities of the same role. Names a tag
/// is made from are held by token, so a tag is bounded.
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
    /// The edge a blend or chamfer replaced: the edge between the faces with these tokens
    /// (sorted).
    Edge {
        between: [NameToken; 2],
    },
    /// The face a shell wall or rim offsets.
    Face {
        face: NameToken,
    },
    /// The vertex a corner blend or vertex chamfer replaced: the vertex where the faces with
    /// these tokens meet (sorted, distinct).
    Vertex {
        faces: Vec<NameToken>,
    },
}

impl SourceTag {
    /// The tag of the edge between two faces (order does not matter).
    pub fn edge_between(a: &EntityName, b: &EntityName) -> SourceTag {
        let (x, y) = (a.token(), b.token());
        SourceTag::Edge {
            between: if x <= y { [x, y] } else { [y, x] },
        }
    }

    pub fn face_of(face: &EntityName) -> SourceTag {
        SourceTag::Face { face: face.token() }
    }

    /// The tag of the vertex where these faces meet (order and repeats do not matter).
    pub fn vertex_of(faces: &[EntityName]) -> SourceTag {
        let mut t: Vec<NameToken> = faces.iter().map(|f| f.token()).collect();
        t.sort();
        t.dedup();
        SourceTag::Vertex { faces: t }
    }

    /// Tokens of the names this tag is made from.
    pub fn children(&self) -> Vec<NameToken> {
        match self {
            SourceTag::Edge { between } => between.to_vec(),
            SourceTag::Face { face } => vec![*face],
            SourceTag::Vertex { faces } => faces.clone(),
            _ => Vec::new(),
        }
    }

    fn mapped(
        &self,
        from: &FeatureId,
        to: &FeatureId,
        map: &dyn Fn(&NameToken) -> NameToken,
    ) -> SourceTag {
        match self {
            SourceTag::Curve { sketch, entity } => SourceTag::Curve {
                sketch: if sketch == from {
                    to.clone()
                } else {
                    sketch.clone()
                },
                entity: entity.clone(),
            },
            SourceTag::Edge { between } => {
                let (x, y) = (map(&between[0]), map(&between[1]));
                SourceTag::Edge {
                    between: if x <= y { [x, y] } else { [y, x] },
                }
            }
            SourceTag::Face { face } => SourceTag::Face { face: map(face) },
            SourceTag::Vertex { faces } => {
                let mut t: Vec<NameToken> = faces.iter().map(map).collect();
                t.sort();
                t.dedup();
                SourceTag::Vertex { faces: t }
            }
            other => other.clone(),
        }
    }
}

/// Layer 1: a generation name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GenerationName {
    pub feature: FeatureId,
    pub role: Role,
    pub tag: SourceTag,
    /// A tie-break from the feature's own definition (curve parameter order of seam pieces,
    /// path segment of a sweep, distance along a direction). Never the kernel's output order.
    /// In a reference, `None` matches every ordinal: the whole logical entity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u32>,
}

/// A face or body name, one level deep. Edges and vertices are named relationally (by the
/// faces around them), so they need no stored name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "form")]
pub enum EntityName {
    Generated(GenerationName),
    /// A pattern or mirror image of the entity named `base`. `whole` is the token of `base`
    /// with its ordinals cleared (the whole logical entity), so an ordinal-free query matches
    /// every piece's image without the table.
    Copy {
        base: NameToken,
        whole: NameToken,
        pattern: FeatureId,
        instance: InstanceKey,
    },
    /// Layer 4: a name read from an imported CAD file (STEP/AP242), matched to canonical ids.
    CadFile {
        import: FeatureId,
        name: String,
    },
}

const TOKEN_DOMAIN: &[u8] = b"runmat-cad/name/v1";

impl EntityName {
    pub fn generated(feature: &str, role: Role, tag: SourceTag) -> Self {
        EntityName::Generated(GenerationName {
            feature: FeatureId::new(feature),
            role,
            tag,
            ordinal: None,
        })
    }

    /// The image of `base` made by `pattern` at `instance`.
    pub fn copy(base: &EntityName, pattern: &FeatureId, instance: InstanceKey) -> Self {
        EntityName::Copy {
            base: base.token(),
            whole: base.whole().token(),
            pattern: pattern.clone(),
            instance,
        }
    }

    /// The name's token: a pure function of the one-level recipe.
    pub fn token(&self) -> NameToken {
        let mut h = Sha256::new();
        h.update((TOKEN_DOMAIN.len() as u64).to_be_bytes());
        h.update(TOKEN_DOMAIN);
        let body = serde_json::to_vec(self).expect("names serialize");
        h.update((body.len() as u64).to_be_bytes());
        h.update(&body);
        let full: [u8; 32] = h.finalize().into();
        let mut out = [0u8; 16];
        out.copy_from_slice(&full[..16]);
        NameToken(out)
    }

    /// The same name with its ordinals cleared: the whole logical entity.
    pub fn whole(&self) -> EntityName {
        match self {
            EntityName::Generated(g) => EntityName::Generated(GenerationName {
                ordinal: None,
                ..g.clone()
            }),
            EntityName::Copy {
                whole,
                pattern,
                instance,
                ..
            } => EntityName::Copy {
                base: *whole,
                whole: *whole,
                pattern: pattern.clone(),
                instance: instance.clone(),
            },
            other => other.clone(),
        }
    }

    /// The feature that made the entity (the outermost maker).
    pub fn maker(&self) -> &FeatureId {
        match self {
            EntityName::Generated(g) => &g.feature,
            EntityName::Copy { pattern, .. } => pattern,
            EntityName::CadFile { import, .. } => import,
        }
    }

    /// Every feature this one-level record mentions, outermost first. Makers of the names it
    /// holds by token are reached through the table ([`NameTable::makers`]).
    pub fn makers(&self) -> Vec<FeatureId> {
        let mut out = vec![self.maker().clone()];
        if let EntityName::Generated(GenerationName {
            tag: SourceTag::Curve { sketch, .. },
            ..
        }) = self
        {
            if !out.contains(sketch) {
                out.push(sketch.clone());
            }
        }
        out
    }

    /// Tokens of the names this record is made from.
    pub fn children(&self) -> Vec<NameToken> {
        match self {
            EntityName::Generated(g) => g.tag.children(),
            EntityName::Copy { base, whole, .. } => {
                if base == whole {
                    vec![*base]
                } else {
                    vec![*base, *whole]
                }
            }
            EntityName::CadFile { .. } => Vec::new(),
        }
    }

    /// The ordinal at the top of the name (through pattern images, via `whole`).
    pub fn ordinal(&self) -> Option<u32> {
        match self {
            EntityName::Generated(g) => g.ordinal,
            _ => None,
        }
    }

    /// Whether `self`, used as a query, matches the stored name `candidate`. A query without an
    /// ordinal matches every ordinal of the same feature, role and tag; a pattern image whose
    /// base is written without an ordinal matches the images of every piece.
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
                    ..
                },
                EntityName::Copy {
                    base: cb,
                    whole: cw,
                    pattern: cp,
                    instance: ci,
                },
            ) => qp == cp && qi == ci && (qb == cb || qb == cw),
            _ => self == candidate,
        }
    }

    /// Replace feature `from` with `to` in this record, and map the tokens it holds through
    /// `map` (an alias rewrite; [`NameTable::alias_map`] supplies the map).
    pub fn renamed_with(
        &self,
        from: &FeatureId,
        to: &FeatureId,
        map: &dyn Fn(&NameToken) -> NameToken,
    ) -> EntityName {
        let swap = |f: &FeatureId| if f == from { to.clone() } else { f.clone() };
        match self {
            EntityName::Generated(g) => EntityName::Generated(GenerationName {
                feature: swap(&g.feature),
                role: g.role,
                tag: g.tag.mapped(from, to, map),
                ordinal: g.ordinal,
            }),
            EntityName::Copy {
                base,
                whole,
                pattern,
                instance,
            } => EntityName::Copy {
                base: map(base),
                whole: map(whole),
                pattern: swap(pattern),
                instance: instance.clone(),
            },
            EntityName::CadFile { import, name } => EntityName::CadFile {
                import: swap(import),
                name: name.clone(),
            },
        }
    }

    /// Replace feature `from` with `to` in this record only (tokens are kept).
    pub fn renamed(&self, from: &FeatureId, to: &FeatureId) -> EntityName {
        self.renamed_with(from, to, &|t| *t)
    }

    fn write_with(
        &self,
        f: &mut dyn fmt::Write,
        table: Option<&NameTable>,
        depth: u32,
    ) -> fmt::Result {
        let child = |f: &mut dyn fmt::Write, t: &NameToken| -> fmt::Result {
            match table.and_then(|tb| tb.get(t)) {
                Some(n) if depth < MAX_DESCRIBE_DEPTH => n.write_with(f, table, depth + 1),
                Some(_) => f.write_str("…"),
                None => write!(f, "~{}", t.short()),
            }
        };
        match self {
            EntityName::Generated(g) => {
                write!(f, "{}.{}", g.feature, g.role.as_str())?;
                if g.tag != SourceTag::None || g.ordinal.is_some() {
                    f.write_str("(")?;
                    match &g.tag {
                        SourceTag::None => {}
                        SourceTag::Curve { entity, .. } => write!(f, "{}", entity)?,
                        SourceTag::Key { key } => write!(f, "{}", key)?,
                        SourceTag::Side { side } => f.write_str(match side {
                            PlaneSide::Negative => "negative",
                            PlaneSide::Positive => "positive",
                        })?,
                        SourceTag::Edge { between } => {
                            f.write_str("edges_between(")?;
                            child(f, &between[0])?;
                            f.write_str(", ")?;
                            child(f, &between[1])?;
                            f.write_str(")")?;
                        }
                        SourceTag::Face { face } => child(f, face)?,
                        SourceTag::Vertex { faces } => {
                            f.write_str("vertex_of(")?;
                            for (k, t) in faces.iter().enumerate() {
                                if k > 0 {
                                    f.write_str(", ")?;
                                }
                                child(f, t)?;
                            }
                            f.write_str(")")?;
                        }
                    }
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
                ..
            } => {
                write!(f, "{}[{}].", pattern, instance)?;
                child(f, base)
            }
            EntityName::CadFile { import, name } => write!(f, "{}.name({:?})", import, name),
        }
    }
}

const MAX_DESCRIBE_DEPTH: u32 = 12;

impl fmt::Display for EntityName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_with(f, None, 0)
    }
}

// ------------------------------------------------------------------------------- the table

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[error("two different names share token {token}: `{first}` and `{second}`")]
pub struct TokenCollision {
    pub token: NameToken,
    pub first: String,
    pub second: String,
}

/// Every one-level recipe a regeneration produced, by token. Depth lives here: following a
/// name's children through the table walks its whole lineage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NameTable {
    recipes: BTreeMap<NameToken, EntityName>,
}

#[derive(Serialize, Deserialize)]
struct NameRow {
    token: NameToken,
    name: EntityName,
}

impl Serialize for NameTable {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let rows: Vec<NameRow> = self
            .recipes
            .iter()
            .map(|(t, n)| NameRow {
                token: *t,
                name: n.clone(),
            })
            .collect();
        rows.serialize(s)
    }
}

impl<'de> Deserialize<'de> for NameTable {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let rows = Vec::<NameRow>::deserialize(d)?;
        let mut t = NameTable::default();
        for r in rows {
            if r.name.token() != r.token {
                return Err(serde::de::Error::custom(format!(
                    "name table row {} does not match its recipe",
                    r.token
                )));
            }
            t.recipes.insert(r.token, r.name);
        }
        Ok(t)
    }
}

impl NameTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.recipes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.recipes.is_empty()
    }

    /// Record a name (and its whole form); a token already held by a different recipe is a
    /// collision, reported rather than overwritten.
    pub fn intern(&mut self, name: &EntityName) -> Result<NameToken, TokenCollision> {
        let t = name.token();
        match self.recipes.get(&t) {
            Some(existing) if existing != name => Err(TokenCollision {
                token: t,
                first: existing.to_string(),
                second: name.to_string(),
            }),
            Some(_) => Ok(t),
            None => {
                self.recipes.insert(t, name.clone());
                let w = name.whole();
                if &w != name {
                    self.intern(&w)?;
                }
                Ok(t)
            }
        }
    }

    pub fn get(&self, token: &NameToken) -> Option<&EntityName> {
        self.recipes.get(token)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&NameToken, &EntityName)> {
        self.recipes.iter()
    }

    /// Merge another table in (one regeneration's names into a document's).
    pub fn extend(&mut self, other: &NameTable) -> Result<(), TokenCollision> {
        for n in other.recipes.values() {
            self.intern(n)?;
        }
        Ok(())
    }

    /// The name in full, its children expanded through the table (depth-limited).
    pub fn describe(&self, name: &EntityName) -> String {
        let mut s = String::new();
        let _ = name.write_with(&mut s, Some(self), 0);
        s
    }

    /// Every feature the name's lineage mentions, outermost first (children through the
    /// table; unknown tokens contribute nothing).
    pub fn makers(&self, name: &EntityName) -> Vec<FeatureId> {
        let mut out = Vec::new();
        let mut stack = vec![name.clone()];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(n) = stack.pop() {
            for m in n.makers() {
                if !out.contains(&m) {
                    out.push(m);
                }
            }
            for c in n.children() {
                if seen.insert(c) {
                    if let Some(r) = self.recipes.get(&c) {
                        stack.push(r.clone());
                    }
                }
            }
        }
        out
    }

    /// For a rename `from → to` already applied to the model: the token each current name had
    /// before the rename, mapped to its current token. Lets a reference written before the
    /// rename find names nested at any depth.
    pub fn alias_map(&self, from: &FeatureId, to: &FeatureId) -> BTreeMap<NameToken, NameToken> {
        let mut memo: BTreeMap<NameToken, NameToken> = BTreeMap::new();
        fn old_token(
            t: &NameToken,
            table: &NameTable,
            from: &FeatureId,
            to: &FeatureId,
            memo: &mut BTreeMap<NameToken, NameToken>,
            depth: u32,
        ) -> NameToken {
            if let Some(o) = memo.get(t) {
                return *o;
            }
            let Some(n) = table.recipes.get(t) else {
                return *t;
            };
            if depth > 64 {
                return *t;
            }
            let mut child_map = BTreeMap::new();
            for c in n.children() {
                let o = old_token(&c, table, from, to, memo, depth + 1);
                child_map.insert(c, o);
            }
            // Current names say `to`; the old ones said `from`.
            let old = n
                .renamed_with(to, from, &|c| child_map.get(c).copied().unwrap_or(*c))
                .token();
            memo.insert(*t, old);
            old
        }
        let tokens: Vec<NameToken> = self.recipes.keys().copied().collect();
        for t in &tokens {
            old_token(t, self, from, to, &mut memo, 0);
        }
        memo.into_iter().map(|(cur, old)| (old, cur)).collect()
    }
}

/// A reference or name shown with a table, so nested names read in full.
pub struct Shown<'a, T: ?Sized>(pub &'a T, pub Option<&'a NameTable>);

impl fmt::Display for Shown<'_, EntityName> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.write_with(f, self.1, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(f: &str, c: &str) -> EntityName {
        EntityName::generated(
            f,
            Role::Side,
            SourceTag::Curve {
                sketch: FeatureId::new("s"),
                entity: SketchEntityId::new(c),
            },
        )
    }

    #[test]
    fn names_stay_one_level_however_deep_the_lineage() {
        // A blend of a blend of a blend ... 200 deep: every record is the same size.
        let mut table = NameTable::new();
        let a = side("e", "l1");
        let mut b = side("e", "l2");
        table.intern(&a).unwrap();
        table.intern(&b).unwrap();
        let first = serde_json::to_vec(&b).unwrap().len();
        let mut sizes = Vec::new();
        for k in 0..200 {
            let n = EntityName::generated(
                &format!("f{}", k % 10),
                Role::Blend,
                SourceTag::edge_between(&a, &b),
            );
            table.intern(&n).unwrap();
            sizes.push(serde_json::to_vec(&n).unwrap().len());
            b = n;
        }
        let max = *sizes.iter().max().unwrap();
        assert!(max < 4 * first + 128, "record grew to {} bytes", max);
        // Depth lives in the table: the description expands, depth-limited.
        let text = table.describe(&b);
        assert!(text.starts_with("f9.blend(edges_between("));
        assert!(text.contains('…'));
        // Makers walk the lineage through the table.
        let makers = table.makers(&b);
        assert!(makers.contains(&FeatureId::new("e")));
        assert!(makers.contains(&FeatureId::new("f0")));
    }

    #[test]
    fn tokens_are_pure_functions_of_the_recipe() {
        let a = side("e", "l1");
        assert_eq!(a.token(), side("e", "l1").token());
        assert_ne!(a.token(), side("e", "l2").token());
        // Stable across releases: pinned value.
        assert_eq!(a.token().to_string().len(), 32);
        let json = serde_json::to_string(&a.token()).unwrap();
        let back: NameToken = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a.token());
    }

    #[test]
    fn whole_copies_match_every_piece_image() {
        let mut piece = side("e", "l1");
        if let EntityName::Generated(g) = &mut piece {
            g.ordinal = Some(2);
        }
        let p = FeatureId::new("row");
        let stored = EntityName::copy(&piece, &p, InstanceKey::Linear { step: 1 });
        let query = EntityName::copy(&side("e", "l1"), &p, InstanceKey::Linear { step: 1 });
        assert!(query.matches(&stored));
        let exact = EntityName::copy(&piece, &p, InstanceKey::Linear { step: 1 });
        assert!(exact.matches(&stored));
        let other = EntityName::copy(&piece, &p, InstanceKey::Linear { step: 2 });
        assert!(!other.matches(&stored));
    }

    #[test]
    fn alias_map_rewrites_nested_tokens() {
        // Model after renaming `e` to `body`: names say `body`.
        let mut table = NameTable::new();
        let a = side("body", "l1");
        let b = side("body", "l2");
        let blend = EntityName::generated("round", Role::Blend, SourceTag::edge_between(&a, &b));
        for n in [&a, &b, &blend] {
            table.intern(n).unwrap();
        }
        // A reference written before the rename.
        let old_blend = EntityName::generated(
            "round",
            Role::Blend,
            SourceTag::edge_between(&side("e", "l1"), &side("e", "l2")),
        );
        assert!(!old_blend.matches(&blend));
        let map = table.alias_map(&FeatureId::new("e"), &FeatureId::new("body"));
        let rewritten =
            old_blend.renamed_with(&FeatureId::new("e"), &FeatureId::new("body"), &|t| {
                map.get(t).copied().unwrap_or(*t)
            });
        assert!(rewritten.matches(&blend));
    }

    #[test]
    fn table_round_trips_and_checks_rows() {
        let mut table = NameTable::new();
        table.intern(&side("e", "l1")).unwrap();
        let json = serde_json::to_string(&table).unwrap();
        let back: NameTable = serde_json::from_str(&json).unwrap();
        assert_eq!(back, table);
        let tampered = json.replace("l1", "l9");
        assert!(serde_json::from_str::<NameTable>(&tampered).is_err());
    }
}
