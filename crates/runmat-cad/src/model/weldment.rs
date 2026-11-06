//! Weldments: structural members from a profile library along sketch segments, trim and
//! extend, end caps, gussets, weld beads, and the automatic cut list.
//!
//! A member is one body per sketch segment (so the cut list is one row per body), keyed by its
//! segment's curve id: faces are tagged `member/profile-curve`, ends by ordinal, so references
//! survive re-sketching, re-ordering and profile changes that keep curve ids.

use crate::ids::{FeatureId, SketchEntityId};
use crate::model::visit::{Params, Visit};
use crate::refs::Reference;
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// A profile from the library: a family plus its sizes, as written (`ValueExpr`, so `20 mm`
/// and `input.wall` work) or evaluated to SI ([`ProfileSizes`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "profile")]
pub enum MemberProfile<T = ValueExpr> {
    /// Square or rectangular tube.
    RectTube {
        width: T,
        height: T,
        wall: T,
    },
    Pipe {
        diameter: T,
        wall: T,
    },
    /// Equal or unequal angle.
    Angle {
        a: T,
        b: T,
        thickness: T,
    },
    Channel {
        depth: T,
        flange: T,
        web: T,
        flange_thickness: T,
    },
    IBeam {
        depth: T,
        flange: T,
        web: T,
        flange_thickness: T,
    },
    FlatBar {
        width: T,
        thickness: T,
    },
    RoundBar {
        diameter: T,
    },
}

/// A profile's sizes evaluated to SI lengths.
pub type ProfileSizes = MemberProfile<f64>;

impl<T> MemberProfile<T> {
    /// Each size with its parameter name, mapped by `f` (evaluation, in regeneration).
    pub fn try_map<U, E>(&self, mut f: impl FnMut(&'static str, &T) -> Result<U, E>) -> Result<MemberProfile<U>, E> {
        use MemberProfile as P;
        Ok(match self {
            P::RectTube { width, height, wall } => P::RectTube { width: f("width", width)?, height: f("height", height)?, wall: f("wall", wall)? },
            P::Pipe { diameter, wall } => P::Pipe { diameter: f("diameter", diameter)?, wall: f("wall", wall)? },
            P::Angle { a, b, thickness } => P::Angle { a: f("a", a)?, b: f("b", b)?, thickness: f("thickness", thickness)? },
            P::Channel { depth, flange, web, flange_thickness } => P::Channel {
                depth: f("depth", depth)?,
                flange: f("flange", flange)?,
                web: f("web", web)?,
                flange_thickness: f("flange_thickness", flange_thickness)?,
            },
            P::IBeam { depth, flange, web, flange_thickness } => P::IBeam {
                depth: f("depth", depth)?,
                flange: f("flange", flange)?,
                web: f("web", web)?,
                flange_thickness: f("flange_thickness", flange_thickness)?,
            },
            P::FlatBar { width, thickness } => P::FlatBar { width: f("width", width)?, thickness: f("thickness", thickness)? },
            P::RoundBar { diameter } => P::RoundBar { diameter: f("diameter", diameter)? },
        })
    }

    /// The sizes, in declaration order.
    pub fn sizes(&self) -> Vec<&T> {
        use MemberProfile as P;
        match self {
            P::RectTube { width, height, wall } => vec![width, height, wall],
            P::Pipe { diameter, wall } => vec![diameter, wall],
            P::Angle { a, b, thickness } => vec![a, b, thickness],
            P::Channel { depth, flange, web, flange_thickness } | P::IBeam { depth, flange, web, flange_thickness } => vec![depth, flange, web, flange_thickness],
            P::FlatBar { width, thickness } => vec![width, thickness],
            P::RoundBar { diameter } => vec![diameter],
        }
    }
}

/// One closed polygon (outer loop first, holes after) per profile, in the member frame
/// (x across, y up), each edge with a stable curve id.
pub type ProfileOutline = Vec<Vec<(SketchEntityId, [f64; 2])>>;

impl ProfileSizes {
    /// The cross-section area (for mass and the cut list).
    pub fn area(&self) -> f64 {
        match *self {
            MemberProfile::RectTube {
                width,
                height,
                wall,
            } => width * height - (width - 2.0 * wall) * (height - 2.0 * wall),
            MemberProfile::Pipe { diameter, wall } => {
                PI / 4.0 * (diameter * diameter - (diameter - 2.0 * wall).powi(2))
            }
            MemberProfile::Angle { a, b, thickness } => (a + b - thickness) * thickness,
            MemberProfile::Channel {
                depth,
                flange,
                web,
                flange_thickness,
            } => 2.0 * flange * flange_thickness + (depth - 2.0 * flange_thickness) * web,
            MemberProfile::IBeam {
                depth,
                flange,
                web,
                flange_thickness,
            } => 2.0 * flange * flange_thickness + (depth - 2.0 * flange_thickness) * web,
            MemberProfile::FlatBar { width, thickness } => width * thickness,
            MemberProfile::RoundBar { diameter } => PI / 4.0 * diameter * diameter,
        }
    }

    /// The cut list's description: the family and its sizes in millimetres, as a fabricator
    /// orders the stock (`Square tube 20 × 20 × 2`, `Pipe ⌀21.3 × 2.6`).
    pub fn description(&self) -> String {
        fn mm(v: f64) -> String {
            let s = format!("{:.2}", v * 1000.0);
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        }
        match *self {
            MemberProfile::RectTube {
                width,
                height,
                wall,
            } => {
                let family = if (width - height).abs() < 1e-12 {
                    "Square tube"
                } else {
                    "Rect tube"
                };
                format!("{family} {} × {} × {}", mm(width), mm(height), mm(wall))
            }
            MemberProfile::Pipe { diameter, wall } => {
                format!("Pipe ⌀{} × {}", mm(diameter), mm(wall))
            }
            MemberProfile::Angle { a, b, thickness } => {
                format!("Angle {} × {} × {}", mm(a), mm(b), mm(thickness))
            }
            MemberProfile::Channel {
                depth,
                flange,
                web,
                flange_thickness,
            } => format!(
                "Channel {} × {} × {} × {}",
                mm(depth),
                mm(flange),
                mm(web),
                mm(flange_thickness)
            ),
            MemberProfile::IBeam {
                depth,
                flange,
                web,
                flange_thickness,
            } => format!(
                "I-beam {} × {} × {} × {}",
                mm(depth),
                mm(flange),
                mm(web),
                mm(flange_thickness)
            ),
            MemberProfile::FlatBar { width, thickness } => {
                format!("Flat bar {} × {}", mm(width), mm(thickness))
            }
            MemberProfile::RoundBar { diameter } => format!("Round bar ⌀{}", mm(diameter)),
        }
    }

    /// The outline as polygons; round profiles are sampled with `segments` sides.
    pub fn outline(&self, segments: usize) -> ProfileOutline {
        let id = |s: &str| SketchEntityId::new(s);
        let rect = |w: f64, h: f64, pre: &str| -> Vec<(SketchEntityId, [f64; 2])> {
            vec![
                (id(&format!("{}bottom", pre)), [-w / 2.0, -h / 2.0]),
                (id(&format!("{}right", pre)), [w / 2.0, -h / 2.0]),
                (id(&format!("{}top", pre)), [w / 2.0, h / 2.0]),
                (id(&format!("{}left", pre)), [-w / 2.0, h / 2.0]),
            ]
        };
        let circle = |d: f64, pre: &str| -> Vec<(SketchEntityId, [f64; 2])> {
            (0..segments)
                .map(|k| {
                    let a = 2.0 * PI * k as f64 / segments as f64;
                    (id(pre), [d / 2.0 * a.cos(), d / 2.0 * a.sin()])
                })
                .collect()
        };
        match *self {
            MemberProfile::RectTube {
                width,
                height,
                wall,
            } => vec![
                rect(width, height, ""),
                rect(width - 2.0 * wall, height - 2.0 * wall, "inner_"),
            ],
            MemberProfile::Pipe { diameter, wall } => {
                vec![
                    circle(diameter, "outer"),
                    circle(diameter - 2.0 * wall, "inner"),
                ]
            }
            MemberProfile::RoundBar { diameter } => vec![circle(diameter, "outer")],
            MemberProfile::FlatBar { width, thickness } => vec![rect(width, thickness, "")],
            MemberProfile::Angle { a, b, thickness } => vec![vec![
                (id("heel_bottom"), [0.0, 0.0]),
                (id("leg_b_end"), [b, 0.0]),
                (id("leg_b_top"), [b, thickness]),
                (id("inner_b"), [thickness, thickness]),
                (id("leg_a_end"), [thickness, a]),
                (id("heel_side"), [0.0, a]),
            ]],
            MemberProfile::Channel {
                depth,
                flange,
                web,
                flange_thickness: tf,
            } => vec![vec![
                (id("back"), [0.0, 0.0]),
                (id("bottom_flange"), [flange, 0.0]),
                (id("bottom_toe"), [flange, tf]),
                (id("bottom_inner"), [web, tf]),
                (id("web_inner"), [web, depth - tf]),
                (id("top_inner"), [flange, depth - tf]),
                (id("top_toe"), [flange, depth]),
                (id("top_flange"), [0.0, depth]),
            ]],
            MemberProfile::IBeam {
                depth,
                flange,
                web,
                flange_thickness: tf,
            } => {
                let (h, f, w) = (depth / 2.0, flange / 2.0, web / 2.0);
                vec![vec![
                    (id("bottom"), [-f, -h]),
                    (id("bottom_right_toe"), [f, -h]),
                    (id("bottom_right_inner"), [f, -h + tf]),
                    (id("web_right"), [w, -h + tf]),
                    (id("top_right_inner"), [w, h - tf]),
                    (id("top_right_toe"), [f, h - tf]),
                    (id("top"), [f, h]),
                    (id("top_left_toe"), [-f, h]),
                    (id("top_left_inner"), [-f, h - tf]),
                    (id("web_left"), [-w, h - tf]),
                    (id("bottom_left_inner"), [-w, -h + tf]),
                    (id("bottom_left_toe"), [-f, -h + tf]),
                ]]
            }
        }
    }
}

/// How two members meet at a shared sketch point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CornerTreatment {
    /// Both ends cut on the bisector.
    #[default]
    Miter,
    /// The first member runs through; the second butts against it.
    Butt1,
    Butt2,
    /// Neither end is trimmed.
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemberGroup {
    pub sketch: FeatureId,
    /// Segments of the sketch (lines), each one member.
    pub segments: Vec<SketchEntityId>,
    pub profile: MemberProfile,
    /// Material grade for the cut list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default)]
    pub corner: CornerTreatment,
    /// Rotation of the profile about the segment (radians).
    #[serde(default = "zero_angle")]
    pub rotation: ValueExpr,
}

fn zero_angle() -> ValueExpr {
    ValueExpr::deg(0.0)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "weld")]
pub enum WeldBeadKind {
    Fillet { leg: ValueExpr },
    Groove { depth: ValueExpr },
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "weldment")]
pub enum WeldmentFeature {
    StructuralMember {
        groups: Vec<MemberGroup>,
    },
    /// Trim or extend members to a face (or to each other).
    TrimExtend {
        members: Vec<Reference>,
        to: Reference,
    },
    /// A plate closing a member's open end.
    EndCap {
        end_face: Reference,
        thickness: ValueExpr,
        #[serde(default)]
        inset: Option<ValueExpr>,
    },
    /// A triangular plate between two members' faces at a corner.
    Gusset {
        faces: [Reference; 2],
        leg: ValueExpr,
        thickness: ValueExpr,
    },
    /// A weld bead along the edges between members.
    WeldBead {
        edges: Vec<Reference>,
        kind: WeldBeadKind,
    },
}

impl Params for WeldmentFeature {
    fn visit<'a>(&'a self, v: &mut Visit<'a>) {
        match self {
            WeldmentFeature::StructuralMember { groups } => {
                for g in groups {
                    v.feature(&g.sketch);
                    v.value(&g.rotation);
                    for size in g.profile.sizes() {
                        v.value(size);
                    }
                }
            }
            WeldmentFeature::TrimExtend { members, to } => {
                v.references("members", members);
                v.reference("to", to);
            }
            WeldmentFeature::EndCap {
                end_face,
                thickness,
                inset,
            } => {
                v.reference("end_face", end_face);
                v.value(thickness);
                v.opt_value(inset);
            }
            WeldmentFeature::Gusset {
                faces,
                leg,
                thickness,
            } => {
                v.reference("faces[0]", &faces[0]);
                v.reference("faces[1]", &faces[1]);
                v.value(leg);
                v.value(thickness);
            }
            WeldmentFeature::WeldBead { edges, kind } => {
                v.references("edges", edges);
                match kind {
                    WeldBeadKind::Fillet { leg } => v.value(leg),
                    WeldBeadKind::Groove { depth } => v.value(depth),
                }
            }
        }
    }
    fn kind_name(&self) -> &'static str {
        match self {
            WeldmentFeature::StructuralMember { .. } => "weldment.member",
            WeldmentFeature::TrimExtend { .. } => "weldment.trim_extend",
            WeldmentFeature::EndCap { .. } => "weldment.end_cap",
            WeldmentFeature::Gusset { .. } => "weldment.gusset",
            WeldmentFeature::WeldBead { .. } => "weldment.weld_bead",
        }
    }
    fn new_body(&self) -> bool {
        !matches!(self, WeldmentFeature::TrimExtend { .. })
    }
}

// ------------------------------------------------------------------------- cut list

/// A member's computed ends: the segment, its trimmed length and the end cut angles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemberCut {
    pub member: SketchEntityId,
    pub start: [f64; 3],
    pub end: [f64; 3],
    /// Length along the member's axis between the two ends' extreme points.
    pub length: f64,
    /// Cut angle of each end from square (radians): a 90° mitre corner gives 45°.
    pub start_angle: f64,
    pub end_angle: f64,
}

/// One cut-list row: identical members grouped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CutListRow {
    pub item: u32,
    pub quantity: u32,
    pub description: String,
    pub material: Option<String>,
    pub length: f64,
    pub start_angle: f64,
    pub end_angle: f64,
    pub members: Vec<SketchEntityId>,
    pub mass_per_member: Option<f64>,
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// End cuts for a group's members (segments as 3D end points): at a point two members share,
/// a mitre cuts each on the bisector (angle half the deviation); a butt joint leaves the
/// through member square and extends the butting member to the through member's far face
/// (`profile_depth` across). The length is measured on the long side of each cut.
pub fn member_cuts(
    segments: &[(SketchEntityId, [f64; 3], [f64; 3])],
    corner: CornerTreatment,
    profile_depth: f64,
) -> Vec<MemberCut> {
    let key = |p: [f64; 3]| {
        [
            (p[0] * 1e9).round() as i64,
            (p[1] * 1e9).round() as i64,
            (p[2] * 1e9).round() as i64,
        ]
    };
    let mut out = Vec::new();
    for (i, (id, a, b)) in segments.iter().enumerate() {
        let d = sub3(*b, *a);
        let l = len3(d);
        let dir = [d[0] / l, d[1] / l, d[2] / l];
        let mut angles = [0.0f64; 2];
        let mut extra = [0.0f64; 2];
        for (end, p) in [(0usize, *a), (1usize, *b)] {
            // Another member sharing this point.
            let other = segments
                .iter()
                .enumerate()
                .find(|(j, (_, c, e))| *j != i && (key(*c) == key(p) || key(*e) == key(p)));
            let Some((j, (_, c, e))) = other else {
                continue;
            };
            let od = if key(*c) == key(p) {
                sub3(*e, *c)
            } else {
                sub3(*c, *e)
            };
            let ol = len3(od);
            let mine = if end == 0 {
                dir
            } else {
                [-dir[0], -dir[1], -dir[2]]
            };
            let cosv = (dot3(mine, od) / ol).clamp(-1.0, 1.0);
            // The angle between the two members leaving the joint.
            let between = cosv.acos();
            match corner {
                CornerTreatment::Miter => {
                    angles[end] = (PI - between) / 2.0;
                    extra[end] = profile_depth / 2.0 * angles[end].tan();
                }
                CornerTreatment::Butt1 | CornerTreatment::Butt2 => {
                    let through = (corner == CornerTreatment::Butt1) == (i < j);
                    if through {
                        extra[end] = profile_depth / 2.0;
                    } else {
                        extra[end] = -profile_depth / 2.0;
                    }
                }
                CornerTreatment::None => {}
            }
        }
        out.push(MemberCut {
            member: id.clone(),
            start: *a,
            end: *b,
            length: l + extra[0] + extra[1],
            start_angle: angles[0],
            end_angle: angles[1],
        });
    }
    out
}

/// Group members with equal profile description, material, length and end angles (to
/// `tolerance`) into numbered rows, longest first.
/// One group of members for the cut list: profile description, material, mass per metre
/// (when known) and the members' cuts.
pub type CutGroup = (String, Option<String>, Option<f64>, Vec<MemberCut>);

pub fn cut_list(groups: &[CutGroup], tolerance: f64) -> Vec<CutListRow> {
    let mut rows: Vec<CutListRow> = Vec::new();
    for (desc, material, density_area, cuts) in groups {
        for c in cuts {
            let (a0, a1) = if c.start_angle <= c.end_angle {
                (c.start_angle, c.end_angle)
            } else {
                (c.end_angle, c.start_angle)
            };
            let found = rows.iter_mut().find(|r| {
                &r.description == desc
                    && &r.material == material
                    && (r.length - c.length).abs() <= tolerance
                    && (r.start_angle - a0).abs() <= 1e-9
                    && (r.end_angle - a1).abs() <= 1e-9
            });
            match found {
                Some(r) => {
                    r.quantity += 1;
                    r.members.push(c.member.clone());
                }
                None => rows.push(CutListRow {
                    item: 0,
                    quantity: 1,
                    description: desc.clone(),
                    material: material.clone(),
                    length: c.length,
                    start_angle: a0,
                    end_angle: a1,
                    members: vec![c.member.clone()],
                    mass_per_member: density_area.map(|da| da * c.length),
                }),
            }
        }
    }
    rows.sort_by(|a, b| {
        b.length
            .partial_cmp(&a.length)
            .expect("finite")
            .then(a.description.cmp(&b.description))
            .then(a.members.cmp(&b.members))
    });
    for (i, r) in rows.iter_mut().enumerate() {
        r.item = i as u32 + 1;
        r.members.sort();
    }
    rows
}
