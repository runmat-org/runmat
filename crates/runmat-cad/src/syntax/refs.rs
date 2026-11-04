//! References, datums and sketch points in design-language text.
//!
//! | Text | Model |
//! |---|---|
//! | `feature.body.body` | the body a feature made |
//! | `feature.body.face.end`, `feature.body.face.start` | a face by its generation name |
//! | `feature.body.face.side(s1.l1)` | a face from the curve `l1` of sketch `s1` |
//! | `feature.h.face.wall(a)`, `feature.cut.face.split_face(side = positive)` | a face by key or side |
//! | `feature.body.face.side(s1.l1, ordinal = 2)` | one piece of a split face |
//! | `feature.step.face("Top")` | a face named in an imported CAD file |
//! | `edges_between(a, b)`, `vertex_of(a, b, c)`, `piece(a, bounded_by = [b])` | relational references |
//! | `origin.xy`, `origin.yz`, `origin.zx`, `plane.mid` | a plane |
//! | `origin.x`, `origin.y`, `origin.z`, `axis.hinge` | an axis |
//! | `origin.point`, `point.p1`, `point(10 mm, 0 mm, 5 mm)`, `point(vertex_of(…))` | a point |
//! | `sketch.s1.l1.end` | a sketch point |
//! | `sketch.s1` | a sketch's closed loops (a profile) |
//!
//! Every form reads back as the value it was printed from. A value with no form here (a name
//! held by token, a geometric fallback) is written in the generic record form, which also
//! reads back exactly. Hashes are never written.

use crate::ids::{FeatureId, SketchEntityId};
use crate::model::datum::{AxisRef, DatumRef, OriginAxis, OriginPlane, PlaneRef, PointRef};
use crate::model::sketch::PointRole;
use crate::refs::name::{EntityName, GenerationName, PlaneSide, Role, SourceTag, TopoKind};
use crate::refs::Reference;
use runmat_design_syntax::ast::{Argument, Expr};
use runmat_design_syntax::json::{is_identifier, quote, Json, LowerError};

/// Function names the reference forms take; a dynamic value calling one of these is written
/// in the record form.
pub const REFERENCE_FUNCTIONS: &[&str] = &["edges_between", "vertex_of", "piece", "point"];

/// Path heads the datum, sketch and reference forms take.
pub const RESERVED_HEADS: &[&str] = &["origin", "plane", "axis", "point", "sketch"];

const TOPOLOGY: &[&str] = &["body", "face", "edge", "vertex"];

/// Whether a dotted path reads as one of the forms here (so a value reference with this path
/// must be written another way).
pub fn is_reserved_path(path: &[String]) -> bool {
    match path {
        [head, ..] if RESERVED_HEADS.contains(&head.as_str()) => true,
        [head, _, kind, ..] if head == "feature" => TOPOLOGY.contains(&kind.as_str()),
        _ => false,
    }
}

// Every role, for reading role words back. The match below fails to compile when the kernel
// adds a role, so the list stays complete.
const ROLES: &[Role] = &[
    Role::Side,
    Role::StartCap,
    Role::EndCap,
    Role::Blend,
    Role::Chamfer,
    Role::ShellInner,
    Role::HoleWall,
    Role::HoleBottom,
    Role::HoleCounterbore,
    Role::HoleCounterboreWall,
    Role::HoleCountersink,
    Role::ShellRim,
    Role::ShellOuter,
    Role::ThinWall,
    Role::DraftStep,
    Role::SplitFace,
    Role::Body,
    Role::Replacement,
    Role::Boundary,
    Role::SurfacePatch,
    Role::OffsetFace,
    Role::Extension,
    Role::ThickenFace,
    Role::ThickenRim,
    Role::MidSurface,
    Role::Ruled,
    Role::SheetFace,
    Role::SheetEdge,
    Role::BendFace,
    Role::Relief,
    Role::FormFace,
    Role::MemberSide,
    Role::MemberEnd,
    Role::CapPlate,
    Role::Gusset,
    Role::WeldBead,
    Role::PartingFace,
    Role::ThreadFace,
    Role::EmbossTop,
    Role::Rib,
];

#[allow(dead_code)]
fn every_role_is_listed(role: Role) {
    match role {
        Role::Side
        | Role::StartCap
        | Role::EndCap
        | Role::Blend
        | Role::Chamfer
        | Role::ShellInner
        | Role::HoleWall
        | Role::HoleBottom
        | Role::HoleCounterbore
        | Role::HoleCounterboreWall
        | Role::HoleCountersink
        | Role::ShellRim
        | Role::ShellOuter
        | Role::ThinWall
        | Role::DraftStep
        | Role::SplitFace
        | Role::Body
        | Role::Replacement
        | Role::Boundary
        | Role::SurfacePatch
        | Role::OffsetFace
        | Role::Extension
        | Role::ThickenFace
        | Role::ThickenRim
        | Role::MidSurface
        | Role::Ruled
        | Role::SheetFace
        | Role::SheetEdge
        | Role::BendFace
        | Role::Relief
        | Role::FormFace
        | Role::MemberSide
        | Role::MemberEnd
        | Role::CapPlate
        | Role::Gusset
        | Role::WeldBead
        | Role::PartingFace
        | Role::ThreadFace
        | Role::EmbossTop
        | Role::Rib => {}
    }
}

fn role_named(word: &str) -> Option<Role> {
    ROLES.iter().copied().find(|role| role.as_str() == word)
}

fn topo_named(word: &str) -> Option<TopoKind> {
    Some(match word {
        "body" => TopoKind::Body,
        "face" => TopoKind::Face,
        "edge" => TopoKind::Edge,
        "vertex" => TopoKind::Vertex,
        _ => return None,
    })
}

fn word(text: &str) -> Option<String> {
    is_identifier(text).then(|| text.to_string())
}

// ----------------------------------------------------------------------------- printing

/// A reference's text, or `None` when it has no text form.
pub fn print_reference(reference: &Reference) -> Option<String> {
    Some(match reference {
        Reference::Name { kind, name } => print_name(*kind, name)?,
        Reference::EdgesBetween { a, b } => {
            format!(
                "edges_between({}, {})",
                print_reference(a)?,
                print_reference(b)?
            )
        }
        Reference::VertexOf { of } if !of.is_empty() => {
            let items: Option<Vec<String>> = of.iter().map(print_reference).collect();
            format!("vertex_of({})", items?.join(", "))
        }
        Reference::Piece {
            base,
            bounded_by,
            not_bounded_by,
        } => {
            let mut parts = vec![print_reference(base)?];
            for (key, list) in [
                ("bounded_by", bounded_by),
                ("not_bounded_by", not_bounded_by),
            ] {
                if !list.is_empty() {
                    let items: Option<Vec<String>> = list.iter().map(print_reference).collect();
                    parts.push(format!("{key} = [{}]", items?.join(", ")));
                }
            }
            format!("piece({})", parts.join(", "))
        }
        _ => return None,
    })
}

fn print_name(kind: TopoKind, name: &EntityName) -> Option<String> {
    match name {
        EntityName::Generated(GenerationName {
            feature,
            role,
            tag,
            ordinal,
        }) => {
            let feature = word(&feature.0)?;
            if kind == TopoKind::Body
                && *role == Role::Body
                && *tag == SourceTag::None
                && ordinal.is_none()
            {
                return Some(format!("feature.{feature}.body"));
            }
            let mut arguments = Vec::new();
            match tag {
                SourceTag::None => {}
                SourceTag::Curve { sketch, entity } => {
                    arguments.push(format!("{}.{}", word(&sketch.0)?, word(&entity.0)?))
                }
                SourceTag::Key { key } => arguments.push(match word(key) {
                    Some(key) => key,
                    None => quote(key),
                }),
                SourceTag::Side { side } => arguments.push(format!(
                    "side = {}",
                    match side {
                        PlaneSide::Negative => "negative",
                        PlaneSide::Positive => "positive",
                    }
                )),
                // Names held by token have no text form.
                _ => return None,
            }
            if let Some(ordinal) = ordinal {
                arguments.push(format!("ordinal = {ordinal}"));
            }
            let head = format!("feature.{feature}.{kind}.{}", role.as_str());
            Some(if arguments.is_empty() {
                head
            } else {
                format!("{head}({})", arguments.join(", "))
            })
        }
        EntityName::CadFile { import, name } => Some(format!(
            "feature.{}.{kind}({})",
            word(&import.0)?,
            quote(name)
        )),
        EntityName::Copy { .. } => None,
    }
}

fn origin_plane(plane: OriginPlane) -> &'static str {
    match plane {
        OriginPlane::Xy => "xy",
        OriginPlane::Yz => "yz",
        OriginPlane::Zx => "zx",
    }
}

fn origin_axis(axis: OriginAxis) -> &'static str {
    match axis {
        OriginAxis::X => "x",
        OriginAxis::Y => "y",
        OriginAxis::Z => "z",
    }
}

/// A plane's text: `origin.xy`, `plane.mid`. A face is written by the field that takes it.
pub fn print_plane(plane: &PlaneRef) -> Option<String> {
    match plane {
        PlaneRef::Datum {
            datum: DatumRef::OriginPlane { plane },
        } => Some(format!("origin.{}", origin_plane(*plane))),
        PlaneRef::Datum {
            datum: DatumRef::Feature { feature },
        } => Some(format!("plane.{}", word(&feature.0)?)),
        _ => None,
    }
}

/// A plane in a field that takes faces too: a face is its reference.
pub fn print_plane_or_face(plane: &PlaneRef) -> Option<String> {
    match plane {
        PlaneRef::Face { face } => print_reference(face),
        other => print_plane(other),
    }
}

pub fn print_axis(axis: &AxisRef) -> Option<String> {
    match axis {
        AxisRef::Datum {
            datum: DatumRef::OriginAxis { axis },
        } => Some(format!("origin.{}", origin_axis(*axis))),
        AxisRef::Datum {
            datum: DatumRef::Feature { feature },
        } => Some(format!("axis.{}", word(&feature.0)?)),
        _ => None,
    }
}

fn role_word(role: PointRole) -> &'static str {
    match role {
        PointRole::At => "at",
        PointRole::Start => "start",
        PointRole::End => "end",
        PointRole::Center => "center",
    }
}

pub fn role_of(word: &str) -> Option<PointRole> {
    Some(match word {
        "at" => PointRole::At,
        "start" => PointRole::Start,
        "end" => PointRole::End,
        "center" => PointRole::Center,
        _ => return None,
    })
}

/// A point's text; `values` prints the coordinates.
pub fn print_point(
    point: &PointRef,
    values: &dyn Fn(&crate::value::ValueExpr) -> Option<String>,
) -> Option<String> {
    Some(match point {
        PointRef::Datum {
            datum: DatumRef::OriginPoint,
        } => "origin.point".into(),
        PointRef::Datum {
            datum: DatumRef::Feature { feature },
        } => format!("point.{}", word(&feature.0)?),
        PointRef::Coords { x, y, z } => {
            format!("point({}, {}, {})", values(x)?, values(y)?, values(z)?)
        }
        PointRef::Vertex { vertex } => format!("point({})", print_reference(vertex)?),
        PointRef::Sketch {
            sketch,
            entity,
            role,
        } => format!(
            "sketch.{}.{}.{}",
            word(&sketch.0)?,
            word(&entity.0)?,
            role_word(*role)
        ),
        _ => return None,
    })
}

// ----------------------------------------------------------------------------- reading

fn error(expr: &Expr, message: impl Into<String>) -> LowerError {
    LowerError::new(expr.span(), message)
}

fn segments(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Reference(reference) => Some(reference.segments()),
        _ => None,
    }
}

/// Reads a reference form; `None` when `expr` is not written as one.
pub fn lower_reference(expr: &Expr) -> Option<Result<Reference, LowerError>> {
    match expr {
        Expr::Reference(reference) => {
            let path = reference.segments();
            match path.as_slice() {
                [head, feature, kind] if head == "feature" && kind == "body" => {
                    Some(Ok(Reference::feature_body(feature)))
                }
                [head, feature, kind, role] if head == "feature" => {
                    let kind = topo_named(kind)?;
                    Some(generated(expr, feature, kind, role, Vec::new()))
                }
                _ => None,
            }
        }
        Expr::Call(call) => {
            let callee = segments(&call.callee()?)?;
            let arguments = call.arguments();
            match callee.as_slice() {
                [head, feature, kind, role] if head == "feature" => {
                    let kind = topo_named(kind)?;
                    Some(generated(expr, feature, kind, role, arguments))
                }
                [head, import, kind] if head == "feature" => {
                    let kind = topo_named(kind)?;
                    Some(match arguments.as_slice() {
                        [Argument::Positional(Expr::String(name))] => Ok(Reference::Name {
                            kind,
                            name: EntityName::CadFile {
                                import: FeatureId::new(import),
                                name: name.value(),
                            },
                        }),
                        _ => Err(error(
                            expr,
                            "a CAD-file name is written `feature.<import>.face(\"<name>\")`",
                        )),
                    })
                }
                [function] if function == "edges_between" => {
                    Some((|| match positional(expr, &arguments)?.as_slice() {
                        [a, b] => Ok(Reference::edges_between(required(a)?, required(b)?)),
                        _ => Err(error(expr, "`edges_between` takes two references")),
                    })())
                }
                [function] if function == "vertex_of" => Some((|| {
                    let items = positional(expr, &arguments)?;
                    if items.is_empty() {
                        return Err(error(expr, "`vertex_of` takes one or more references"));
                    }
                    Ok(Reference::VertexOf {
                        of: items.iter().map(required).collect::<Result<_, _>>()?,
                    })
                })()),
                [function] if function == "piece" => Some(piece(expr, &arguments)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn required(expr: &Expr) -> Result<Reference, LowerError> {
    lower_reference(expr).unwrap_or_else(|| Err(error(expr, "expected a reference here")))
}

fn positional(expr: &Expr, arguments: &[Argument]) -> Result<Vec<Expr>, LowerError> {
    arguments
        .iter()
        .map(|argument| match argument {
            Argument::Positional(value) => Ok(value.clone()),
            Argument::Named(_) => Err(error(expr, "this form takes no named arguments")),
        })
        .collect()
}

fn reference_list(expr: &Expr) -> Result<Vec<Reference>, LowerError> {
    match expr {
        Expr::List(list) => list.items().iter().map(required).collect(),
        other => Err(error(other, "expected a list of references")),
    }
}

fn piece(expr: &Expr, arguments: &[Argument]) -> Result<Reference, LowerError> {
    let mut base = None;
    let (mut bounded_by, mut not_bounded_by) = (Vec::new(), Vec::new());
    for argument in arguments {
        match argument {
            Argument::Positional(value) if base.is_none() => base = Some(required(value)?),
            Argument::Named(named) => {
                let name = named.name().map(|n| n.text()).unwrap_or_default();
                let value = named
                    .value()
                    .ok_or_else(|| error(expr, format!("`{name}` needs a value")))?;
                match name.as_str() {
                    "bounded_by" => bounded_by = reference_list(&value)?,
                    "not_bounded_by" => not_bounded_by = reference_list(&value)?,
                    _ => return Err(error(expr, format!("`piece` takes no `{name}`"))),
                }
            }
            Argument::Positional(_) => {
                return Err(error(expr, "`piece` takes one base reference"));
            }
        }
    }
    let base = base.ok_or_else(|| error(expr, "`piece` needs a base reference"))?;
    Ok(Reference::Piece {
        base: Box::new(base),
        bounded_by,
        not_bounded_by,
    })
}

fn generated(
    expr: &Expr,
    feature: &str,
    kind: TopoKind,
    role: &str,
    arguments: Vec<Argument>,
) -> Result<Reference, LowerError> {
    let role = role_named(role).ok_or_else(|| error(expr, format!("`{role}` is not a role")))?;
    let mut tag = SourceTag::None;
    let mut ordinal = None;
    for argument in arguments {
        match argument {
            Argument::Positional(value) if tag == SourceTag::None => {
                tag = match &value {
                    Expr::String(text) => SourceTag::Key { key: text.value() },
                    Expr::Reference(reference) => match reference.segments().as_slice() {
                        [key] => SourceTag::Key { key: key.clone() },
                        [sketch, entity] => SourceTag::Curve {
                            sketch: FeatureId::new(sketch),
                            entity: SketchEntityId::new(entity),
                        },
                        _ => return Err(error(&value, "expected `<sketch>.<curve>` or a key")),
                    },
                    _ => return Err(error(&value, "expected `<sketch>.<curve>` or a key")),
                }
            }
            Argument::Named(named) => {
                let name = named.name().map(|n| n.text()).unwrap_or_default();
                let value = named
                    .value()
                    .ok_or_else(|| error(expr, format!("`{name}` needs a value")))?;
                match (name.as_str(), &value) {
                    ("ordinal", Expr::Number(number)) => {
                        ordinal = Some(
                            number
                                .text()
                                .parse::<u32>()
                                .map_err(|_| error(&value, "an ordinal is a whole number"))?,
                        )
                    }
                    ("side", Expr::Reference(side)) if tag == SourceTag::None => {
                        tag = SourceTag::Side {
                            side: match side.segments().as_slice() {
                                [s] if s == "negative" => PlaneSide::Negative,
                                [s] if s == "positive" => PlaneSide::Positive,
                                _ => {
                                    return Err(error(&value, "a side is `negative` or `positive`"))
                                }
                            },
                        }
                    }
                    _ => return Err(error(&value, format!("unexpected `{name}` here"))),
                }
            }
            Argument::Positional(value) => {
                return Err(error(&value, "a name takes one curve or key"));
            }
        }
    }
    Ok(Reference::Name {
        kind,
        name: EntityName::Generated(GenerationName {
            feature: FeatureId::new(feature),
            role,
            tag,
            ordinal,
        }),
    })
}

fn datum_feature(name: &str) -> DatumRef {
    DatumRef::Feature {
        feature: FeatureId::new(name),
    }
}

/// Reads a datum, profile or point form as its model JSON; `None` when `expr` is not one.
pub fn lower_datum(
    expr: &Expr,
    values: &dyn Fn(&Expr) -> Result<crate::value::ValueExpr, LowerError>,
) -> Option<Result<Json, LowerError>> {
    let encode = |value: Result<Json, LowerError>| value;
    let json = |value: &dyn erased::Encode| -> Result<Json, LowerError> {
        value.encode().map_err(|message| error(expr, message))
    };
    let path = match expr {
        Expr::Reference(reference) => reference.segments(),
        Expr::Call(call) => {
            let callee = segments(&call.callee()?)?;
            if callee != ["point"] {
                return None;
            }
            let arguments = call.arguments();
            return Some(encode((|| match positional(expr, &arguments)?.as_slice() {
                [x, y, z] => json(&PointRef::Coords {
                    x: values(x)?,
                    y: values(y)?,
                    z: values(z)?,
                }),
                [vertex] => json(&PointRef::Vertex {
                    vertex: required(vertex)?,
                }),
                _ => Err(error(
                    expr,
                    "a point is `point(x, y, z)` or `point(<vertex reference>)`",
                )),
            })()));
        }
        _ => return None,
    };
    let result = match path.as_slice() {
        [origin, which] if origin == "origin" => match which.as_str() {
            "xy" | "yz" | "zx" => json(&PlaneRef::Datum {
                datum: DatumRef::OriginPlane {
                    plane: match which.as_str() {
                        "xy" => OriginPlane::Xy,
                        "yz" => OriginPlane::Yz,
                        _ => OriginPlane::Zx,
                    },
                },
            }),
            "x" | "y" | "z" => json(&AxisRef::Datum {
                datum: DatumRef::OriginAxis {
                    axis: match which.as_str() {
                        "x" => OriginAxis::X,
                        "y" => OriginAxis::Y,
                        _ => OriginAxis::Z,
                    },
                },
            }),
            "point" => json(&PointRef::Datum {
                datum: DatumRef::OriginPoint,
            }),
            _ => Err(error(expr, format!("`origin.{which}` is not a datum"))),
        },
        [kind, name] if kind == "plane" => json(&PlaneRef::Datum {
            datum: datum_feature(name),
        }),
        [kind, name] if kind == "axis" => json(&AxisRef::Datum {
            datum: datum_feature(name),
        }),
        [kind, name] if kind == "point" => json(&PointRef::Datum {
            datum: datum_feature(name),
        }),
        [kind, name] if kind == "sketch" => Ok(Json::Object(vec![(
            "sketch".into(),
            Json::String(name.clone()),
        )])),
        [kind, sketch, entity, role] if kind == "sketch" => match role_of(role) {
            Some(role) => json(&PointRef::Sketch {
                sketch: FeatureId::new(sketch),
                entity: SketchEntityId::new(entity),
                role,
            }),
            None => Err(error(expr, format!("`{role}` is not a sketch point"))),
        },
        [head, ..] if RESERVED_HEADS.contains(&head.as_str()) => Err(error(
            expr,
            format!("`{}` is not a datum, sketch or point", path.join(".")),
        )),
        _ => return None,
    };
    Some(result)
}

/// Serialising a model value to JSON, behind one object-safe call.
mod erased {
    use runmat_design_syntax::json::Json;

    pub trait Encode {
        fn encode(&self) -> Result<Json, String>;
    }

    impl<T: serde::Serialize> Encode for T {
        fn encode(&self) -> Result<Json, String> {
            Json::from_serialize(self).map_err(|e| e.to_string())
        }
    }
}
