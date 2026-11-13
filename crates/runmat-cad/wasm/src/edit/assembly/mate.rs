//! `assembly.mate`: a mate between two picks, each a component's face, edge, vertex, part
//! datum (plane, axis, point, frame or mate connector), a connector the assembly has, or the
//! whole component, written as a mate connector per pick plus the mate. Each connector anchors
//! to what the component's part publishes ([`super::anchors`]), so the mate follows the part's
//! edits.
//!
//! The standard mates map onto the connector mates the solver has:
//!
//! | Mate          | Picks                                   | Written as                          |
//! |---------------|-----------------------------------------|-------------------------------------|
//! | coincident    | two planar faces                        | `planar`                            |
//! |               | two axes (cylinders, circles, lines)    | `cylindrical`                       |
//! |               | two points (vertices, datum points)     | `ball`                              |
//! |               | a point and a plane                     | `distance` 0 (plane to point)       |
//! |               | two connectors or frames                | `fasten`                            |
//! | concentric    | two axes                                | `cylindrical`                       |
//! | distance      | two planar faces                        | `distance` (plane to point) + `parallel` |
//! |               | anything else                           | `distance` (point to point)         |
//! | angle         | any two                                 | `angle` between the Z axes          |
//! | parallel      | any two                                 | `parallel`                          |
//! | perpendicular | any two                                 | `angle` between the Z axes, 90°     |
//! | tangent       | a planar face and a cylindrical face    | `tangent` (plane to cylinder)       |
//!
//! Joints (fasten, revolute, slider, cylindrical, planar, ball) are written as themselves, a
//! revolute's or cylindrical's angle limits and a slider's or cylindrical's travel limits as
//! ranges. Pin slot, screw (`pitch` is the lead), gear (`ratio`) and rack and pinion (`pitch`
//! is the rack's travel per turn) too. The advanced mates ([`advanced`]): width (the first two
//! picks are a groove face and a tab face, `first2` and `second2` their other faces), symmetric
//! (`plane`), cam (`profile`, a closed path the first pick's part publishes), path (`path`, a
//! path the first pick's part publishes), linear coupler (`ratio`), hinge (angle limits) and
//! universal joint (`joint_point`).

use super::anchors::{Anchor, Publisher};
use super::frames::{
    cross, datum_frame, dot, edge_frame, face_frame, norm, to_world, vertex_frame, Frame,
    FrameKind, V3,
};
use super::mate_edits::{limits, ANGLE, TRAVEL};
use super::{model, parse_pick, Compiled, Entity};
use crate::features::{picks_of, value};
use runmat_cad::assembly::model::{
    AnchorRef, AngleMeasure, AssemblyDocument, CamProfile, DistanceMeasure, MateConnector, MateDef,
    MateKind, OffsetExpr, RelPath, RotationExpr, TangentSpec, TargetExpr,
};
use runmat_cad::ids::PublishName;
use runmat_cad::refs::TopoKind;
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

type Values = Map<String, Value>;

pub(super) fn quantity(value: f64, unit: &str) -> ValueExpr {
    ValueExpr::Quantity {
        value,
        unit: unit.to_owned(),
    }
}

/// A field's value as a value expression in `unit`.
pub(super) fn field(values: &Values, id: &str, unit: &str) -> Result<ValueExpr, EditFailure> {
    serde_json::from_value(value(values, id, unit)?)
        .map_err(|e| model(format!("`{id}` isn't a value: {e}")))
}

/// The `ratio` field as a plain number (a gear's teeth ratio or a coupler's), 1 when absent.
fn ratio(values: &Values) -> Result<ValueExpr, EditFailure> {
    let v = match values.get("ratio") {
        None | Some(Value::Null) => 1.0,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(1.0),
        Some(Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| model(format!("`ratio` isn't a number: {s}")))?,
        Some(other) => {
            return serde_json::from_value(value(values, "ratio", "")?)
                .map_err(|e| model(format!("`ratio` isn't a value: {e} ({other})")))
        }
    };
    if v == 0.0 {
        return Err(model("the ratio must be non-zero"));
    }
    Ok(ValueExpr::Number { value: v })
}

/// Micrometre rounding keeps written coordinates short (`12.5 mm`, not `12.500000000001 mm`).
fn mm(metres: f64) -> f64 {
    let v = (metres * 1e6).round() / 1e3;
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn round_unit(v: V3) -> [f64; 3] {
    v.map(|c| {
        let r = (c * 1e6).round() / 1e6;
        if r == 0.0 {
            0.0
        } else {
            r
        }
    })
}

/// A pick resolved: the component's name, the frame on it (None for the whole component) and
/// what its connector anchors to.
struct Resolved {
    occurrence: String,
    frame: Option<Frame>,
    /// The frame in assembly coordinates, where the component sits now.
    world: Option<Frame>,
    anchor: Anchor,
}

/// The entity `list` (`faces`, `edges`, `vertices`) of a body, by index.
fn entity<'a>(body: Option<&'a Value>, list: &str, index: usize) -> Option<&'a Value> {
    body?[list]
        .as_array()?
        .iter()
        .find(|e| e["index"].as_u64() == Some(index as u64))
}

fn resolve(
    document: &AssemblyDocument,
    compiled: &Compiled,
    id: &str,
    what: &str,
) -> Result<Resolved, EditFailure> {
    // A connector the assembly declares: reused as it is.
    if let Some(name) = id.strip_prefix("connector:") {
        let c = document
            .connector(name)
            .ok_or_else(|| model(format!("the assembly has no connector `{name}`")))?;
        let [segment] = c.occurrence.0.as_slice() else {
            return Err(model(format!(
                "connector `{name}` isn't on a component of this assembly"
            )));
        };
        let kind = Frame {
            kind: FrameKind::Frame,
            origin: [0.0; 3],
            z: [0.0, 0.0, 1.0],
            x: [1.0, 0.0, 0.0],
            radius: None,
        };
        return Ok(Resolved {
            occurrence: segment.name.clone(),
            frame: Some(kind),
            world: None,
            anchor: Anchor::Connector(name.to_owned()),
        });
    }
    let pick = parse_pick(id).ok_or_else(|| {
        model(format!(
            "pick a face, edge, vertex, datum or component for {what}"
        ))
    })?;
    let occurrence = compiled.occurrence_name(&pick.uuid)?;
    let Some(picked) = pick.entity else {
        return Ok(Resolved {
            occurrence,
            frame: None,
            world: None,
            anchor: Anchor::Origin,
        });
    };
    let (prototype, transform) = compiled.placed(&pick.uuid).ok_or_else(|| {
        model(format!(
            "`{occurrence}` isn't drawn, so its faces can't be mated"
        ))
    })?;
    let file = compiled.prototype_file(prototype);
    let (frame, named) = match &picked {
        Entity::Face { body, face } => {
            let e = entity(compiled.body(prototype, *body), "faces", *face);
            (e.and_then(face_frame), e.map(|e| (TopoKind::Face, e)))
        }
        Entity::Edge { body, edge } => {
            let e = entity(compiled.body(prototype, *body), "edges", *edge);
            (e.and_then(edge_frame), e.map(|e| (TopoKind::Edge, e)))
        }
        Entity::Vertex { body, vertex } => {
            let e = entity(compiled.body(prototype, *body), "vertices", *vertex);
            (e.and_then(vertex_frame), e.map(|e| (TopoKind::Vertex, e)))
        }
        Entity::Datum { feature } => (
            compiled.reference(prototype, feature).and_then(datum_frame),
            None,
        ),
    };
    let frame = frame.ok_or_else(|| {
        model(format!(
            "the picked {what} on `{occurrence}` has no frame to mate to"
        ))
    })?;
    let anchor = match (&picked, named, file) {
        (Entity::Datum { feature }, _, Some(file)) => Anchor::Datum {
            file,
            feature: feature.clone(),
        },
        (_, Some((kind, e)), Some(file)) if e["reference"].is_string() => Anchor::Entity {
            file,
            kind,
            reference: e["reference"].as_str().unwrap_or_default().to_owned(),
        },
        (Entity::Datum { .. }, _, None) => {
            return Err(model(format!(
                "`{occurrence}`'s part file isn't known, so its datum can't be published"
            )))
        }
        _ => Anchor::Frozen {
            origin: frame.origin,
            x: frame.x,
            z: frame.z,
        },
    };
    Ok(Resolved {
        occurrence,
        frame: Some(frame),
        world: Some(to_world(frame, &transform)),
        anchor,
    })
}

fn fresh(taken: impl Fn(&str) -> bool, stem: &str) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|name| !taken(name))
        .expect("an unused name")
}

/// Adds a connector on `pick` and answers its name. A face, edge, vertex or datum is published
/// from its part ([`Publisher`]) and the connector anchors to that name; a connector the
/// assembly has is reused (a flipped copy when `flip` asks for the other way).
fn connector(
    document: &mut AssemblyDocument,
    publisher: &mut Publisher,
    pick: &Resolved,
    flip: bool,
) -> Result<String, EditFailure> {
    if let Anchor::Connector(existing) = &pick.anchor {
        if !flip {
            return Ok(existing.clone());
        }
    }
    let stem = format!("{}_c", pick.occurrence);
    let name = fresh(|n| document.connectors.iter().any(|c| c.name == n), &stem);
    let frozen = |origin: [f64; 3], x: [f64; 3], z: [f64; 3]| AnchorRef::Explicit {
        offset: OffsetExpr {
            translation: origin.map(|c| quantity(mm(c), "mm")),
            rotation: RotationExpr::Basis {
                x: round_unit(x),
                z: round_unit(z),
            },
        },
    };
    let made = match &pick.anchor {
        Anchor::Connector(existing) => {
            let mut copy = document
                .connector(existing)
                .cloned()
                .ok_or_else(|| model(format!("the assembly has no connector `{existing}`")))?;
            copy.name = name.clone();
            copy.flip = !copy.flip;
            copy
        }
        anchor => {
            let anchor = match (anchor, publisher.name_of(anchor)?) {
                (_, Some(published)) => AnchorRef::Published { name: published },
                (Anchor::Frozen { origin, x, z }, None) => frozen(*origin, *x, *z),
                _ => AnchorRef::Origin,
            };
            MateConnector {
                name: name.clone(),
                occurrence: RelPath::new(&[&pick.occurrence]),
                anchor,
                offset: None,
                flip,
                at: None,
                z: None,
                x: None,
            }
        }
    };
    document.connectors.push(made);
    Ok(name)
}

/// The angle field as radians, when it is a plain number of degrees.
fn literal_angle(values: &Values) -> Option<f64> {
    match field(values, "angle", "deg").ok()? {
        ValueExpr::Quantity { value, unit } if unit == "deg" => Some(value.to_radians()),
        ValueExpr::Quantity { value, unit } if unit == "rad" => Some(value),
        _ => None,
    }
}

fn kind_of(frame: &Option<Frame>) -> Option<FrameKind> {
    frame.map(|f| f.kind)
}

pub(super) fn mate(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
) -> Result<String, EditFailure> {
    let kind = values
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("coincident")
        .to_owned();
    let first = picks_of(values, "first")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the first face, edge or component"))?;
    let second = picks_of(values, "second")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the second face, edge or component"))?;
    let mut a = resolve(document, compiled, &first, "the first pick")?;
    let mut b = resolve(document, compiled, &second, "the second pick")?;
    if a.occurrence == b.occurrence {
        return Err(model("pick faces or edges on two different components"));
    }
    // The second pick's component moves and the first stays, unless the second is the
    // assembly's first component, which is fixed as the ground.
    let ground = document.occurrences.first().map(|o| o.name.clone());
    let moving = if ground.as_deref() == Some(b.occurrence.as_str()) {
        a.occurrence.clone()
    } else {
        b.occurrence.clone()
    };
    let flip = values.get("flip").and_then(Value::as_bool).unwrap_or(false);
    let exact = |v: ValueExpr| TargetExpr::Exact { value: v };
    let both = |k: FrameKind| kind_of(&a.frame) == Some(k) && kind_of(&b.frame) == Some(k);
    let mut extra = Vec::new();
    let mate_kind = match kind.as_str() {
        "coincident" if both(FrameKind::Plane) => MateKind::Planar,
        "coincident" if both(FrameKind::Axis) => MateKind::Cylindrical {
            angle: None,
            distance: None,
        },
        "coincident" if both(FrameKind::Point) => MateKind::Ball,
        "coincident" if both(FrameKind::Frame) => MateKind::Fasten,
        // A point on a plane: the plane's connector first, its distance to the point zero.
        "coincident" if matches!((kind_of(&a.frame), kind_of(&b.frame)), (Some(FrameKind::Plane), Some(FrameKind::Point)) | (Some(FrameKind::Point), Some(FrameKind::Plane))) => {
            if kind_of(&a.frame) == Some(FrameKind::Point) {
                std::mem::swap(&mut a, &mut b);
            }
            MateKind::Distance {
                measure: DistanceMeasure::PlanePoint,
                target: exact(quantity(0.0, "mm")),
            }
        }
        "coincident" => {
            return Err(model(
                "a coincident mate joins two planes, two axes, two points, a point and a plane, or two connectors",
            ))
        }
        "concentric" if both(FrameKind::Axis) => MateKind::Cylindrical {
            angle: None,
            distance: None,
        },
        "concentric" => {
            return Err(model(
                "a concentric mate joins two cylindrical faces or circular edges",
            ))
        }
        "distance" if both(FrameKind::Plane) => {
            extra.push(MateKind::Parallel);
            MateKind::Distance {
                measure: DistanceMeasure::PlanePoint,
                target: exact(field(values, "offset", "mm")?),
            }
        }
        "distance" => MateKind::Distance {
            measure: DistanceMeasure::PointPoint,
            target: exact(field(values, "offset", "mm")?),
        },
        "angle" => MateKind::Angle {
            measure: AngleMeasure::BetweenZ,
            target: exact(field(values, "angle", "deg")?),
        },
        "parallel" => MateKind::Parallel,
        "perpendicular" => MateKind::Angle {
            measure: AngleMeasure::BetweenZ,
            target: exact(quantity(90.0, "deg")),
        },
        "tangent" => {
            // The plane's connector comes first, then the cylinder's axis.
            if kind_of(&a.frame) == Some(FrameKind::Axis) {
                std::mem::swap(&mut a, &mut b);
            }
            let radius = match (kind_of(&a.frame), b.frame) {
                (Some(FrameKind::Plane), Some(Frame { kind: FrameKind::Axis, radius: Some(r), .. })) => r,
                _ => return Err(model("a tangent mate joins a planar face and a cylindrical face")),
            };
            MateKind::Tangent {
                kind: TangentSpec::PlaneCylinder {
                    radius: quantity(mm(radius), "mm"),
                },
            }
        }
        "fasten" => MateKind::Fasten,
        "revolute" => MateKind::Revolute {
            angle: limits(values, "angle_limits", ANGLE, "deg")?,
        },
        "slider" => MateKind::Slider {
            distance: limits(values, "distance_limits", TRAVEL, "mm")?,
        },
        "cylindrical" => MateKind::Cylindrical {
            angle: limits(values, "angle_limits", ANGLE, "deg")?,
            distance: limits(values, "distance_limits", TRAVEL, "mm")?,
        },
        "planar" => MateKind::Planar,
        "ball" => MateKind::Ball,
        // The pin's connector first, then the slot's (the slot runs along its X).
        "pin_slot" => MateKind::PinSlot,
        "screw" => MateKind::Screw {
            lead: field(values, "pitch", "mm")?,
        },
        "gear" => MateKind::Gear {
            ratio: ratio(values)?,
        },
        // `pitch` is the rack's travel per turn of the pinion.
        "rack_pinion" => MateKind::RackPinion {
            travel: field(values, "pitch", "mm")?,
        },
        other => {
            if other == "width" {
                // The tab's other face is on the tab, the groove's on the groove's component.
                for (id, on, what) in [("first2", &a.occurrence, "groove"), ("second2", &b.occurrence, "tab")] {
                    if let Some(p) = picks_of(values, id).into_iter().next() {
                        if &resolve(document, compiled, &p, "the other face")?.occurrence != on {
                            return Err(model(format!("pick the {what}'s other face on `{on}`")));
                        }
                    }
                }
            }
            advanced(document, compiled, publisher, values, other, None)?
                .ok_or_else(|| model(format!("there is no `{other}` mate")))?
        }
    };
    // Turn the moving component into the mated orientation first, as feature-based CAD tools move a part
    // when a mate is added: the solve then only finishes the position, and starts where the
    // angle residual has a slope (parallel faces under a perpendicular mate have none).
    let (moving_frame, other_frame) = if moving == a.occurrence {
        (a.world, b.world)
    } else {
        (b.world, a.world)
    };
    if let (Some(m), Some(o)) = (moving_frame, other_frame) {
        let cos = dot(m.z, o.z).clamp(-1.0, 1.0);
        // Planes and axes mate either way round; Flip takes the other way from the nearest.
        let nearest = if (cos >= 0.0) != flip {
            0.0
        } else {
            std::f64::consts::PI
        };
        let wanted = match kind.as_str() {
            "perpendicular" | "tangent" => Some(std::f64::consts::FRAC_PI_2),
            "angle" => literal_angle(values),
            "parallel" | "coincident" | "concentric" | "distance" | "width" => Some(nearest),
            _ => None,
        };
        if let Some(theta) = wanted {
            let now = cos.acos();
            let axis = cross(m.z, o.z);
            if norm(axis) < 1e-9 {
                if (theta - now).abs() > 1e-9 {
                    super::place::turn(document, &moving, m.x, theta - now)?;
                }
            } else if (now - theta).abs() > 1e-9 {
                super::place::turn(document, &moving, axis, now - theta)?;
            }
        }
    }
    let ca = connector(document, publisher, &a, false)?;
    let cb = connector(document, publisher, &b, flip)?;
    let name = fresh(|n| document.mates.iter().any(|m| m.name == n), &kind);
    for (i, k) in std::iter::once(mate_kind).chain(extra).enumerate() {
        let mate_name = if i == 0 {
            name.clone()
        } else {
            format!("{name}_{}", k.name())
        };
        document.mates.push(MateDef {
            name: mate_name,
            kind: k,
            a: ca.clone(),
            b: cb.clone(),
            when: None,
            suppressed: false,
        });
    }
    // The moving component floats so the mate can place it.
    if let Some(occurrence) = document.occurrences.iter_mut().find(|o| o.name == moving) {
        occurrence.fixed = false;
    }
    let fixed = if moving == a.occurrence {
        &b.occurrence
    } else {
        &a.occurrence
    };
    Ok(format!("Mated {moving} to {fixed} ({kind})"))
}

/// A pick of field `id` as a new connector, or `None` when the field is empty.
fn extra_connector(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
    id: &str,
    what: &str,
) -> Result<Option<String>, EditFailure> {
    match picks_of(values, id).into_iter().next() {
        Some(p) => {
            let pick = resolve(document, compiled, &p, what)?;
            Ok(Some(connector(document, publisher, &pick, false)?))
        }
        None => Ok(None),
    }
}

/// A published name from text field `id`.
fn published(values: &Values, id: &str, what: &str) -> Result<PublishName, EditFailure> {
    match values.get(id).and_then(Value::as_str).map(str::trim) {
        Some(name) if !name.is_empty() => Ok(PublishName::new(name)),
        _ => Err(model(format!("name {what}"))),
    }
}

/// The advanced mates from the form's values, on the mate's own two connectors: width's other
/// faces and symmetric's plane become new connectors in `document`. `keep` is the mate being
/// edited: a width or symmetric mate keeps the connectors it has where the form picks none.
/// `None` when `kind` isn't an advanced mate.
pub(super) fn advanced(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    publisher: &mut Publisher,
    values: &Values,
    kind: &str,
    keep: Option<&MateKind>,
) -> Result<Option<MateKind>, EditFailure> {
    Ok(Some(match kind {
        "width" => {
            let (old_a2, old_b2) = match keep {
                Some(MateKind::Width { a2, b2 }) => (Some(a2.clone()), Some(b2.clone())),
                _ => (None, None),
            };
            let a2 = extra_connector(
                document,
                compiled,
                publisher,
                values,
                "first2",
                "the groove's other face",
            )?
            .or(old_a2);
            let b2 = extra_connector(
                document,
                compiled,
                publisher,
                values,
                "second2",
                "the tab's other face",
            )?
            .or(old_b2);
            MateKind::Width {
                a2: a2.ok_or_else(|| model("pick the groove's other face"))?,
                b2: b2.ok_or_else(|| model("pick the tab's other face"))?,
            }
        }
        "symmetric" => {
            let old = match keep {
                Some(MateKind::Symmetric { plane }) => Some(plane.clone()),
                _ => None,
            };
            let plane = extra_connector(
                document,
                compiled,
                publisher,
                values,
                "plane",
                "the symmetry plane",
            )?
            .or(old);
            MateKind::Symmetric {
                plane: plane.ok_or_else(|| model("pick the symmetry plane"))?,
            }
        }
        "cam" => MateKind::Cam {
            profile: CamProfile {
                profile: published(
                    values,
                    "profile",
                    "the closed path the cam's part publishes",
                )?,
            },
        },
        "path" => MateKind::Path {
            path: published(
                values,
                "path",
                "the path the first component's part publishes",
            )?,
        },
        "linear_coupler" => MateKind::LinearCoupler {
            ratio: ratio(values)?,
        },
        "hinge" => MateKind::Hinge {
            angle: limits(values, "angle_limits", ANGLE, "deg")?,
        },
        "universal_joint" => MateKind::UniversalJoint {
            joint_point: values
                .get("joint_point")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        },
        _ => return Ok(None),
    }))
}
