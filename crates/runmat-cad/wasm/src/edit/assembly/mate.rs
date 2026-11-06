//! `assembly.mate`: a mate between two picks, each a component's face, edge or the whole
//! component, written as a mate connector per pick plus the mate.
//!
//! The standard mates map onto the connector mates the solver has:
//!
//! | Mate          | Picks                                   | Written as                          |
//! |---------------|-----------------------------------------|-------------------------------------|
//! | coincident    | two planar faces                        | `planar`                            |
//! |               | two axes (cylinders, circles, lines)    | `cylindrical`                       |
//! | concentric    | two axes                                | `cylindrical`                       |
//! | distance      | two planar faces                        | `distance` (plane to point) + `parallel` |
//! |               | anything else                           | `distance` (point to point)         |
//! | angle         | any two                                 | `angle` between the Z axes          |
//! | parallel      | any two                                 | `parallel`                          |
//! | perpendicular | any two                                 | `angle` between the Z axes, 90°     |
//! | tangent       | a planar face and a cylindrical face    | `tangent` (plane to cylinder)       |
//!
//! Joints (fasten, revolute, slider, cylindrical, planar, ball) are written as themselves.

use super::frames::{cross, dot, edge_frame, face_frame, norm, to_world, Frame, FrameKind, V3};
use super::{model, parse_pick, Compiled, Entity};
use crate::features::{picks_of, value};
use runmat_cad::assembly::model::{
    AnchorRef, AngleMeasure, AssemblyDocument, DistanceMeasure, MateConnector, MateDef,
    MateKind, OffsetExpr, RelPath, RotationExpr, TangentSpec, TargetExpr,
};
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

/// The `ratio` field as a plain number (a gear's teeth ratio), 1 when absent.
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
        return Err(model("a gear ratio must be non-zero"));
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

/// A pick resolved: the component's name and the frame on it (None for the whole component).
struct Resolved {
    occurrence: String,
    frame: Option<Frame>,
    /// The frame in assembly coordinates, where the component sits now.
    world: Option<Frame>,
}

fn resolve(compiled: &Compiled, id: &str, what: &str) -> Result<Resolved, EditFailure> {
    let pick = parse_pick(id).ok_or_else(|| model(format!("pick a face, edge or component for {what}")))?;
    let occurrence = compiled.occurrence_name(&pick.uuid)?;
    let Some(entity) = pick.entity else {
        return Ok(Resolved {
            occurrence,
            frame: None,
            world: None,
        });
    };
    let (prototype, transform) = compiled
        .placed(&pick.uuid)
        .ok_or_else(|| model(format!("`{occurrence}` isn't drawn, so its faces can't be mated")))?;
    let frame = match entity {
        Entity::Face { body, face } => compiled
            .body(prototype, body)
            .and_then(|b| b["faces"].as_array()?.iter().find(|f| f["index"].as_u64() == Some(face as u64)))
            .and_then(face_frame),
        Entity::Edge { body, edge } => compiled
            .body(prototype, body)
            .and_then(|b| b["edges"].as_array()?.iter().find(|e| e["index"].as_u64() == Some(edge as u64)))
            .and_then(edge_frame),
    }
    .ok_or_else(|| model(format!("the picked {what} on `{occurrence}` has no frame to mate to")))?;
    Ok(Resolved {
        occurrence,
        frame: Some(frame),
        world: Some(to_world(frame, &transform)),
    })
}

fn fresh(taken: impl Fn(&str) -> bool, stem: &str) -> String {
    (1..)
        .map(|n| format!("{stem}{n}"))
        .find(|name| !taken(name))
        .expect("an unused name")
}

/// Adds a connector on `pick` and answers its name.
fn connector(document: &mut AssemblyDocument, pick: &Resolved, flip: bool) -> String {
    let stem = format!("{}_c", pick.occurrence);
    let name = fresh(|n| document.connectors.iter().any(|c| c.name == n), &stem);
    let anchor = match pick.frame {
        None => AnchorRef::Origin,
        Some(frame) => AnchorRef::Explicit {
            offset: OffsetExpr {
                translation: frame.origin.map(|c| quantity(mm(c), "mm")),
                rotation: RotationExpr::Basis {
                    x: round_unit(frame.x),
                    z: round_unit(frame.z),
                },
            },
        },
    };
    document.connectors.push(MateConnector {
        name: name.clone(),
        occurrence: RelPath::new(&[&pick.occurrence]),
        anchor,
        offset: None,
        flip,
    });
    name
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
    let mut a = resolve(compiled, &first, "the first pick")?;
    let mut b = resolve(compiled, &second, "the second pick")?;
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
        "coincident" => return Err(model("a coincident mate joins two planar faces or two axes")),
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
        "revolute" => MateKind::Revolute { angle: None },
        "slider" => MateKind::Slider { distance: None },
        "cylindrical" => MateKind::Cylindrical {
            angle: None,
            distance: None,
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
            return Err(model(format!(
                "the `{other}` mate is written in the source editor in this build"
            )))
        }
    };
    // Turn the moving component into the mated orientation first, as feature-based CAD tools move a part
    // when a mate is added: the solve then only finishes the position, and starts where the
    // angle residual has a slope (parallel faces under a perpendicular mate have none).
    let (moving_frame, other_frame) = if moving == a.occurrence { (a.world, b.world) } else { (b.world, a.world) };
    if let (Some(m), Some(o)) = (moving_frame, other_frame) {
        let cos = dot(m.z, o.z).clamp(-1.0, 1.0);
        let nearest = if cos >= 0.0 { 0.0 } else { std::f64::consts::PI };
        let wanted = match kind.as_str() {
            "perpendicular" | "tangent" => Some(std::f64::consts::FRAC_PI_2),
            "angle" => literal_angle(values),
            "parallel" | "coincident" | "concentric" | "distance" => Some(nearest),
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
    let ca = connector(document, &a, false);
    let cb = connector(document, &b, flip);
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
    let fixed = if moving == a.occurrence { &b.occurrence } else { &a.occurrence };
    Ok(format!("Mated {moving} to {fixed} ({kind})"))
}
