//! Mate commands on mates that exist, and the smart mate.
//!
//! - `assembly.flip-mate`: turns each selected mate's second connector over (its `flip`), so
//!   the mated faces or axes face the other way; the floating component turns half a turn
//!   about the connector's X axis, since planes and axes solve from where they start.
//! - `assembly.edit-mate`: writes the selected mate's type, distance or angle target, limits,
//!   ratio, pitch, radius, profile, path and alignment from Edit mate's form, on the connectors
//!   the mate already has; a width's other faces and a symmetric mate's plane are picked anew
//!   or kept.
//!   The form starts from the mate's current values (`mate_form_values` in the index).
//! - `assembly.smart-mate`: a component dropped onto a face or axis of another. A planar target
//!   takes the component's planar face that faces it most squarely (nearest first) in a
//!   coincident mate; an axis takes the component's axis whose radius matches best (nearest
//!   first) in a concentric mate. The mate itself is `assembly.mate`'s.
//!
//! Mates are picked by their Outline IDs, `mate:<assembly>/<name>`; only mates of the open
//! assembly change here.

use super::frames::{dot, edge_frame, face_frame, norm, sub, to_world, Frame, FrameKind};
use super::mate::field;
use super::{model, parse_pick, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::{
    AnchorRef, AngleMeasure, AssemblyDocument, DistanceMeasure, MateKind, OffsetExpr, RotationExpr,
    TangentSpec, TargetExpr,
};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::value::ValueExpr;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;

/// The names of the open assembly's mates in `ids` (`mate:<assembly>/<name>`).
fn mate_names(ids: &[String]) -> Result<Vec<String>, EditFailure> {
    let mut names = Vec::new();
    for id in ids {
        let Some((assembly, name)) = id.strip_prefix("mate:").and_then(|r| r.rsplit_once('/'))
        else {
            continue;
        };
        if !assembly.is_empty() && assembly != "/" {
            return Err(model(format!(
                "`{name}` belongs to the sub-assembly `{assembly}`; open it to change the mate"
            )));
        }
        if !names.iter().any(|n| n == name) {
            names.push(name.to_owned());
        }
    }
    if names.is_empty() {
        return Err(model("select a mate first"));
    }
    Ok(names)
}

fn mate_at(document: &AssemblyDocument, name: &str) -> Result<usize, EditFailure> {
    document
        .mates
        .iter()
        .position(|m| m.name == name)
        .ok_or_else(|| model(format!("the assembly has no mate `{name}`")))
}

/// Sets (or toggles, when `to` is None) the flip of mate `at`'s second connector.
fn set_flip(
    document: &mut AssemblyDocument,
    at: usize,
    to: Option<bool>,
) -> Result<bool, EditFailure> {
    let b = document.mates[at].b.clone();
    let connector = document
        .connectors
        .iter_mut()
        .find(|c| c.name == b)
        .ok_or_else(|| {
            model(format!(
                "`{}` names no connector `{b}` of this assembly",
                document.mates[at].name
            ))
        })?;
    connector.flip = to.unwrap_or(!connector.flip);
    Ok(connector.flip)
}

/// The X axis of connector `name` in assembly coordinates, and the floating component it sits
/// on, when it has an explicit or origin anchor on a top-level component.
fn connector_x(
    document: &AssemblyDocument,
    compiled: &Compiled,
    name: &str,
) -> Option<(String, [f64; 3])> {
    let c = document.connector(name)?;
    let [segment] = c.occurrence.0.as_slice() else {
        return None;
    };
    let local = match &c.anchor {
        AnchorRef::Origin => [1.0, 0.0, 0.0],
        AnchorRef::Explicit {
            offset:
                OffsetExpr {
                    rotation: RotationExpr::Basis { x, .. },
                    ..
                },
        } => *x,
        AnchorRef::Explicit {
            offset:
                OffsetExpr {
                    rotation: RotationExpr::Identity,
                    ..
                },
        } => [1.0, 0.0, 0.0],
        _ => return None,
    };
    let uuid = compiled
        .rows
        .iter()
        .find(|r| r["path"].as_str() == Some(segment.name.as_str()))?["uuid"]
        .as_str()?
        .to_owned();
    let (_, t) = compiled.placed(&uuid)?;
    (t.len() >= 9).then(|| {
        (
            segment.name.clone(),
            [0, 1, 2]
                .map(|i| t[3 * i] * local[0] + t[3 * i + 1] * local[1] + t[3 * i + 2] * local[2]),
        )
    })
}

/// Toggles (or sets) mate `name`'s flip. Planes and axes mate either way round, so the solve
/// keeps whichever way the components start: the floating component turns half a turn about
/// its connector's X axis first, and the mate holds it there.
fn flip_mate(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    name: &str,
    to: Option<bool>,
) -> Result<(), EditFailure> {
    let at = mate_at(document, name)?;
    let was = document
        .connector(&document.mates[at].b)
        .is_some_and(|c| c.flip);
    if set_flip(document, at, to)? == was {
        return Ok(());
    }
    let floating = |o: &str| document.occurrences.iter().any(|x| x.name == o && !x.fixed);
    let (a, b) = (document.mates[at].a.clone(), document.mates[at].b.clone());
    let turn = [b, a]
        .iter()
        .filter_map(|c| connector_x(document, compiled, c))
        .find(|(o, _)| floating(o));
    if let Some((occurrence, axis)) = turn {
        super::place::turn(document, &occurrence, axis, std::f64::consts::PI)?;
    }
    Ok(())
}

pub(super) fn flip(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    selection: &[String],
) -> Result<String, EditFailure> {
    let names = mate_names(selection)?;
    for name in &names {
        flip_mate(document, compiled, name, None)?;
    }
    Ok(format!("Flipped {}", names.join(", ")))
}

fn checked(values: &Values, id: &str) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(false)
}

/// The exact value of field `id`, or the range `[low, high]` when `flag` is checked.
fn target(
    values: &Values,
    id: &str,
    flag: &str,
    [low, high]: [&str; 2],
    unit: &str,
) -> Result<TargetExpr, EditFailure> {
    if checked(values, flag) {
        return Ok(TargetExpr::Range {
            min: field(values, low, unit)?,
            max: field(values, high, unit)?,
        });
    }
    Ok(TargetExpr::Exact {
        value: field(values, id, unit)?,
    })
}

/// A range when `flag` is checked, else no target (free).
pub(super) fn limits(
    values: &Values,
    flag: &str,
    ends: [&str; 2],
    unit: &str,
) -> Result<Option<TargetExpr>, EditFailure> {
    if checked(values, flag) {
        target(values, "", flag, ends, unit).map(Some)
    } else {
        Ok(None)
    }
}

pub(super) const ANGLE: [&str; 2] = ["min", "max"];
pub(super) const TRAVEL: [&str; 2] = ["near", "far"];

pub(super) fn edit(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    selection: &[String],
    values: &Values,
) -> Result<String, EditFailure> {
    let names = mate_names(selection)?;
    let [name] = names.as_slice() else {
        return Err(model("select one mate to edit"));
    };
    let at = mate_at(document, name)?;
    let old = document.mates[at].kind.clone();
    let wanted = values
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or(old.name());
    let kind = match wanted {
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
        "parallel" => MateKind::Parallel,
        "pin_slot" => MateKind::PinSlot,
        "distance" => MateKind::Distance {
            // A plane-to-point distance stays one; any other mate becomes point to point.
            measure: match &old {
                MateKind::Distance { measure, .. } => *measure,
                _ => DistanceMeasure::PointPoint,
            },
            target: TargetExpr::Exact {
                value: field(values, "offset", "mm")?,
            },
        },
        "angle" => MateKind::Angle {
            measure: match &old {
                MateKind::Angle { measure, .. } => *measure,
                _ => AngleMeasure::BetweenZ,
            },
            target: TargetExpr::Exact {
                value: field(values, "angle", "deg")?,
            },
        },
        "gear" => MateKind::Gear {
            ratio: ratio(values)?,
        },
        "screw" => MateKind::Screw {
            lead: field(values, "pitch", "mm")?,
        },
        "rack_pinion" => MateKind::RackPinion {
            travel: field(values, "pitch", "mm")?,
        },
        // A plane on a sphere stays one; any other mate becomes a plane on a cylinder.
        "tangent" => {
            let radius = field(values, "radius", "mm")?;
            MateKind::Tangent {
                kind: match &old {
                    MateKind::Tangent {
                        kind: TangentSpec::PlaneSphere { .. },
                    } => TangentSpec::PlaneSphere { radius },
                    _ => TangentSpec::PlaneCylinder { radius },
                },
            }
        }
        other => super::mate::advanced(document, compiled, values, other, Some(&old))?
            .ok_or_else(|| model(format!("there is no `{other}` mate")))?,
    };
    let changed_kind = kind.name() != old.name();
    document.mates[at].kind = kind;
    // A plane-to-point distance holds its faces parallel with a companion mate; it goes when
    // the distance does.
    if changed_kind && matches!(old, MateKind::Distance { .. }) {
        let companion = format!("{name}_parallel");
        document.mates.retain(|m| m.name != companion);
    }
    flip_mate(document, compiled, name, Some(checked(values, "flip")))?;
    Ok(format!("Edited {name} ({wanted})"))
}

fn ratio(values: &Values) -> Result<ValueExpr, EditFailure> {
    match values.get("ratio") {
        Some(Value::Number(n)) if n.as_f64() == Some(0.0) => {
            Err(model("a gear ratio must be non-zero"))
        }
        Some(Value::Number(n)) => Ok(ValueExpr::Number {
            value: n.as_f64().unwrap_or(1.0),
        }),
        _ => field(values, "ratio", ""),
    }
}

/// A candidate on the dropped component: its pick ID and frame in assembly coordinates.
struct Candidate {
    id: String,
    world: Frame,
}

/// Every face and edge frame of component `uuid`, in assembly coordinates.
fn candidates(compiled: &Compiled, uuid: &str) -> Vec<Candidate> {
    let Some((prototype, transform)) = compiled.placed(uuid) else {
        return Vec::new();
    };
    let bodies = compiled.display["prototypes"][prototype]["part"]["bodies"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for body in &bodies {
        let b = body["index"].as_u64().unwrap_or(0);
        for (list, kind, frame) in [
            ("faces", "face", face_frame as fn(&Value) -> Option<Frame>),
            ("edges", "edge", edge_frame),
        ] {
            for entity in body[list].as_array().into_iter().flatten() {
                let (Some(i), Some(f)) = (entity["index"].as_u64(), frame(entity)) else {
                    continue;
                };
                out.push(Candidate {
                    id: format!("occurrence:{uuid}/{kind}:{b}:{i}"),
                    world: to_world(f, &transform),
                });
            }
        }
    }
    out
}

pub(super) fn smart(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Values,
) -> Result<String, EditFailure> {
    let component = picks_of(values, "component")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the component to mate"))?;
    let onto = picks_of(values, "target")
        .into_iter()
        .next()
        .ok_or_else(|| model("pick the face or axis to mate it onto"))?;
    let moving = parse_pick(&component).ok_or_else(|| model("pick a component to mate"))?;
    let target = parse_pick(&onto)
        .filter(|p| p.entity.is_some())
        .ok_or_else(|| model("pick a face or edge of another component to mate onto"))?;
    if moving.uuid == target.uuid {
        return Err(model("drop the component onto another component"));
    }
    let to = candidates(compiled, &target.uuid)
        .into_iter()
        .find(|c| c.id == onto)
        .map(|c| c.world)
        .ok_or_else(|| model("the picked face or edge has no frame to mate to"))?;
    let all = candidates(compiled, &moving.uuid);
    let near = |c: &Candidate| norm(sub(c.world.origin, to.origin));
    let (kind, best) = match to.kind {
        FrameKind::Plane => {
            // Facing it most squarely, then nearest.
            let best = all
                .iter()
                .filter(|c| c.world.kind == FrameKind::Plane && c.id.contains("/face:"))
                .min_by(|a, b| {
                    let (fa, fb) = (dot(a.world.z, to.z), dot(b.world.z, to.z));
                    if (fa - fb).abs().gt(&1e-6) {
                        fa.total_cmp(&fb)
                    } else {
                        near(a).total_cmp(&near(b))
                    }
                });
            ("coincident", best)
        }
        FrameKind::Axis => {
            let radius = to.radius.unwrap_or(0.0);
            let best = all
                .iter()
                .filter(|c| c.world.kind == FrameKind::Axis && c.world.radius.is_some())
                .min_by(|a, b| {
                    let (ra, rb) = (
                        (a.world.radius.unwrap_or(0.0) - radius).abs(),
                        (b.world.radius.unwrap_or(0.0) - radius).abs(),
                    );
                    if (ra - rb).abs().gt(&1e-7) {
                        ra.total_cmp(&rb)
                    } else {
                        near(a).total_cmp(&near(b))
                    }
                });
            ("concentric", best)
        }
    };
    let best = best.ok_or_else(|| {
        model(match kind {
            "coincident" => "the component has no flat face to put on that face",
            _ => "the component has no round face or edge to line up with that axis",
        })
    })?;
    let request = json!({ "type": kind, "first": [onto], "second": [best.id], "flip": false });
    let summary = super::mate::mate(document, compiled, request.as_object().expect("an object"))?;
    Ok(format!(
        "Smart mate: {}",
        summary.trim_start_matches("Mated ")
    ))
}
