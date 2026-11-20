//! Motion studies and mate positions from the canvas.
//!
//! - `motion.animation` adds a motion study, or changes the one named (`motion:<name>` in the
//!   selection, or the `name` field): the picked mate driven from `from` to `to` (an angle or a
//!   distance, by `quantity`) over `duration` seconds with the `profile` motion law; `track` names
//!   the component whose motion is plotted.
//! - `motion.mate-controller` saves a named mate position (`position:<name>`): `save` keeps the
//!   picked mates' current values under the name; `move` holds each of the position's mates at
//!   its saved value, so the mate solve moves the mechanism there.
//! - `motion.basic` adds a dynamic study, or changes the one named: the mechanism moves under
//!   `gravity` (a direction, or none) for `duration`, with an optional spring, damper and motor
//!   on the picked `mate` (`quantity` angle: N·m/rad, N·m·s/rad, N·m; distance: N/m, N·s/m, N).
//! - `assembly.delete` on `motion:<name>` and `position:<name>` rows removes those studies and
//!   positions.

use super::mate::field;
use super::{model, Compiled};
use crate::features::picks_of;
use runmat_cad::assembly::model::AssemblyDocument;
use runmat_cad::assembly::motion::kinematics::{drivable, mate_value, set_mate_value};
use runmat_cad::assembly::motion::{
    MatePosition, MateValue, MotionDrive, MotionKind, MotionMotor, MotionProfile, MotionSpring,
    MotionStudy,
};
use runmat_cad::syntax::edit::EditFailure;
use runmat_cad::units::{Dimension, Units};
use runmat_cad::value::ValueExpr;
use runmat_cad::value::{Evaluator, NoDynamicValues};
use serde_json::{Map, Value};

type Values = Map<String, Value>;

/// An identifier from a typed name: `Open close` → `open_close`.
fn identifier(name: &str) -> String {
    let mut id: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while id.contains("__") {
        id = id.replace("__", "_");
    }
    let id = id.trim_matches('_').to_owned();
    if id.is_empty() || id.starts_with(|c: char| c.is_ascii_digit()) {
        format!("motion_{id}").trim_end_matches('_').to_owned()
    } else {
        id
    }
}

/// The studies the selection names (`motion:<name>`).
pub(super) fn selected(selection: &[String]) -> impl Iterator<Item = &str> {
    selection.iter().filter_map(|s| s.strip_prefix("motion:"))
}

/// The open assembly's mate a pick names (`mate:/<name>`, or a bare name).
fn mate_name(document: &AssemblyDocument, pick: &str) -> Result<String, EditFailure> {
    let name = match pick.strip_prefix("mate:") {
        Some(rest) => match rest.rsplit_once('/') {
            Some((assembly, name)) if assembly.is_empty() || assembly == "/" => name,
            Some((assembly, name)) => {
                return Err(model(format!(
                    "`{name}` belongs to the sub-assembly `{assembly}`; open it to drive the mate"
                )))
            }
            None => rest,
        },
        None => pick,
    };
    let mate = document
        .mates
        .iter()
        .find(|m| m.name == name)
        .ok_or_else(|| model(format!("the assembly has no mate `{name}`")))?;
    if drivable(&mate.kind).is_none() {
        return Err(model(format!(
            "`{name}` can't be driven: pick a revolute, slider, cylindrical, distance or angle mate"
        )));
    }
    Ok(name.to_owned())
}

/// `motion.animation`: adds or changes a motion study.
pub(super) fn animation(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Values,
    selection: &[String],
) -> Result<String, EditFailure> {
    let editing = selected(selection).next().map(str::to_owned);
    let typed = values
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(identifier);
    let name = typed
        .clone()
        .or_else(|| editing.clone())
        .unwrap_or_else(|| "motion".into());
    let existing = editing
        .as_deref()
        .or(Some(name.as_str()))
        .and_then(|n| document.motions.iter().position(|s| s.name == n));
    if let (Some(i), Some(t)) = (existing, &typed) {
        if document
            .motions
            .iter()
            .enumerate()
            .any(|(k, s)| k != i && &s.name == t)
        {
            return Err(model(format!("there is already a motion study `{t}`")));
        }
    }
    let mate = match picks_of(values, "mate").first() {
        Some(pick) => mate_name(document, pick)?,
        None => match existing.and_then(|i| document.motions[i].drives.first()) {
            Some(d) => d.mate.clone(),
            None => return Err(model("pick the mate to drive")),
        },
    };
    let distance = values.get("quantity").and_then(Value::as_str) == Some("distance");
    let (from, to) = if distance {
        (
            field(values, "from_distance", "mm")?,
            field(values, "to_distance", "mm")?,
        )
    } else {
        (
            field(values, "from_angle", "deg")?,
            field(values, "to_angle", "deg")?,
        )
    };
    let duration = field(values, "duration", "s")?;
    let profile = match values.get("profile").and_then(Value::as_str) {
        Some("cycloidal") => MotionProfile::Cycloidal,
        _ => MotionProfile::Linear,
    };
    let track = match picks_of(values, "track").first() {
        Some(pick) => match pick.strip_prefix("occurrence:") {
            Some(uuid) => Some(compiled.occurrence_name(uuid)?),
            None => Some(pick.clone()),
        },
        None => None,
    };
    let study = MotionStudy {
        name: name.clone(),
        duration,
        drives: vec![MotionDrive { mate, from, to }],
        rate: existing.and_then(|i| document.motions[i].rate),
        profile,
        track,
        kind: MotionKind::Kinematic,
        gravity: None,
        springs: Vec::new(),
        motors: Vec::new(),
    };
    Ok(match existing {
        Some(i) => {
            document.motions[i] = study;
            format!("Changed motion study {name}")
        }
        None => {
            document.motions.push(study);
            format!("Added motion study {name}")
        }
    })
}

/// The positions the selection names (`position:<name>`).
/// Whether a typed value is zero (an unused spring or motor field).
fn is_zero(e: &ValueExpr) -> bool {
    matches!(e, ValueExpr::Quantity { value, .. } if *value == 0.0)
        || matches!(e, ValueExpr::Number { value } if *value == 0.0)
}

/// `motion.basic`: adds or changes a dynamic study.
pub(super) fn basic(
    document: &mut AssemblyDocument,
    compiled: &Compiled,
    values: &Values,
    selection: &[String],
) -> Result<String, EditFailure> {
    let editing = selected(selection).next().map(str::to_owned);
    let typed = values
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(identifier);
    let name = typed
        .clone()
        .or_else(|| editing.clone())
        .unwrap_or_else(|| "swing".into());
    let existing = editing
        .as_deref()
        .or(Some(name.as_str()))
        .and_then(|n| document.motions.iter().position(|s| s.name == n));
    if let (Some(i), Some(t)) = (existing, &typed) {
        if document
            .motions
            .iter()
            .enumerate()
            .any(|(k, s)| k != i && &s.name == t)
        {
            return Err(model(format!("there is already a motion study `{t}`")));
        }
    }
    let duration = field(values, "duration", "s")?;
    let g = |x: f64, y: f64, z: f64| {
        Some(
            [x, y, z]
                .map(|v| ValueExpr::Quantity {
                    value: v,
                    unit: "m/s^2".into(),
                })
                .to_vec(),
        )
    };
    const G: f64 = 9.80665;
    let gravity = match values
        .get("gravity")
        .and_then(Value::as_str)
        .unwrap_or("-z")
    {
        "-z" => None,
        "-y" => g(0.0, -G, 0.0),
        "-x" => g(-G, 0.0, 0.0),
        "+z" => g(0.0, 0.0, G),
        "none" => g(0.0, 0.0, 0.0),
        other => {
            return Err(model(format!(
                "gravity `{other}`: pick a direction or none"
            )))
        }
    };
    let mut springs = Vec::new();
    let mut motors = Vec::new();
    if let Some(pick) = picks_of(values, "mate").first() {
        let mate = mate_name(document, pick)?;
        let angle = values.get("quantity").and_then(Value::as_str) != Some("distance");
        let (k, c, load) = if angle {
            (
                field(values, "stiffness", "N*m/rad")?,
                field(values, "damping", "N*m*s/rad")?,
                field(values, "torque", "N*m")?,
            )
        } else {
            (
                field(values, "stiffness_linear", "N/mm")?,
                field(values, "damping_linear", "N*s/m")?,
                field(values, "force", "N")?,
            )
        };
        if !is_zero(&k) || !is_zero(&c) {
            springs.push(MotionSpring {
                mate: mate.clone(),
                stiffness: (!is_zero(&k)).then_some(k),
                damping: (!is_zero(&c)).then_some(c),
                free: None,
            });
        }
        if !is_zero(&load) {
            motors.push(MotionMotor {
                mate,
                torque: angle.then(|| load.clone()),
                force: (!angle).then_some(load),
            });
        }
    }
    let track = match picks_of(values, "track").first() {
        Some(pick) => match pick.strip_prefix("occurrence:") {
            Some(uuid) => Some(compiled.occurrence_name(uuid)?),
            None => Some(pick.clone()),
        },
        None => None,
    };
    let study = MotionStudy {
        name: name.clone(),
        kind: MotionKind::Dynamic,
        duration,
        drives: Vec::new(),
        gravity,
        springs,
        motors,
        rate: existing.and_then(|i| document.motions[i].rate),
        profile: MotionProfile::Linear,
        track,
    };
    Ok(match existing {
        Some(i) => {
            document.motions[i] = study;
            format!("Changed motion study {name}")
        }
        None => {
            document.motions.push(study);
            format!("Added motion study {name}")
        }
    })
}

pub(super) fn selected_positions(selection: &[String]) -> impl Iterator<Item = &str> {
    selection.iter().filter_map(|s| s.strip_prefix("position:"))
}

/// `motion.mate-controller`: saves a named position, or moves the mechanism to one.
pub(super) fn controller(
    document: &mut AssemblyDocument,
    values: &Values,
    selection: &[String],
) -> Result<String, EditFailure> {
    let editing = selected_positions(selection).next().map(str::to_owned);
    let name = values
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(identifier)
        .or(editing)
        .ok_or_else(|| model("name the position"))?;
    let existing = document.positions.iter().position(|p| p.name == name);
    if values.get("action").and_then(Value::as_str) == Some("move") {
        let i = existing.ok_or_else(|| model(format!("the assembly has no position `{name}`")))?;
        let ev = Evaluator {
            units: &Units,
            dynamic: &NoDynamicValues,
        };
        let env = ev.evaluate_env(&document.inputs, &document.lets, &Default::default());
        for v in document.positions[i].values.clone() {
            let angle = ev.eval_as(&v.value, &env, Dimension::ANGLE).is_ok();
            let mate = document
                .mates
                .iter_mut()
                .find(|m| m.name == v.mate)
                .ok_or_else(|| model(format!("the assembly has no mate `{}` any more", v.mate)))?;
            if !set_mate_value(&mut mate.kind, v.value.clone(), angle) {
                return Err(model(format!(
                    "`{}` can't take the position's value any more",
                    v.mate
                )));
            }
        }
        return Ok(format!("Moved to position {name}"));
    }
    let picked = picks_of(values, "mates");
    let mut saved = Vec::new();
    for pick in &picked {
        let mate = mate_name(document, pick)?;
        let kind = &document
            .mates
            .iter()
            .find(|m| m.name == mate)
            .expect("checked")
            .kind;
        let value = mate_value(kind).cloned().ok_or_else(|| {
            model(format!(
                "`{mate}` has no value to save; give it one with Edit mate first"
            ))
        })?;
        saved.push(MateValue { mate, value });
    }
    if saved.is_empty() {
        match existing {
            // From its row with no picks: the position keeps its mates, each at its value now.
            Some(i) => {
                for v in document.positions[i].values.clone() {
                    let kind = &document
                        .mates
                        .iter()
                        .find(|m| m.name == v.mate)
                        .ok_or_else(|| {
                            model(format!("the assembly has no mate `{}` any more", v.mate))
                        })?
                        .kind;
                    let value = mate_value(kind).cloned().unwrap_or(v.value);
                    saved.push(MateValue {
                        mate: v.mate,
                        value,
                    });
                }
            }
            None => return Err(model("pick the mates whose values the position keeps")),
        }
    }
    let position = MatePosition {
        name: name.clone(),
        values: saved,
    };
    Ok(match existing {
        Some(i) => {
            document.positions[i] = position;
            format!("Saved position {name}")
        }
        None => {
            document.positions.push(position);
            format!("Added position {name}")
        }
    })
}

/// `assembly.delete` on motion study and position rows.
pub(super) fn delete(
    document: &mut AssemblyDocument,
    selection: &[String],
) -> Result<String, EditFailure> {
    let positions: Vec<String> = selected_positions(selection).map(str::to_owned).collect();
    if !positions.is_empty() {
        for n in &positions {
            if !document.positions.iter().any(|p| &p.name == n) {
                return Err(model(format!("the assembly has no position `{n}`")));
            }
        }
        document.positions.retain(|p| !positions.contains(&p.name));
        if selected(selection).next().is_none() {
            return Ok(format!("Deleted position {}", positions.join(", ")));
        }
    }
    let names: Vec<String> = selected(selection).map(str::to_owned).collect();
    for n in &names {
        if !document.motions.iter().any(|s| &s.name == n) {
            return Err(model(format!("the assembly has no motion study `{n}`")));
        }
    }
    document.motions.retain(|s| !names.contains(&s.name));
    Ok(format!("Deleted motion study {}", names.join(", ")))
}
