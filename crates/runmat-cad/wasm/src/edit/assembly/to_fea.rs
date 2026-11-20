//! `motion.to-fea`: the loads a dynamic motion study puts on one component at one instant,
//! handed to the study-authoring path that `study.new` uses (FEA's Simulation setup guide).
//!
//! The compile already solves every study and records, frame by frame, each mate's load on the
//! bodies it joins (`index.motion[].reactions`: point in mm, force in N, moment in N·m about the
//! connector, world axes; the joint's reaction plus any spring, damper or motor on the mate) and
//! each body's motion (`index.motion[].bodies`: centre-of-mass acceleration, angular velocity and
//! angular acceleration, from the same solve). This handler picks the frame nearest the asked
//! time and turns the chosen component's loads and motion into its own part axes with that
//! frame's pose: `R^T (p − t)` for a point, `R^T v` for a vector.
//!
//! The mate loads hold the part in dynamic balance with its weight and its inertia (d'Alembert's
//! principle; Haug 1989, §11.2): `Σ f + m g − m a = 0`, and about the centre of mass
//! `Σ (m_j + r_j × f_j) − (J α + ω × J ω) = 0`. A static study under the mate loads alone is not
//! in balance, so the request also hands over the inertia, in the forms `.fea` has:
//!
//! - weight and the linear inertia together as one uniform `body_force`, `ρ (g − a)` (N/m³), from
//!   the part's density;
//! - the rotation (`ω`, `α` about the centre of mass) as data, with the moment it leaves
//!   unbalanced. `.fea` has no rotational body load (centrifugal `ρ ω × (ω × r)` or
//!   `ρ α × r`, which vary over the part), and the request says so.
//!
//! Nothing is written: the reply is the request the Desktop sends to the guide, which writes the
//! `.fea` (supports, material and acceptance with the user, as `study.new` does).

use super::model;
use crate::edit::EditOutput;
use crate::runtime::{assembly_index, Files};
use runmat_cad::assembly::motion::dynamics::BodyState;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{Map, Value};

/// The study a row selection names (`motion:<name>`), else the `study` value.
fn study_name(values: &Map<String, Value>, selection: &[String]) -> Option<String> {
    selection
        .iter()
        .find_map(|id| id.strip_prefix("motion:").map(str::to_owned))
        .or_else(|| {
            values
                .get("study")
                .and_then(Value::as_str)
                .map(|s| {
                    s.trim()
                        .strip_prefix("motion:")
                        .unwrap_or(s.trim())
                        .to_owned()
                })
                .filter(|s| !s.is_empty())
        })
}

/// The asked time in seconds: a number (s) or a quantity (`0.2 s`, `200 ms`).
fn seconds(values: &Map<String, Value>) -> Result<f64, EditFailure> {
    match values.get("time") {
        None | Some(Value::Null) => Ok(0.0),
        Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or(0.0)),
        Some(Value::String(s)) => {
            let s = s.trim();
            let (number, unit) = s
                .find(|c: char| c.is_ascii_alphabetic())
                .map(|at| (s[..at].trim(), s[at..].trim()))
                .unwrap_or((s, "s"));
            let v: f64 = number
                .parse()
                .map_err(|_| model(format!("`time` isn't a time: {s}")))?;
            match unit {
                "s" => Ok(v),
                "ms" => Ok(v / 1000.0),
                other => Err(model(format!("`time` takes s or ms, not `{other}`"))),
            }
        }
        Some(Value::Object(q)) => {
            let v = q.get("value").and_then(Value::as_f64).unwrap_or(0.0);
            Ok(if q.get("unit").and_then(Value::as_str) == Some("ms") {
                v / 1000.0
            } else {
                v
            })
        }
        Some(other) => Err(model(format!("`time` isn't a time: {other}"))),
    }
}

fn vec3(v: &Value) -> [f64; 3] {
    let at = |k: usize| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    [at(0), at(1), at(2)]
}

/// `R^T v`, with `t` the display transform's rotation row by row (then translation).
fn into_part(t: &[f64], v: [f64; 3]) -> [f64; 3] {
    let r = |i: usize, j: usize| {
        t.get(i * 3 + j)
            .copied()
            .unwrap_or(if i == j { 1.0 } else { 0.0 })
    };
    [0, 1, 2].map(|j| (0..3).map(|i| r(i, j) * v[i]).sum())
}

fn shown(v: [f64; 3], places: i32) -> String {
    let scale = 10f64.powi(places);
    let c = |x: f64| {
        let r = (x * scale).round() / scale;
        // No "-0" in the request.
        if r == 0.0 {
            "0".to_owned()
        } else {
            format!("{r}")
        }
    };
    format!("[{}, {}, {}]", c(v[0]), c(v[1]), c(v[2]))
}

/// One mate's load: the mate, the point, the force and the moment about the point.
pub(crate) type JointLoad = (String, [f64; 3], [f64; 3], [f64; 3]);

/// The loads on one part at one frame of a dynamic study, in the part's own axes: what Results
/// to .fea hands over. Points in mm; everything else SI.
#[derive(Clone, Debug)]
pub(crate) struct Handoff {
    pub file: String,
    pub assembly: String,
    pub study: String,
    /// The frame's time (s).
    pub time: f64,
    /// Each mate's load: the mate, the point, the force and the moment about the point.
    pub joints: Vec<JointLoad>,
    pub gravity: [f64; 3],
    pub mass: f64,
    /// The centre of mass (mm).
    pub center: [f64; 3],
    /// The inertia tensor about the centre of mass (kg·m²).
    pub inertia: [[f64; 3]; 3],
    pub acceleration: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub angular_acceleration: [f64; 3],
    /// kg/m³, when the part has one.
    pub density: Option<f64>,
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[cfg(test)]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Handoff {
    /// `m g` (N).
    pub fn weight(&self) -> [f64; 3] {
        self.gravity.map(|g| g * self.mass)
    }

    /// The inertia load (d'Alembert), as the dynamics states it: the force `−m a` (N) at the
    /// centre of mass and the moment `−(J α + ω × J ω)` about it (N·m).
    fn inertia(&self) -> ([f64; 3], [f64; 3]) {
        let state = BodyState {
            center: self.center.map(|c| c / 1000.0),
            acceleration: self.acceleration,
            angular_velocity: self.angular_velocity,
            angular_acceleration: self.angular_acceleration,
        };
        // Already in the part's axes.
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        state.inertia_load(self.mass, &self.inertia, &identity)
    }

    /// The linear inertia load `−m a` (N), at the centre of mass.
    pub fn inertia_force(&self) -> [f64; 3] {
        self.inertia().0
    }

    /// The rotational inertia load's resultant about the centre of mass, `−(J α + ω × J ω)`
    /// (N·m).
    pub fn inertia_moment(&self) -> [f64; 3] {
        self.inertia().1
    }

    /// Weight and linear inertia as one uniform body-force density, `ρ (g − a)` (N/m³).
    pub fn body_force(&self) -> Option<[f64; 3]> {
        self.density
            .map(|rho| [0, 1, 2].map(|k| rho * (self.gravity[k] - self.acceleration[k])))
    }

    /// Whether the part turns or spins up at this instant.
    pub fn turns(&self) -> bool {
        self.angular_velocity
            .iter()
            .chain(&self.angular_acceleration)
            .any(|v| v.abs() > 1e-9)
    }

    /// The mate loads' resultant: the force, and the moment about the centre of mass (the
    /// balance check's side of it).
    #[cfg(test)]
    pub fn joint_resultant(&self) -> ([f64; 3], [f64; 3]) {
        self.joints
            .iter()
            .fold(([0.0; 3], [0.0; 3]), |(f, m), (_, p, jf, jm)| {
                let arm = [0, 1, 2].map(|k| (p[k] - self.center[k]) / 1000.0);
                (add(f, *jf), add(m, add(*jm, cross(arm, *jf))))
            })
    }
}

/// The loads on the picked part at the frame nearest the asked time.
pub(crate) fn loads(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<Handoff, EditFailure> {
    let (index, _) =
        assembly_index(path, text, files).map_err(|_| model("the assembly doesn't compile"))?;
    let studies = index["motion"].as_array().cloned().unwrap_or_default();
    let dynamic: Vec<&Value> = studies.iter().filter(|s| s["kind"] == "dynamic").collect();
    let name = match study_name(values, selection) {
        Some(n) => n,
        None => match dynamic.as_slice() {
            [only] => only["name"].as_str().unwrap_or_default().to_owned(),
            [] => {
                return Err(model(
                    "add a Basic motion study first: its joint loads are what a study takes",
                ))
            }
            _ => return Err(model("pick the motion study, from its row or by name")),
        },
    };
    let study = studies
        .iter()
        .find(|s| s["name"].as_str() == Some(name.as_str()))
        .ok_or_else(|| model(format!("there is no motion study `{name}`")))?;
    if study["kind"] != "dynamic" {
        return Err(model(format!(
            "`{name}` drives its mates, so it has no loads: Results to .fea takes a Basic motion (dynamic) study"
        )));
    }
    let times: Vec<f64> = study["times"]
        .as_array()
        .map(|t| t.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    if times.is_empty() {
        return Err(model(format!("`{name}` solved no frames")));
    }
    let asked = seconds(values)?;
    let k = (0..times.len())
        .min_by(|&a, &b| {
            (times[a] - asked)
                .abs()
                .total_cmp(&(times[b] - asked).abs())
        })
        .unwrap_or(0);

    let rows = index["occurrences"].as_array().cloned().unwrap_or_default();
    let row_by_uuid = |uuid: &str| rows.iter().find(|r| r["uuid"].as_str() == Some(uuid));
    let row_by_path = |p: &str| rows.iter().find(|r| r["path"].as_str() == Some(p));
    let reactions = study["reactions"].as_array().cloned().unwrap_or_default();
    // Every (mate, load) at frame k.
    let all: Vec<(String, Value)> = reactions
        .iter()
        .flat_map(|m| {
            let mate = m["mate"].as_str().unwrap_or_default().to_owned();
            m["frames"][k]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .map(move |l| (mate.clone(), l))
        })
        .collect();
    let component = match values
        .get("component")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        Some(pick) => match pick.strip_prefix("occurrence:") {
            Some(uuid) => row_by_uuid(uuid.split('/').next().unwrap_or(uuid))
                .and_then(|r| r["path"].as_str())
                .ok_or_else(|| model(format!("no component `{pick}`")))?
                .to_owned(),
            None => pick.trim().to_owned(),
        },
        None => all
            .first()
            .and_then(|(_, l)| l["occurrence"].as_str())
            .ok_or_else(|| model(format!("`{name}` has no joint loads")))?
            .to_owned(),
    };
    let mine: Vec<&(String, Value)> = all
        .iter()
        .filter(|(_, l)| l["occurrence"].as_str() == Some(component.as_str()))
        .collect();
    if mine.is_empty() {
        return Err(model(format!(
            "no mate loads `{component}` in `{name}`: pick a moving component joined by a mate"
        )));
    }
    let row =
        row_by_path(&component).ok_or_else(|| model(format!("no component `{component}`")))?;
    let part = row["definition"]["file"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    if !part.ends_with(".part") {
        return Err(model(format!(
            "`{component}` is not a part: pick the part to study"
        )));
    }
    let body = study["bodies"]
        .as_array()
        .and_then(|b| {
            b.iter()
                .find(|b| b["occurrence"].as_str() == Some(component.as_str()))
        })
        .ok_or_else(|| {
            model(format!(
                "`{component}` doesn't move on its own in `{name}`: pick the part a mate carries"
            ))
        })?;
    // The component's pose at frame k (a moving one), else where it is placed.
    let uuid = row["uuid"].as_str().unwrap_or_default();
    let pose: Vec<f64> = study["frames"][k]
        .as_array()
        .and_then(|f| f.iter().find(|o| o["uuid"].as_str() == Some(uuid)))
        .and_then(|o| o["transform"].as_array())
        .map(|t| t.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_else(|| world_transform(&row["world"]));
    let origin =
        [pose.get(9), pose.get(10), pose.get(11)].map(|v| v.copied().unwrap_or(0.0) * 1000.0);
    let joints = mine
        .iter()
        .map(|(mate, l)| {
            let p = vec3(&l["point"]);
            (
                mate.clone(),
                into_part(
                    &pose,
                    [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]],
                ),
                into_part(&pose, vec3(&l["force"])),
                into_part(&pose, vec3(&l["moment"])),
            )
        })
        .collect();
    let state = &body["frames"][k];
    let gravity = study["setup"]["gravity"]
        .as_array()
        .map(|_| vec3(&study["setup"]["gravity"]))
        .unwrap_or([0.0, 0.0, -9.80665]);
    let inertia = body["inertia"]
        .as_array()
        .map(|r| [0, 1, 2].map(|i| vec3(r.get(i).unwrap_or(&Value::Null))))
        .unwrap_or_default();
    let configuration = row["configuration"]
        .as_str()
        .filter(|c| !c.is_empty() && *c != "default");
    let density = files
        .get(&part)
        .and_then(|t| crate::mass::part_mass(&part, t, configuration).ok())
        .and_then(|r| r.parts.first().and_then(|p| p.density));
    Ok(Handoff {
        file: part.rsplit('/').next().unwrap_or(&part).to_owned(),
        assembly: path.rsplit('/').next().unwrap_or(path).to_owned(),
        study: name,
        time: times[k],
        joints,
        gravity: into_part(&pose, gravity),
        mass: body["mass"].as_f64().unwrap_or(0.0),
        center: vec3(&body["center"]),
        inertia,
        acceleration: into_part(&pose, vec3(&state["acceleration"])),
        angular_velocity: into_part(&pose, vec3(&state["angular_velocity"])),
        angular_acceleration: into_part(&pose, vec3(&state["angular_acceleration"])),
        density,
    })
}

/// The request for the Simulation setup guide; the agent panel renders it as Markdown, so each
/// load family is its own paragraph.
pub(crate) fn request(h: &Handoff) -> String {
    let Handoff { file, .. } = h;
    let mut lines = vec![format!(
        "Set up a structural simulation of {file} under the loads it carries at {} s of motion study {} in {}.",
        h.time, h.study, h.assembly
    )];
    lines.push(String::new());
    lines.push(format!("Mate loads, in {file}'s own axes (points in mm, forces in N, moments in N*m about the point):"));
    lines.push(String::new());
    for (mate, p, f, m) in &h.joints {
        lines.push(format!(
            "- mate {mate} at {}: force {}, moment {}",
            shown(*p, 3),
            shown(*f, 6),
            shown(*m, 6)
        ));
    }
    lines.push(String::new());
    let g = shown(h.gravity, 6);
    let a = shown(h.acceleration, 6);
    let total = add(h.weight(), h.inertia_force());
    match (h.body_force(), h.density) {
        (Some(b), Some(rho)) => lines.push(format!(
            "Weight and linear inertia as one body_force on {file}'s solid: {} N/m^3. This is rho (g - a), with density rho = {} kg/m^3, gravity g = {g} m/s^2 and the centre of mass's acceleration a = {a} m/s^2. Over the part's {} kg it totals {} N: weight {} N plus inertia force -m a = {} N.",
            shown(b, 3),
            round(rho, 6),
            round(h.mass, 9),
            shown(total, 6),
            shown(h.weight(), 6),
            shown(h.inertia_force(), 6),
        )),
        _ => lines.push(format!(
            "{file} has no density, so the weight and linear inertia can't be written as a body_force (N/m^3). Gravity g = {g} m/s^2 and the centre of mass's acceleration a = {a} m/s^2; over the part's {} kg they total {} N.",
            round(h.mass, 9),
            shown(total, 6),
        )),
    }
    lines.push(String::new());
    if h.turns() {
        lines.push(format!(
            "Rotation: {file} turns at omega = {} rad/s with angular acceleration alpha = {} rad/s^2, about its centre of mass at {} mm. The .fea language has no rotational body load yet (the centrifugal density rho omega x (omega x r) or rho alpha x r, which vary over the part), so the study can't apply this part of the inertia. Its resultant about the centre of mass is the moment -(J alpha + omega x J omega) = {} N*m; without it, the loads above leave that moment unbalanced, and the stress the spin itself causes is not in the study.",
            shown(h.angular_velocity, 6),
            shown(h.angular_acceleration, 6),
            shown(h.center, 3),
            shown(h.inertia_moment(), 6),
        ));
    } else {
        lines.push(format!("{file} doesn't turn at this instant, so the mate loads and the body force balance on their own."));
    }
    lines.join("\n")
}

fn round(v: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    (v * scale).round() / scale
}

pub(crate) fn handoff(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<EditOutput, EditFailure> {
    let h = loads(path, text, files, values, selection)?;
    Ok(EditOutput {
        files: Vec::new(),
        summary: request(&h),
    })
}

/// An index row's `world` transform as the display's 12 numbers (rotation rows, translation m).
fn world_transform(world: &Value) -> Vec<f64> {
    let rotation: Vec<f64> = world["rotation"]
        .as_array()
        .map(|rows| rows.iter().flat_map(vec3).collect())
        .unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let mut out = rotation;
    out.extend(vec3(&world["translation"]));
    out
}
