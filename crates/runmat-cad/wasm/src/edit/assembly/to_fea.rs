//! `motion.to-fea`: the joint loads a dynamic motion study puts on one component at one instant,
//! handed to the study-authoring path that `study.new` uses (FEA's Simulation setup guide).
//!
//! The compile already solves every study and projects each mate's reaction on the bodies it
//! joins, frame by frame (`index.motion[].reactions`: point in mm, force in N, moment in N·m
//! about the connector, all in world axes). This handler picks the frame nearest the asked time,
//! takes the reactions on the chosen component, and turns them into the component's own part
//! axes with that frame's pose: `R^T (p − t)` for the point, `R^T f` and `R^T m` for the force
//! and moment. Gravity goes the same way. Nothing is written: the reply is the request the
//! Desktop sends to the guide, which writes the `.fea` (supports, material and acceptance with
//! the user, as `study.new` does).
//!
//! The joint loads hold the component in dynamic balance with its weight and its inertia at that
//! instant (d'Alembert's principle; Haug 1989, ch. 11), and the reply says so: a static study of
//! the part under the joint loads alone is not in balance unless it carries those too.

use super::model;
use crate::edit::EditOutput;
use crate::runtime::{assembly_index, Files};
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
                .map(|s| s.trim().strip_prefix("motion:").unwrap_or(s.trim()).to_owned())
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
            Ok(if q.get("unit").and_then(Value::as_str) == Some("ms") { v / 1000.0 } else { v })
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
    let r = |i: usize, j: usize| t.get(i * 3 + j).copied().unwrap_or(if i == j { 1.0 } else { 0.0 });
    [0, 1, 2].map(|j| (0..3).map(|i| r(i, j) * v[i]).sum())
}

fn shown(v: [f64; 3], places: i32) -> String {
    let scale = 10f64.powi(places);
    let c = |x: f64| {
        let r = (x * scale).round() / scale;
        // No "-0" in the request.
        if r == 0.0 { "0".to_owned() } else { format!("{r}") }
    };
    format!("[{}, {}, {}]", c(v[0]), c(v[1]), c(v[2]))
}

pub(crate) fn handoff(
    path: &str,
    text: &str,
    files: &Files,
    values: &Map<String, Value>,
    selection: &[String],
) -> Result<EditOutput, EditFailure> {
    let (index, _) = assembly_index(path, text, files).map_err(|_| model("the assembly doesn't compile"))?;
    let studies = index["motion"].as_array().cloned().unwrap_or_default();
    let dynamic: Vec<&Value> = studies.iter().filter(|s| s["kind"] == "dynamic").collect();
    let name = match study_name(values, selection) {
        Some(n) => n,
        None => match dynamic.as_slice() {
            [only] => only["name"].as_str().unwrap_or_default().to_owned(),
            [] => return Err(model("add a Basic motion study first: its joint loads are what a study takes")),
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
    let times: Vec<f64> = study["times"].as_array().map(|t| t.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
    if times.is_empty() {
        return Err(model(format!("`{name}` solved no frames")));
    }
    let asked = seconds(values)?;
    let k = (0..times.len())
        .min_by(|&a, &b| (times[a] - asked).abs().total_cmp(&(times[b] - asked).abs()))
        .unwrap_or(0);
    let at = times[k];

    let rows = index["occurrences"].as_array().cloned().unwrap_or_default();
    let row_by_uuid = |uuid: &str| rows.iter().find(|r| r["uuid"].as_str() == Some(uuid));
    let row_by_path = |p: &str| rows.iter().find(|r| r["path"].as_str() == Some(p));
    let reactions = study["reactions"].as_array().cloned().unwrap_or_default();
    // Every (mate, load) at frame k.
    let loads: Vec<(String, Value)> = reactions
        .iter()
        .flat_map(|m| {
            let mate = m["mate"].as_str().unwrap_or_default().to_owned();
            m["frames"][k].as_array().cloned().unwrap_or_default().into_iter().map(move |l| (mate.clone(), l))
        })
        .collect();
    let component = match values.get("component").and_then(Value::as_str).filter(|s| !s.trim().is_empty()) {
        Some(pick) => match pick.strip_prefix("occurrence:") {
            Some(uuid) => row_by_uuid(uuid.split('/').next().unwrap_or(uuid))
                .and_then(|r| r["path"].as_str())
                .ok_or_else(|| model(format!("no component `{pick}`")))?
                .to_owned(),
            None => pick.trim().to_owned(),
        },
        None => loads
            .first()
            .and_then(|(_, l)| l["occurrence"].as_str())
            .ok_or_else(|| model(format!("`{name}` has no joint loads")))?
            .to_owned(),
    };
    let mine: Vec<&(String, Value)> = loads.iter().filter(|(_, l)| l["occurrence"].as_str() == Some(component.as_str())).collect();
    if mine.is_empty() {
        return Err(model(format!(
            "no mate loads `{component}` in `{name}`: pick a moving component joined by a mate"
        )));
    }
    let row = row_by_path(&component).ok_or_else(|| model(format!("no component `{component}`")))?;
    let part = row["definition"]["file"].as_str().unwrap_or_default().to_owned();
    if !part.ends_with(".part") {
        return Err(model(format!("`{component}` is not a part: pick the part to study")));
    }
    // The component's pose at frame k (a moving one), else where it is placed.
    let uuid = row["uuid"].as_str().unwrap_or_default();
    let pose: Vec<f64> = study["frames"][k]
        .as_array()
        .and_then(|f| f.iter().find(|o| o["uuid"].as_str() == Some(uuid)))
        .and_then(|o| o["transform"].as_array())
        .map(|t| t.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_else(|| world_transform(&row["world"]));
    let origin = [pose.get(9), pose.get(10), pose.get(11)].map(|v| v.copied().unwrap_or(0.0) * 1000.0);

    let file = part.rsplit('/').next().unwrap_or(&part);
    let assembly = path.rsplit('/').next().unwrap_or(path);
    let mut lines = vec![format!(
        "Set up a structural simulation of {file} under the joint loads it carries at {at} s of motion study {name} in {assembly}."
    )];
    lines.push(format!("In {file}'s own axes (points in mm, forces in N, moments in N*m about the point):"));
    for (mate, l) in &mine {
        let p = vec3(&l["point"]);
        let point = into_part(&pose, [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]]);
        lines.push(format!(
            "- mate {mate} at {}: force {}, moment {}",
            shown(point, 3),
            shown(into_part(&pose, vec3(&l["force"])), 6),
            shown(into_part(&pose, vec3(&l["moment"])), 6),
        ));
    }
    // Gravity is its own paragraph: the agent panel renders the request as Markdown.
    lines.push(String::new());
    let gravity = study["setup"]["gravity"].as_array().map(|_| vec3(&study["setup"]["gravity"])).unwrap_or([0.0, 0.0, -9.80665]);
    lines.push(format!(
        "Gravity in {file}'s axes: {} m/s^2. These joint loads balance the part's weight and its inertia at that instant.",
        shown(into_part(&pose, gravity), 6)
    ));
    Ok(EditOutput { files: Vec::new(), summary: lines.join("\n") })
}

/// An index row's `world` transform as the display's 12 numbers (rotation rows, translation m).
fn world_transform(world: &Value) -> Vec<f64> {
    let rotation: Vec<f64> = world["rotation"]
        .as_array()
        .map(|rows| rows.iter().flat_map(|r| vec3(r)).collect())
        .unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let mut out = rotation;
    out.extend(vec3(&world["translation"]));
    out
}
