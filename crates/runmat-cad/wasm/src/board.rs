//! Circuit boards in assemblies: an occurrence of a `.circuit` (or its `.circuit.layout`)
//! draws the board's 3D model, as the circuit's own 3D tab shows it
//! (`runmat_circuit::board3d`): substrate, mask, copper, finish, silkscreen and component
//! bodies, one body per material in its color. The model is in millimetres with z = 0 at the
//! board's bottom face; the display is in metres.
use crate::runtime::Files;
use runmat_cad::display::{BodyDisplay, Bounds, FaceDisplay, PartDisplay};

/// The layout a circuit occurrence draws: the `.circuit.layout` itself, or a `.circuit`'s
/// sibling layout.
pub fn layout_path(file: &str) -> String {
    if file.ends_with(".circuit.layout") {
        file.to_owned()
    } else {
        format!("{file}.layout")
    }
}

/// The board's layout compiled in a circuit session, or why it doesn't compile to a board.
fn compiled_board(file: &str, files: &Files) -> Result<runmat_circuit::runtime::Session, String> {
    let layout = layout_path(file);
    let text = files
        .get(&layout)
        .ok_or_else(|| format!("{layout} isn't open or in the workspace"))?;
    let mut circuit_files = runmat_circuit::runtime::Files::new();
    // The circuit, its lock and what they name (resolved paths, as the session reads them).
    let mut queue = runmat_circuit::runtime::dependencies(&layout, text);
    while let Some(dep) = queue.pop() {
        if circuit_files.contains_key(&dep) {
            continue;
        }
        if let Some(t) = files.get(&dep) {
            queue.extend(runmat_circuit::runtime::dependencies(&dep, t));
            circuit_files.insert(dep, t.clone());
        }
    }
    circuit_files.remove(&layout);
    let mut session = runmat_circuit::runtime::Session::new();
    let out = session.compile(&layout, text, &circuit_files, &Default::default());
    if session.board_model().is_none() {
        let first = out
            .diagnostics
            .iter()
            .find_map(|d| serde_json::to_value(d).ok().and_then(|v| v["message"].as_str().map(str::to_owned)));
        return Err(format!("{layout} has no board{}", first.map(|m| format!(": {m}")).unwrap_or_default()));
    }
    Ok(session)
}

/// The board's 3D model as a part display (metres), or why there is none.
pub fn board_display(file: &str, files: &Files) -> Result<PartDisplay, String> {
    let model = compiled_board(file, files)?.board_model().ok_or("no board")?;
    let mm = |v: f64| (v * 1e-3) as f32;
    let bodies = model
        .bodies
        .iter()
        .enumerate()
        .map(|(b, body)| BodyDisplay {
            index: b as u32,
            reference: Some(body.reference.clone()),
            faces: body
                .faces
                .iter()
                .enumerate()
                .map(|(f, face)| FaceDisplay {
                    index: f as u32,
                    reference: face.reference.clone(),
                    names: Vec::new(),
                    regions: Vec::new(),
                    features: face.features.clone(),
                    positions: face.positions.iter().map(|&v| mm(v)).collect(),
                    triangles: face.triangles.clone(),
                    normals: Vec::new(),
                    surface: None,
                })
                .collect(),
            edges: Vec::new(),
            vertices: Vec::new(),
            triangle_count: body.triangle_count as u32,
            color: Some(body.color.map(|c| c as f32)),
        })
        .collect();
    Ok(PartDisplay {
        bodies,
        bounds: Some(Bounds {
            min: model.bounds.min.map(mm),
            max: model.bounds.max.map(mm),
        }),
        ..Default::default()
    })
}

/// How a parts list describes a circuit board: its outline size, thickness, copper layer
/// count and the components on it, e.g. `Circuit board daq, 90 x 60 x 1.6 mm, 6 copper layers,
/// 38 components`.
pub fn board_description(file: &str, files: &Files) -> Result<String, String> {
    let session = compiled_board(file, files)?;
    let index = session.index().ok_or("no board index")?;
    let board = index.board.as_ref().ok_or("no board index")?;
    let model = session.board_model().ok_or("no board")?;
    let width = model.bounds.max[0] - model.bounds.min[0];
    let depth = model.bounds.max[1] - model.bounds.min[1];
    let thickness: f64 = board.layers.iter().map(|l| l.thickness_mm).sum();
    let copper = board.layers.iter().filter(|l| l.kind == "copper").count();
    let trim = |v: f64| {
        let t = format!("{v:.1}");
        t.strip_suffix(".0").map(str::to_owned).unwrap_or(t)
    };
    Ok(format!(
        "Circuit board {}, {} x {} x {} mm, {copper} copper layers, {} components",
        board.name,
        trim(width),
        trim(depth),
        trim(thickness),
        board.footprints.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAQ: &str = "../../runmat-circuit/tests/projects/daq";

    fn daq_files() -> Files {
        let mut files = Files::new();
        for name in ["daq.circuit", "daq.circuit.layout", "daq.circuit.lock"] {
            let text = std::fs::read_to_string(format!("{}/{DAQ}/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            files.insert(format!("/p/{name}"), text);
        }
        files
    }

    #[test]
    fn a_circuit_draws_its_board_model_in_metres_with_body_colors() {
        let display = board_display("/p/daq.circuit", &daq_files()).expect("the daq board");
        let b = display.bounds.as_ref().unwrap();
        assert!(((b.max[0] - b.min[0]) - 0.090).abs() < 1e-3, "90 mm wide: {b:?}");
        assert!(((b.max[1] - b.min[1]) - 0.060).abs() < 1e-3, "60 mm deep: {b:?}");
        assert!(display.bodies.len() > 3, "mask, copper, silk and component bodies");
        assert!(display.bodies.iter().all(|body| body.color.is_some()));
    }

    #[test]
    fn a_parts_list_describes_a_board_by_size_layers_and_components() {
        let d = board_description("/p/daq.circuit", &daq_files()).expect("the daq board");
        assert!(d.starts_with("Circuit board daq, 90 x 60 x "), "{d}");
        assert!(d.contains(" copper layers, ") && d.ends_with(" components"), "{d}");
    }

    #[test]
    fn an_assembly_places_a_circuit_board_as_a_drawn_prototype() {
        let mut files = daq_files();
        let text = "runmat 0.7\nstd 1\n\nassembly box {\n    use circuit daq from \"daq.circuit\" as board {\n        placement = { placement = offset, offset = { translation = [15 mm, 10 mm, 10.5 mm] } }\n    }\n}\n";
        files.insert("/p/box.assembly".into(), text.into());
        let deps = crate::runtime::dependencies("/p/daq.circuit", &files["/p/daq.circuit"]);
        assert!(deps.contains(&"/p/daq.circuit.layout".to_owned()), "{deps:?}");
        let (_, display, diagnostics) = crate::runtime::compile_assembly("/p/box.assembly", text, &files).expect("compiles");
        let display: serde_json::Value = serde_json::to_value(&display).unwrap();
        let shape = &display["prototypes"][0];
        assert_eq!(shape["shape"], "mesh", "{diagnostics:?}");
        assert!(shape["part"]["bodies"].as_array().is_some_and(|b| b.len() > 3));
        assert!((display["occurrences"][0]["transform"][9].as_f64().unwrap() - 0.015).abs() < 1e-6);
    }
}
