//! Circuit boards in assemblies: an occurrence of a `.circuit` (or its `.circuit.layout`)
//! draws the board's 3D model, as the circuit's own 3D tab shows it
//! (`runmat_circuit::board3d`): substrate, mask, copper, finish, silkscreen and component
//! bodies, one body per material in its color. The model is in millimetres with z = 0 at the
//! board's bottom face; the display is in metres.
use crate::runtime::Files;
use runmat_cad::display::{BodyDisplay, Bounds, EdgeDisplay, FaceDisplay, PartDisplay};
use std::collections::BTreeMap;

/// The bodies a drawing and the 3D view outline: the substrate and its mask (the board
/// outline and cutouts), drills (holes) and the component bodies. Copper, pads and silkscreen
/// are surface art, and terminals, leads and pins sit inside their package's outline at
/// drawing scale: outlining them would bury the drawing in strokes, so a simplified
/// board representation leaves them out.
fn outlined(body: &str) -> bool {
    !matches!(body, "Copper" | "Pads" | "Silkscreen" | "Terminals" | "Leads" | "Pins")
}

/// A mesh's feature edges as polylines of segments (mm): where only one triangle meets a
/// segment (an open boundary, such as a cap's rim), or where the two triangles meeting there
/// turn by more than 20° (a box's corner, the board's edge). Vertices are matched by position,
/// so triangle soup and shared-vertex meshes give the same edges, and coplanar triangles
/// (a quad's diagonal) give none.
fn feature_edges(positions: &[f64], triangles: &[u32]) -> Vec<[[f64; 3]; 2]> {
    let at = |i: u32| -> [f64; 3] {
        let k = 3 * i as usize;
        [positions[k], positions[k + 1], positions[k + 2]]
    };
    let key = |p: [f64; 3]| p.map(|v| (v * 1e3).round() as i64);
    let normal = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> Option<[f64; 3]> {
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        (l > 1e-12).then(|| n.map(|x| x / l))
    };
    type Key = [i64; 3];
    let mut uses: BTreeMap<(Key, Key), (Vec<[f64; 3]>, [f64; 3], [f64; 3])> = BTreeMap::new();
    for t in triangles.chunks_exact(3) {
        if t.iter().any(|&i| 3 * i as usize + 2 >= positions.len()) {
            continue;
        }
        let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
        let Some(n) = normal(a, b, c) else { continue };
        for (p, q) in [(a, b), (b, c), (c, a)] {
            let (kp, kq) = (key(p), key(q));
            let id = if kp <= kq { (kp, kq) } else { (kq, kp) };
            uses.entry(id).or_insert_with(|| (Vec::new(), p, q)).0.push(n);
        }
    }
    let crease = 20f64.to_radians().cos();
    uses.into_values()
        .filter(|(normals, _, _)| match normals.as_slice() {
            [_] => true,
            [m, n] => m[0] * n[0] + m[1] * n[1] + m[2] * n[2] < crease,
            _ => true,
        })
        .map(|(_, p, q)| [p, q])
        .collect()
}

/// The layout a circuit occurrence draws: the `.circuit.layout` itself, or a `.circuit`'s
/// sibling layout.
pub fn layout_path(file: &str) -> String {
    if file.ends_with(".circuit.layout") {
        file.to_owned()
    } else {
        format!("{file}.layout")
    }
}

/// What a board's compile reads: its layout's path and text, and the circuit files the layout
/// names (the circuit, its lock, and what they name), by path.
pub(crate) struct BoardInputs {
    pub layout: String,
    pub text: String,
    pub circuit_files: runmat_circuit::runtime::Files,
}

/// The files board `file` compiles from, or why its layout isn't there.
pub(crate) fn board_inputs(file: &str, files: &Files) -> Result<BoardInputs, String> {
    let layout = layout_path(file);
    let text = files
        .get(&layout)
        .ok_or_else(|| format!("{layout} isn't open or in the workspace"))?;
    let mut circuit_files = runmat_circuit::runtime::Files::new();
    // The circuit, its lock and what they name (resolved paths, as the session reads them).
    let mut queue = runmat_circuit::runtime::dependencies(&layout, text);
    let mut listed = std::collections::BTreeSet::new();
    while let Some(dep) = queue.pop() {
        // A folder (`/p/esc/`): the layout names its circuit by symbol, so every file directly
        // in the folder is a candidate.
        if dep.ends_with('/') {
            if listed.insert(dep.clone()) {
                queue.extend(
                    files
                        .keys()
                        .filter(|p| p.strip_prefix(dep.as_str()).is_some_and(|rest| !rest.contains('/')))
                        .cloned(),
                );
            }
            continue;
        }
        if circuit_files.contains_key(&dep) {
            continue;
        }
        if let Some(t) = files.get(&dep) {
            queue.extend(runmat_circuit::runtime::dependencies(&dep, t));
            circuit_files.insert(dep, t.clone());
        }
    }
    circuit_files.remove(&layout);
    Ok(BoardInputs { layout, text: text.clone(), circuit_files })
}

/// The board's layout compiled in a circuit session, or why it doesn't compile to a board.
pub(crate) fn compiled_board(file: &str, files: &Files) -> Result<runmat_circuit::runtime::Session, String> {
    compile_inputs(&board_inputs(file, files)?)
}

fn compile_inputs(inputs: &BoardInputs) -> Result<runmat_circuit::runtime::Session, String> {
    let BoardInputs { layout, text, circuit_files } = inputs;
    let mut session = runmat_circuit::runtime::Session::new();
    let out = session.compile(layout, text, circuit_files, &Default::default());
    if session.board_model().is_none() {
        let first = out
            .diagnostics
            .iter()
            .find_map(|d| serde_json::to_value(d).ok().and_then(|v| v["message"].as_str().map(str::to_owned)));
        return Err(format!("{layout} has no board{}", first.map(|m| format!(": {m}")).unwrap_or_default()));
    }
    Ok(session)
}

/// The board's 3D model as a part display (metres), or why there is none. Kept across compiles
/// by everything its compile reads ([`crate::runtime::part_cache::board_key`]), so an assembly
/// recompile reuses an unchanged board.
pub fn board_display(file: &str, files: &Files) -> Result<PartDisplay, String> {
    let inputs = board_inputs(file, files)?;
    let key = crate::runtime::part_cache::board_key(file, &inputs.layout, &inputs.text, &inputs.circuit_files);
    if crate::runtime::part_cache::skip(&key) {
        return Err("its stored model isn't loaded yet".into());
    }
    crate::runtime::part_cache::board_part(key, || board_display_of(&inputs))
}

fn board_display_of(inputs: &BoardInputs) -> Result<PartDisplay, String> {
    let model = compile_inputs(inputs)?.board_model().ok_or("no board")?;
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
                    label: None,
                    features: face.features.clone(),
                    positions: face.positions.iter().map(|&v| mm(v)).collect(),
                    triangles: face.triangles.clone(),
                    normals: Vec::new(),
                    surface: None,
                })
                .collect(),
            // The outline a drawing projects and the 3D view draws over the shading: each
            // feature segment its own edge on the body's face.
            edges: if outlined(&body.reference) {
                body.faces
                    .iter()
                    .enumerate()
                    .flat_map(|(f, face)| feature_edges(&face.positions, &face.triangles).into_iter().map(move |s| (f as u32, s)))
                    .enumerate()
                    .map(|(e, (f, [p, q]))| EdgeDisplay {
                        index: e as u32,
                        reference: None,
                        faces: [f, f],
                        points: p.into_iter().chain(q).map(mm).collect(),
                    })
                    .collect()
            } else {
                Vec::new()
            },
            vertices: Vec::new(),
            triangle_count: body.triangle_count as u32,
            color: Some(body.color.map(|c| c as f32)),
            flat_pattern: false,
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
        // The board's outline and the component bodies carry their edges; surface art doesn't.
        let edges = |name: &str| display.bodies.iter().filter(|b| b.reference.as_deref() == Some(name)).map(|b| b.edges.len()).sum::<usize>();
        assert!(edges("Board edge") >= 8, "the substrate's top and bottom outline and its corners");
        assert!(edges("Packages") >= 12, "a package body is at least a box: {}", edges("Packages"));
        assert_eq!(edges("Silkscreen"), 0);
        assert_eq!(edges("Copper"), 0);
        assert_eq!(edges("Terminals"), 0);
    }

    /// A board recompiles only when a file its compile reads changes: an unchanged one is reused,
    /// a file outside its folder changes nothing (the layout reads its circuit's folder), and an
    /// edit to its layout compiles it again.
    #[test]
    fn an_unchanged_board_is_reused_and_an_edited_one_compiles_again() {
        use crate::runtime::part_cache::{clear, stats};
        clear();
        let mut files = daq_files();
        let first = board_display("/p/daq.circuit", &files).expect("the daq board");
        files.insert("/q/arm.part".into(), "part arm {}".into());
        let again = board_display("/p/daq.circuit", &files).expect("the daq board");
        assert_eq!(first, again);
        assert_eq!((stats().hits, stats().misses), (1, 1));
        let layout = files.get_mut("/p/daq.circuit.layout").unwrap();
        layout.push_str("\n// edited\n");
        board_display("/p/daq.circuit", &files).expect("the edited board");
        assert_eq!((stats().hits, stats().misses), (1, 2));
    }

    #[test]
    fn feature_edges_follow_creases_and_open_rims_but_not_coplanar_diagonals() {
        // A unit square as two triangles: four rim edges, no diagonal.
        let square = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(feature_edges(&square, &[0, 1, 2, 0, 2, 3]).len(), 4);
        // The same square as soup (no shared vertices) gives the same four.
        let soup = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(feature_edges(&soup, &[0, 1, 2, 3, 4, 5]).len(), 4);
        // Folded 90° along the diagonal: the fold is an edge too.
        let folded = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(feature_edges(&folded, &[0, 1, 2, 0, 2, 3]).len(), 5);
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
        let text = "runmat 0.7\nstd 1\n\nassembly box {\n    use circuit daq as board {\n        placement = { placement = offset, offset = { translation = [15 mm, 10 mm, 10.5 mm] } }\n    }\n}\n";
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
