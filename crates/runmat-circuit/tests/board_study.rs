//! Board studies: `study.new` on a board writes a `.fea` study beside the layout that reads the
//! board through FEA's `pcb` source, in the form FEA set (sister-comms
//! 2026-11-10-design-to-fea-pcb-board-studies.md); and the source names every pad, so a study
//! puts IR-drop sources and sinks on `pcb/<board>/pad/<designator>.<pin>`.
//!
//! Until FEA's `pcb` reader lands, FEA's check refuses these studies' geometry; the Desktop
//! shows them under the "not yet in the current release" flag (ledger row
//! `design.ecad.board-studies`).

use runmat_circuit::fea::{canonical_id, pcb_source};
use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::json;
use std::path::PathBuf;

/// A drone board's folder, read whole into `/p/<board>/…`.
fn board_files(board: &str) -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-cad/wasm/tests/drone/electrical")
        .join(board);
    std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .map(|p| {
            (
                format!("/p/{board}/{}", p.file_name().unwrap().to_string_lossy()),
                std::fs::read_to_string(&p).unwrap(),
            )
        })
        .collect()
}

fn new_study(
    files: &Files,
    path: &str,
    physics: &str,
) -> Result<runtime::EditOutput, runtime::EditError> {
    let request = EditRequest {
        command: "study.new".into(),
        values: json!({ "physics": physics, "name": "study.fea" })
            .as_object()
            .cloned()
            .unwrap(),
        selection: Vec::new(),
        options: CompileOptions::default(),
    };
    runtime::edit(path, &files[path], files, &request)
}

/// The one new file a study.new run writes.
fn created(files: &Files, path: &str, physics: &str) -> (String, String) {
    let out =
        new_study(files, path, physics).unwrap_or_else(|e| panic!("{physics}: {}", e.message));
    assert_eq!(
        out.files.len(),
        1,
        "{physics}: one new file, the board untouched"
    );
    let file = &out.files[0];
    assert!(
        file.reprinted && file.edits.is_empty(),
        "{physics}: a whole new file"
    );
    (file.path.clone(), file.text.clone())
}

const ESC: &str = "/p/esc/esc.circuit.layout";
const FC: &str = "/p/flight_controller/flight_controller.circuit.layout";

#[test]
fn a_thermal_study_reads_the_board_with_its_component_blocks_and_power_map() {
    let files = board_files("esc");
    let (path, text) = created(&files, ESC, "thermal");
    assert_eq!(path, "/p/esc/esc_thermal.fea");
    for line in [
        "geometry esc {\n    pcb = \"./esc.circuit.layout\"\n    body = board\n    components = blocks\n",
        "    region board : solid { select name \"pcb/esc/board\"; expect exactly_one }\n",
        "    region top : face { select name \"pcb/esc/surface/top\"; expect exactly_one }\n",
        "    region bottom : face { select name \"pcb/esc/surface/bottom\"; expect exactly_one }\n",
        "    assign stackup from geometry.esc { treatment = homogenized }\n",
        "    load components { heat power_map from geometry.esc { unknown = 0 W } }\n",
        "            target = geometry.esc.region.top\n",
        "            target = geometry.esc.region.bottom\n",
        "study esc_thermal {\n    geometry = geometry.esc\n    mesh = mesh.esc\n    model = model.board_heat\n    analysis heat { thermal steady {} }\n",
        "    output junction_temperature = analysis.heat.pcb.junction_temperature\n",
    ] {
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }
    // The stackup is read, never copied: no material literal in the study.
    assert!(
        !text.contains("material ") && !text.contains("conductivity ="),
        "{text}"
    );
    // The same command from the schematic writes the same study.
    let (from_circuit, again) = created(&files, "/p/esc/esc.circuit", "thermal");
    assert_eq!((from_circuit, again), (path, text));
}

#[test]
fn an_ir_drop_study_holds_the_battery_pad_and_draws_current_at_the_farthest_bridge() {
    let files = board_files("esc");
    let (path, text) = created(&files, ESC, "ir_drop");
    assert_eq!(path, "/p/esc/esc_ir_drop.fea");
    assert!(text.contains("geometry esc_vbat {\n    pcb = \"./esc.circuit.layout\"\n    body = copper\n    nets = [\"VBAT\"]\n"), "{text}");
    assert!(
        text.contains(
            "    region source : face { select name \"pcb/esc/pad/P1.1\"; expect one_or_more }\n"
        ),
        "the battery pad is the source:\n{text}"
    );
    let load = text.lines().find(|l| l.contains("region load")).unwrap();
    let designator = load
        .split("pad/")
        .nth(1)
        .unwrap()
        .split('.')
        .next()
        .unwrap();
    assert!(
        designator.starts_with('Q'),
        "the load is a bridge FET: {load}"
    );
    // The pads the study names are the source's own pads.
    let board = parse_board(ESC, &files[ESC]).board.unwrap();
    let source = pcb_source(&board, None, None, &Default::default());
    for line in text.lines().filter(|l| l.contains("select name")) {
        let name = line.split('"').nth(1).unwrap();
        assert!(
            source.pads.iter().any(|p| p.region == name),
            "{name} is a pad of the board"
        );
    }
    for line in [
        "    electromagnetic { equation = steady_conduction }\n",
        "    assign stackup from geometry.esc_vbat { treatment = layered }\n",
        "electrical.potential { target = geometry.esc_vbat.region.source; value = 0 V }",
        "electrical.integral_current { target = geometry.esc_vbat.region.load; value = 1 A }",
        "    assessment within_rail_budget {\n        value = output.worst_drop\n        maximum = 100 mV\n",
    ] {
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }
}

#[test]
fn a_warpage_study_cools_a_layered_shell_from_reflow_and_checks_bow_and_twist() {
    let files = board_files("flight_controller");
    let (path, text) = created(&files, FC, "warpage");
    assert_eq!(path, "/p/flight_controller/flight_controller_warpage.fea");
    for line in [
        "    pcb = \"./flight_controller.circuit.layout\"\n    body = board\n    components = none\n",
        "    triangles {\n",
        "    thermomechanical { reference_temperature = 518.15 K }\n",
        "    assign stackup from geometry.flight_controller { treatment = layered }\n",
        "structural.three_two_one { target = geometry.flight_controller.region.board }",
        "temperature_field on geometry.flight_controller.region.board = 293.15 K",
        "    assessment bow_within_ipc_6012 {\n        value = output.bow\n        maximum = 0.0075\n",
        "    assessment twist_within_ipc_6012 {\n        value = output.twist\n        maximum = 0.0075\n",
    ] {
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }
}

#[test]
fn a_taken_name_and_a_part_physics_are_refused() {
    let mut files = board_files("esc");
    files.insert("/p/esc/esc_thermal.fea".into(), "// mine\n".into());
    let taken = new_study(&files, ESC, "thermal").unwrap_err();
    assert_eq!(
        taken.message,
        "esc_thermal.fea already exists; open it, or give the new study another name"
    );
    let modal = new_study(&files, ESC, "modal").unwrap_err();
    assert_eq!(modal.message, "a board study is Thermal, IR drop or Warpage; `modal` studies run on a part or an assembly");
}

#[test]
fn the_pcb_source_names_every_pad_by_designator_and_pin_on_each_of_its_layers() {
    let files = board_files("esc");
    let board = parse_board(ESC, &files[ESC]).board.unwrap();
    let source = pcb_source(&board, None, None, &Default::default());
    let fitted: usize = board
        .footprints
        .values()
        .filter(|f| f.fitted)
        .map(|f| f.pads(board.copper_count()).len())
        .sum();
    assert_eq!(source.pads.len(), fitted);
    for pad in &source.pads {
        assert_eq!(
            pad.region,
            format!("pcb/esc/pad/{}.{}", pad.designator, pad.pin)
        );
        assert_eq!(pad.id, canonical_id(&pad.region));
        assert!(
            !pad.layers.is_empty() && pad.layers.iter().all(|(_, area)| area.area() > 0.0),
            "{}",
            pad.region
        );
    }
    // The battery pad is a plated hole: its copper is on all six layers. A surface-mount FET
    // pad is on one outer layer.
    let battery = source
        .pads
        .iter()
        .find(|p| p.region == "pcb/esc/pad/P1.1")
        .unwrap();
    assert_eq!(battery.net.as_deref(), Some("VBAT"));
    assert_eq!(
        battery
            .layers
            .iter()
            .map(|(l, _)| l.as_str())
            .collect::<Vec<_>>(),
        ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"]
    );
    let fet = source
        .pads
        .iter()
        .find(|p| p.designator.starts_with('Q'))
        .unwrap();
    assert_eq!(fet.layers.len(), 1, "{}", fet.region);
    let surfaces: Vec<&str> = source.surfaces.iter().map(|s| s.region.as_str()).collect();
    assert_eq!(surfaces, ["pcb/esc/surface/top", "pcb/esc/surface/bottom"]);
}
