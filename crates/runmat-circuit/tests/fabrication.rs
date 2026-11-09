//! The fabrication set of the quadcopter's flight-controller board, checked with readers that
//! did not write it: every Gerber file parses with an independent Gerber parser, the Excellon
//! files and the IPC-D-356A netlist parse line by line against their formats, and the
//! IPC-2581 file validates against the IPC-2581C schema when `RUNMAT_IPC2581_XSD` names it
//! (with `xmllint`). `RUNMAT_FAB_OUT=<dir>` also writes the set there.

use gerber_parser::gerber_types::{Command, DCode, FunctionCode, Operation};
use runmat_circuit::fab::set::FabRequest;
use runmat_circuit::runtime::{export_fabrication, CompileOptions, FabricationExport, Files};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn drone() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-cad/wasm/tests/drone/electrical/flight_controller")
}

fn flight_controller() -> FabricationExport {
    flight_controller_with(&FabRequest::default())
}

fn flight_controller_with(request: &FabRequest) -> FabricationExport {
    let dir = drone();
    let files: Files = [".circuit", ".circuit.layout", ".circuit.lock"]
        .iter()
        .filter_map(|x| {
            let name = format!("flight_controller{x}");
            Some((
                format!("/p/{name}"),
                std::fs::read_to_string(dir.join(&name)).ok()?,
            ))
        })
        .collect();
    let path = "/p/flight_controller.circuit.layout";
    let out = export_fabrication(
        path,
        &files[path],
        &files,
        &CompileOptions::default(),
        request,
        None,
    )
    .unwrap_or_else(|d| panic!("{:?}", d.iter().map(|d| &d.message).collect::<Vec<_>>()));
    if let Ok(dir) = std::env::var("RUNMAT_FAB_OUT") {
        for (p, t) in &out.files {
            let f = PathBuf::from(&dir).join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, t).unwrap();
        }
    }
    out
}

#[derive(Default, Debug)]
struct Ops {
    flashes: usize,
    draws: usize,
    moves: usize,
}

fn parse_gerber(name: &str, text: &str) -> Ops {
    let doc = match gerber_parser::parse(std::io::BufReader::new(text.as_bytes())) {
        Ok(d) => d,
        Err((_, e)) => panic!("{name}: {e:?}"),
    };
    let errors: Vec<String> = doc.errors().iter().map(|e| format!("{e:?}")).collect();
    assert!(errors.is_empty(), "{name}: {errors:#?}");
    assert!(
        doc.units.is_some() && doc.format_specification.is_some(),
        "{name}: no units or format"
    );
    let mut ops = Ops::default();
    for c in doc.commands() {
        if let Command::FunctionCode(FunctionCode::DCode(DCode::Operation(op))) = c {
            match op {
                Operation::Flash(..) => ops.flashes += 1,
                Operation::Interpolate(..) => ops.draws += 1,
                Operation::Move(..) => ops.moves += 1,
            }
        }
    }
    ops
}

#[test]
fn every_gerber_parses_with_an_independent_reader() {
    let out = flight_controller();
    let gerbers: BTreeMap<&String, &String> = out
        .files
        .iter()
        .filter(|(p, _)| p.ends_with(".gbr"))
        .collect();
    // 4 copper, mask, paste and legend per side, profile; drill map; fabrication drawing; the
    // assembly drawing of the top (every part of this board is on top).
    assert_eq!(gerbers.len(), 4 + 6 + 1 + 2 + 1, "{:?}", gerbers.keys());
    let mut by_name = BTreeMap::new();
    for (p, t) in &gerbers {
        let ops = parse_gerber(p, t);
        // Every part sits on the top, so the bottom paste and legend are empty.
        if !p.ends_with("B_Paste.gbr") && !p.ends_with("B_Silkscreen.gbr") {
            assert!(ops.flashes + ops.draws > 0, "{p}: empty image");
        }
        by_name.insert(p.rsplit('/').next().unwrap().to_owned(), ops);
    }
    let top = &by_name["flight_controller-F_Cu.gbr"];
    assert!(top.flashes > 100, "top copper: {top:?}");
    // The drill map strokes its symbols: it draws more than the outline and text.
    assert!(by_name["flight_controller-drill-map.gbr"].draws > 300);
    // X2 attributes: every copper pad carries its net, pin and component.
    let cu = &out.files["gerber/flight_controller-F_Cu.gbr"];
    assert!(cu.contains("%TF.FileFunction,Copper,L1,Top*%"));
    assert!(cu.contains("%TO.P,U1,"), "pin attributes on the MCU");
    assert!(cu.contains("%TO.N,GND*%"));
    let job: serde_json::Value =
        serde_json::from_str(&out.files["gerber/flight_controller.gbrjob"]).unwrap();
    assert_eq!(job["GeneralSpecs"]["LayerNumber"], 4);
    let listed = job["FilesAttributes"].as_array().unwrap().len();
    let drills = out.files.keys().filter(|p| p.ends_with(".drl")).count();
    assert_eq!(listed, 4 + 6 + 1 + drills);
}

/// Excellon: a header (`M48` … `%`), tool definitions `T<n>C<d>`, then hits under each tool,
/// `M30` at the end. Returns tool → hits.
fn parse_excellon(name: &str, text: &str) -> BTreeMap<u32, usize> {
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("M48"), "{name}");
    let mut tools = BTreeMap::new();
    let mut header = true;
    let mut current = None;
    let mut ended = false;
    for l in lines {
        if l.starts_with(';') || l.is_empty() {
            continue;
        }
        assert!(!ended, "{name}: data after M30");
        if header {
            match l {
                "%" => header = false,
                "FMAT,2" | "METRIC" | "G90" => {}
                t if t.starts_with('T') => {
                    let (n, d) = t[1..]
                        .split_once('C')
                        .unwrap_or_else(|| panic!("{name}: {t}"));
                    let d: f64 = d.parse().unwrap();
                    assert!(d > 0.0);
                    tools.insert(n.parse::<u32>().unwrap(), 0usize);
                }
                other => panic!("{name}: header line {other}"),
            }
            continue;
        }
        match l {
            "G05" => {}
            "M30" => ended = true,
            t if t.starts_with('T') => current = Some(t[1..].parse::<u32>().unwrap()),
            hit if hit.starts_with('X') => {
                let t = current.unwrap_or_else(|| panic!("{name}: hit before a tool"));
                let coords: Vec<&str> = hit.split("G85").collect();
                for c in coords {
                    let (x, y) = c[1..]
                        .split_once('Y')
                        .unwrap_or_else(|| panic!("{name}: {c}"));
                    x.parse::<f64>().unwrap();
                    y.parse::<f64>().unwrap();
                }
                *tools
                    .get_mut(&t)
                    .unwrap_or_else(|| panic!("{name}: undefined tool T{t}")) += 1;
            }
            other => panic!("{name}: body line {other}"),
        }
    }
    assert!(ended, "{name}: no M30");
    tools
}

#[test]
fn drill_files_and_the_drill_table_agree() {
    let out = flight_controller();
    let mut hits = 0;
    for (p, t) in out.files.iter().filter(|(p, _)| p.ends_with(".drl")) {
        let tools = parse_excellon(p, t);
        assert!(tools.values().all(|&n| n > 0), "{p}: an unused tool");
        hits += tools.values().sum::<usize>();
    }
    assert!(hits > 0);
    let map = &out.files["drill/flight_controller-drill-map.gbr"];
    assert!(map.contains("%TF.FileFunction,Drillmap*%"));
}

#[test]
fn ipc356_records_are_fixed_column() {
    let out = flight_controller();
    let t = &out.files["netlist/flight_controller.ipc"];
    let mut records = 0;
    for l in t.lines() {
        match &l[..1] {
            "C" | "P" => {}
            "9" => assert_eq!(l, "999"),
            "3" => {
                records += 1;
                assert!(l.starts_with("317") || l.starts_with("327"), "{l}");
                // Net name in columns 4-17, then reference at 21, X at 42 and Y at 50 (1-based).
                assert_eq!(&l[41..42], "X", "{l}");
                assert_eq!(&l[49..50], "Y", "{l}");
            }
            other => panic!("unexpected record {other}: {l}"),
        }
    }
    assert!(records > 100);
}

#[test]
fn ipc2581_validates_against_the_schema() {
    let out = flight_controller();
    let xml = &out.files["ipc2581/flight_controller.xml"];
    assert!(xml.contains("<IPC-2581 revision=\"C\""));
    let Ok(xsd) = std::env::var("RUNMAT_IPC2581_XSD") else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("runmat-ipc2581-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("board.xml");
    std::fs::write(&f, xml).unwrap();
    let r = std::process::Command::new("xmllint")
        .args(["--noout", "--schema", &xsd])
        .arg(&f)
        .output()
        .expect("xmllint");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
}

#[test]
fn odb_tree_is_consistent() {
    let out = flight_controller();
    let odb: BTreeMap<&str, &String> = out
        .files
        .iter()
        .filter_map(|(p, t)| Some((p.strip_prefix("odb/")?, t)))
        .collect();
    let matrix = odb["flight_controller/matrix/matrix"];
    let names: Vec<String> = matrix
        .lines()
        .filter_map(|l| l.trim().strip_prefix("NAME="))
        .filter(|n| *n != "PCB")
        .map(|n| n.to_lowercase())
        .collect();
    assert!(names.len() >= 4 + 6 + 2 + 1, "{names:?}");
    for n in &names {
        let f = format!("flight_controller/steps/pcb/layers/{n}/features");
        let t = odb.get(f.as_str()).unwrap_or_else(|| panic!("no {f}"));
        assert!(t.starts_with("UNITS=MM"));
        // Every symbol a feature uses is defined.
        let defined: usize = t.lines().filter(|l| l.starts_with('$')).count();
        for l in t
            .lines()
            .filter(|l| l.starts_with("P ") || l.starts_with("L "))
        {
            let f: Vec<&str> = l.split(' ').collect();
            let sym: usize = f[if l.starts_with('P') { 3 } else { 5 }].parse().unwrap();
            assert!(sym < defined, "{n}: {l}");
        }
    }
    let eda = odb["flight_controller/steps/pcb/eda/data"];
    assert!(eda.lines().filter(|l| l.starts_with("NET ")).count() > 20);
    assert!(eda.contains("FID C "));
    let comps = odb["flight_controller/steps/pcb/layers/comp_+_top/components"];
    assert!(comps
        .lines()
        .any(|l| l.starts_with("CMP ") && l.contains(" U1 ")));
}

#[test]
fn bom_and_placement_per_variant() {
    let out = flight_controller();
    let bom = &out.files["assembly/flight_controller-bom.csv"];
    assert!(bom.starts_with("Designators,Quantity,Value,Package,MPN,Manufacturer\n"));
    let pos = &out.files["assembly/flight_controller-pos.csv"];
    assert!(pos.lines().count() > 20);
    for v in out.variants.iter().flatten() {
        assert!(out
            .files
            .contains_key(&format!("assembly/flight_controller-{v}-bom.csv")));
    }
    assert_eq!(out.layers.iter().filter(|l| l.kind == "copper").count(), 4);
}

#[test]
fn the_set_is_deterministic() {
    assert_eq!(flight_controller().files, flight_controller().files);
}

#[test]
fn the_layer_table_follows_the_stackup_and_settings_pick_the_gerbers() {
    use runmat_circuit::fab::plan::layer_rows;
    use runmat_circuit::fab::settings::{FabPreset, FabSettings};
    let out = flight_controller();
    let rows = layer_rows(&out.filled);
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(&names[..4], ["F.Paste", "F.Silkscreen", "F.Mask", "F.Cu"]);
    assert_eq!(names.last(), Some(&"Edge.Cuts"));
    let copper: Vec<&runmat_circuit::fab::plan::LayerRow> =
        rows.iter().filter(|r| r.kind == "copper").collect();
    assert_eq!(copper.len(), 4);
    assert!(copper
        .iter()
        .all(|r| r.copper_oz.is_some() && r.thickness_mm.is_some() && r.file.is_some()));
    assert!(rows
        .iter()
        .filter(|r| r.kind == "core" || r.kind == "prepreg")
        .all(|r| r.file.is_none() && r.thickness_mm.is_some()));
    assert!(rows
        .iter()
        .filter(|r| r.kind == "courtyard")
        .all(|r| !r.default_on));
    // A fab-house set with the paste layers left out and the top courtyard added.
    let mut settings = FabSettings {
        preset: FabPreset::FabHouse,
        ..Default::default()
    };
    let mut layers: std::collections::BTreeSet<String> = rows
        .iter()
        .filter(|r| r.default_on && r.kind != "paste")
        .map(|r| r.name.clone())
        .collect();
    layers.insert("F.Courtyard".into());
    settings.layers = Some(layers);
    let files = runmat_circuit::fab::set::fabrication(
        &out.filled,
        &[],
        &Default::default(),
        &settings.request(),
    );
    assert!(files.keys().any(|k| k.ends_with("-F_Courtyard.gbr")));
    assert!(!files.keys().any(|k| k.ends_with("_Paste.gbr")));
    assert!(!files.keys().any(|k| k.starts_with("assembly/")));
    let job = &files["gerber/flight_controller.gbrjob"];
    assert!(job.contains("F_Courtyard") && !job.contains("F_Paste"));
}

#[test]
fn the_layout_keeps_its_fabrication_settings() {
    use runmat_circuit::fab::settings::{FabOutput, FabPreset};
    use runmat_circuit::syntax::board::{parse_board, print_board};
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/projects/daq");
    let text = std::fs::read_to_string(dir.join("daq.circuit.layout")).unwrap();
    let parsed = parse_board("/p/daq.circuit.layout", &text);
    let mut board = parsed
        .board
        .unwrap_or_else(|| panic!("{:?}", parsed.diagnostics));
    let mut settings = runmat_circuit::fab::settings::FabSettings {
        preset: FabPreset::Custom,
        ..Default::default()
    };
    settings.outputs.insert(FabOutput::Odb);
    settings.layers = Some(["F.Cu".to_string(), "Edge.Cuts".into()].into());
    settings.variants = vec!["".into(), "lite".into()];
    settings.folder = Some("fab/fc-A".into());
    settings.include_through_hole = true;
    settings.revision = Some("B".into());
    settings.part_number = Some("FC-210".into());
    settings.gerber = runmat_circuit::fab::GerberFormat {
        units: runmat_circuit::fab::LengthUnit::Inch,
        decimals: 5,
    };
    settings.pick_place.sides = runmat_circuit::fab::assembly::PlaceSides::Separate;
    settings.bom.columns = vec![
        runmat_circuit::fab::assembly::BomColumn::Designators,
        runmat_circuit::fab::assembly::BomColumn::Mpn,
    ];
    settings.bom.group_by_value = false;
    board.fabrication = Some(settings.clone());
    let printed = print_board(&board, &[]);
    assert!(printed.contains("fabrication = { preset = custom, outputs = [gerber, drill, drill_map, drawing, ipc356, odb]"), "{}", printed.lines().find(|l| l.contains("fabrication")).unwrap_or(""));
    let again = parse_board("/p/daq.circuit.layout", &printed);
    assert!(again.diagnostics.is_empty(), "{:?}", again.diagnostics);
    assert_eq!(again.board.unwrap().fabrication, Some(settings.clone()));
    // Written into the file in place: once added, then replaced, the rest kept.
    let added = runmat_circuit::syntax::board::with_fabrication_settings(&text, &settings).unwrap();
    let replaced =
        runmat_circuit::syntax::board::with_fabrication_settings(&added, &Default::default())
            .unwrap();
    assert_eq!(replaced.matches("fabrication = ").count(), 1);
    assert_eq!(
        replaced.len() - text.len(),
        replaced
            .lines()
            .find(|l| l.contains("fabrication = "))
            .unwrap()
            .len()
            + 1
    );
    let back = parse_board("/p/daq.circuit.layout", &added).board.unwrap();
    assert_eq!(back.fabrication, Some(settings));
}

#[test]
fn gerber_units_and_decimals_set_the_coordinate_format() {
    use runmat_circuit::fab::{GerberFormat, LengthUnit};
    let out = flight_controller();
    let mut req = FabRequest {
        revision: Some("B".into()),
        ..Default::default()
    };
    req.gerber_format = GerberFormat {
        units: LengthUnit::Inch,
        decimals: 5,
    };
    let files = runmat_circuit::fab::set::fabrication(&out.filled, &[], &Default::default(), &req);
    let inch = &files["gerber/flight_controller-F_Cu.gbr"];
    assert!(inch.contains("%FSLAX25Y25*%\n%MOIN*%"), "{}", &inch[..400]);
    assert!(
        inch.contains(",B*%"),
        "the revision in the project attribute"
    );
    assert!(files["gerber/flight_controller.gbrjob"].contains("\"Revision\": \"B\""));
    // The independent reader takes it, and the image is the same: the outline's extent in
    // inches is the millimetre extent / 25.4, within the 10 µin resolution.
    let mm = &out.files["gerber/flight_controller-Edge_Cuts.gbr"];
    let span = |text: &str| {
        let xs: Vec<i64> = text
            .lines()
            .filter_map(|l| l.strip_prefix('X')?.split('Y').next()?.parse().ok())
            .collect();
        (xs.iter().max().unwrap() - xs.iter().min().unwrap()) as f64
    };
    for (name, text) in &files {
        if name.ends_with(".gbr") {
            parse_gerber(name, text);
        }
    }
    let w_mm = span(mm) / 1e6;
    let w_in = span(&files["gerber/flight_controller-Edge_Cuts.gbr"]) / 1e5;
    assert!((w_mm / 25.4 - w_in).abs() < 2e-5, "{w_mm} mm vs {w_in} in");
    // The drill map and drawing follow the same format.
    assert!(files["drill/flight_controller-drill-map.gbr"].contains("%MOIN*%"));
}

#[test]
fn pick_and_place_sides_and_units_and_bom_columns() {
    use runmat_circuit::fab::assembly::{BomColumn, PlaceSides};
    use runmat_circuit::fab::LengthUnit;
    let base = flight_controller().files;
    let mut req = FabRequest::default();
    req.pick_place_format.sides = PlaceSides::Separate;
    req.pick_place_format.units = LengthUnit::Inch;
    req.bom_format.columns = vec![BomColumn::Designators, BomColumn::Value];
    req.bom_format.group_by_value = false;
    let files = flight_controller_with(&req).files;
    let both = &base["assembly/flight_controller-pos.csv"];
    let top = &files["assembly/flight_controller-pos-top.csv"];
    let bottom = &files["assembly/flight_controller-pos-bottom.csv"];
    assert!(!files.contains_key("assembly/flight_controller-pos.csv"));
    assert!(top.starts_with("Designator,Value,Package,X_in,Y_in,Rotation,Side\n"));
    // Every part of this board is on top: the top file lists them all, the bottom none.
    assert_eq!(top.lines().count(), both.lines().count());
    assert_eq!(bottom.lines().count(), 1);
    let first = |t: &str, col: usize| -> f64 {
        t.lines()
            .nth(1)
            .unwrap()
            .split(',')
            .nth(col)
            .unwrap()
            .parse()
            .unwrap()
    };
    assert!((first(both, 3) / 25.4 - first(top, 3)).abs() < 1e-5);
    let bom = &files["assembly/flight_controller-bom.csv"];
    let grouped = &base["assembly/flight_controller-bom.csv"];
    assert!(bom.starts_with("Designators,Value\n"));
    // Ungrouped: one line per designator, as many as the grouped quantities add up to.
    let parts: usize = grouped
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(1).unwrap().parse::<usize>().unwrap())
        .sum();
    assert_eq!(bom.lines().count() - 1, parts);
    assert!(bom
        .lines()
        .skip(1)
        .all(|l| !l.split(',').next().unwrap().contains(' ')));
}

#[test]
fn the_default_folder_names_the_revision() {
    let mut s = runmat_circuit::fab::settings::FabSettings::default();
    assert_eq!(s.folder_or_default("fc"), "fab/fc");
    s.revision = Some("B".into());
    assert_eq!(s.folder_or_default("fc"), "fab/fc-B");
    s.folder = Some("out".into());
    assert_eq!(s.folder_or_default("fc"), "out");
}

#[test]
fn the_fab_drawing_names_the_revision_and_its_dimensions_have_arrowheads() {
    use runmat_circuit::fab::drawing::{fabrication_drawing, Mark};
    let board = flight_controller().filled;
    let texts = |rev: Option<&str>| -> Vec<String> {
        fabrication_drawing(&board, rev)
            .marks
            .iter()
            .filter_map(|m| match m {
                Mark::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    };
    let title = texts(Some("b"))
        .into_iter()
        .find(|t| t.contains("FABRICATION DRAWING"))
        .unwrap();
    assert!(title.ends_with("REV B"), "{title}");
    assert!(!texts(None).iter().any(|t| t.contains("REV")));
    // Each end of the two overall-size dimension lines: a V of two strokes whose tip is the
    // line's end on the board's extent.
    let bb = board.outline.bbox();
    let tips: Vec<_> = fabrication_drawing(&board, None)
        .marks
        .iter()
        .filter_map(|m| match m {
            Mark::Line { points, .. } if points.len() == 3 => Some(points[1]),
            _ => None,
        })
        .collect();
    let on_extent = tips
        .iter()
        .filter(|p| p.x == bb.min.x || p.x == bb.max.x || p.y == bb.min.y || p.y == bb.max.y)
        .count();
    assert!(on_extent >= 4, "{tips:?}");
}

#[test]
fn odb_legend_carries_silk_text_once_as_the_gerber_strokes() {
    use runmat_circuit::fab::{gerber, odb::odb, GerberLayer};
    use runmat_circuit::geom::{mm, Point};
    use runmat_circuit::layout::{BoardText, ItemId, Layer, Side};
    let plain = flight_controller().filled;
    let mut lettered = plain.clone();
    let id = ItemId(u64::MAX);
    lettered.texts.insert(
        id,
        BoardText {
            id,
            layer: Layer::Silk(Side::Top),
            at: Point::new(mm(2.0), mm(2.0)),
            text: "RUNMAT FC".into(),
            size: mm(1.0),
            thickness: 0,
            angle: 0.0,
            mirrored: false,
        },
    );
    // Strokes drawn outside regions: D01 in the Gerber, L records in ODB++.
    let gerber_draws = |b| {
        let mut region = false;
        gerber(b, GerberLayer::Legend(Side::Top))
            .lines()
            .filter(|l| {
                region = (region || l.starts_with("G36")) && !l.starts_with("G37");
                !region && l.ends_with("D01*")
            })
            .count()
    };
    let odb_legend = |b| {
        odb(b, &BTreeMap::new())["flight_controller/steps/pcb/layers/top_legend/features"].clone()
    };
    let lines = |t: &str| t.lines().filter(|l| l.starts_with("L ")).count();
    let added = gerber_draws(&lettered) - gerber_draws(&plain);
    assert!(added > 0);
    let legend = odb_legend(&lettered);
    assert_eq!(lines(&legend) - lines(&odb_legend(&plain)), added);
    assert!(
        !legend.lines().any(|l| l.starts_with("T ")),
        "text drawn twice"
    );
}
