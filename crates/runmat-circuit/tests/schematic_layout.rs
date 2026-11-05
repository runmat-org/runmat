//! The schematic auto-layout on the showcase (`tests/projects/daq`) and the small io board:
//! wires never cross a symbol, texts never overlap, every pin is wired, labelled, given a
//! power port or marked no-connect, signal flows left to right, and recorded places stick
//! through an edit. Set `RUNMAT_SCHEMATIC_SVG` to a folder to write each page as SVG.

use runmat_circuit::runtime::{compile, edit, CompileOptions, EditRequest, Files};
use runmat_circuit::schematic::{SchematicIndex, SchematicSheet};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn project(name: &str) -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects").join(name);
    ["circuit", "circuit.layout", "circuit.lock"]
        .iter()
        .filter_map(|ext| {
            let f = format!("{name}.{ext}");
            std::fs::read_to_string(dir.join(&f))
                .ok()
                .map(|t| (format!("/p/{f}"), t))
        })
        .collect()
}

fn drawn(files: &Files, circuit: &str) -> SchematicIndex {
    let out = compile(circuit, &files[circuit], files, &CompileOptions::default());
    let index = out.index.unwrap_or_else(|| panic!("compiles: {:?}", out.diagnostics));
    index.schematic
}

const EPS: f64 = 1e-6;

fn overlaps(a: &[f64; 4], b: &[f64; 4], shrink: f64) -> bool {
    a[0] + shrink < b[2] && b[0] + shrink < a[2] && a[1] + shrink < b[3] && b[1] + shrink < a[3]
}

/// Whether segment `p`–`q` (axis-aligned) passes through the inside of `r`.
fn crosses(p: [f64; 2], q: [f64; 2], r: &[f64; 4]) -> bool {
    let (x0, x1) = (p[0].min(q[0]), p[0].max(q[0]));
    let (y0, y1) = (p[1].min(q[1]), p[1].max(q[1]));
    x1 > r[0] + EPS && x0 < r[2] - EPS && y1 > r[1] + EPS && y0 < r[3] - EPS
}

fn body_box(s: &runmat_circuit::schematic::SchematicSymbol) -> [f64; 4] {
    s.body[0].iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| {
        [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])]
    })
}

fn check_page(page: &SchematicSheet) {
    // Wires: orthogonal, never through a body or a sheet symbol.
    let mut bodies: Vec<(String, [f64; 4])> = page
        .symbols
        .iter()
        .map(|s| (s.designator.clone(), body_box(s)))
        .collect();
    for s in &page.sheet_symbols {
        bodies.push((s.path.clone(), [s.x_mm, s.y_mm, s.x_mm + s.w_mm, s.y_mm + s.h_mm]));
    }
    for w in &page.wires {
        for seg in w.points.windows(2) {
            let (p, q) = (seg[0], seg[1]);
            assert!(
                (p[0] - q[0]).abs() < EPS || (p[1] - q[1]).abs() < EPS,
                "{}: wire {} is not orthogonal: {p:?} {q:?}",
                page.path,
                w.net
            );
            for (name, b) in &bodies {
                assert!(
                    !crosses(p, q, b),
                    "{}: wire {} crosses {name}: {p:?}–{q:?} through {b:?}",
                    page.path,
                    w.net
                );
            }
        }
    }
    // Texts: no two overlap (symbol texts and labels).
    let mut texts: Vec<(String, [f64; 4])> = Vec::new();
    for s in &page.symbols {
        for (k, t) in runmat_circuit::schematic::symbol_texts(s).into_iter().enumerate() {
            texts.push((format!("{} text {k}", s.designator), t));
        }
    }
    for l in &page.labels {
        texts.push((format!("label {} at {},{}", l.net, l.x_mm, l.y_mm), l.extent()));
    }
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            assert!(
                !overlaps(&texts[i].1, &texts[j].1, 0.05),
                "{}: {} overlaps {} ({:?} / {:?})",
                page.path,
                texts[i].0,
                texts[j].0,
                texts[i].1,
                texts[j].1
            );
        }
    }
    // Every pin: wired, labelled, a port, or marked no-connect.
    let key = |p: [f64; 2]| ((p[0] * 1e3).round() as i64, (p[1] * 1e3).round() as i64);
    let mut ends: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for w in &page.wires {
        for p in [w.points[0], *w.points.last().unwrap()] {
            *ends.entry(key(p)).or_default() += 1;
        }
    }
    let ports: Vec<(i64, i64)> = page.labels.iter().map(|l| key([l.x_mm, l.y_mm])).collect();
    let ncs: Vec<(i64, i64)> = page.no_connects.iter().map(|p| key(*p)).collect();
    for s in &page.symbols {
        for p in &s.pins {
            let k = key([p.x_mm, p.y_mm]);
            if p.net.is_some() {
                assert!(
                    ends.contains_key(&k) || ports.contains(&k),
                    "{}: {}.{} is left unconnected",
                    page.path,
                    s.designator,
                    p.name
                );
            } else {
                assert!(ncs.contains(&k), "{}: {}.{} has no no-connect mark", page.path, s.designator, p.name);
            }
        }
        assert!((s.pin_text_mm - 1.27).abs() < EPS && (s.designator_text_mm - 1.524).abs() < EPS);
    }
}

fn svg(page: &SchematicSheet) -> String {
    let h = page.height_mm;
    let y = |v: f64| h - v;
    let mut s = format!(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {} {}' width='{}' height='{}' font-family='Helvetica'><rect width='100%' height='100%' fill='white'/>",
        page.width_mm,
        h,
        page.width_mm * 4.0,
        h * 4.0
    );
    let line = |pts: &[[f64; 2]], c: &str, w: f64| {
        let d: Vec<String> = pts.iter().map(|p| format!("{},{}", p[0], y(p[1]))).collect();
        format!("<polyline points='{}' fill='none' stroke='{c}' stroke-width='{w}'/>", d.join(" "))
    };
    let text = |t: &str, at: [f64; 2], size: f64, anchor: &str, c: &str| {
        format!(
            "<text x='{}' y='{}' font-size='{size}' text-anchor='{anchor}' fill='{c}'>{}</text>",
            at[0],
            y(at[1]),
            t.replace('&', "&amp;").replace('<', "&lt;")
        )
    };
    for sym in &page.symbols {
        if sym.art == "box" || sym.art == "resistor" {
            s += &line(&sym.body[0], "#a00", 0.25);
        }
        for b in &sym.body[1..] {
            s += &line(b, "#a00", 0.15);
        }
        for b in &sym.strokes {
            s += &line(b, "#a00", 0.25);
        }
        s += &text(&runmat_circuit::schematic::designator_text(sym), sym.designator_at, 1.524, &sym.text_anchor, "#000");
        if let Some(v) = &sym.value {
            s += &text(v, sym.value_at, 1.27, &sym.text_anchor, "#555");
        }
        for p in &sym.pins {
            if !p.hidden {
                s += &text(&p.name, p.name_at, 1.27, &p.name_anchor, "#036");
            }
        }
    }
    for sh in &page.sheet_symbols {
        s += &format!(
            "<rect x='{}' y='{}' width='{}' height='{}' fill='#fef' stroke='#808' stroke-width='0.25'/>",
            sh.x_mm,
            y(sh.y_mm + sh.h_mm),
            sh.w_mm,
            sh.h_mm
        );
        s += &text(&sh.path, [sh.x_mm, sh.y_mm + sh.h_mm + 0.8], 1.6, "start", "#808");
        for p in &sh.pins {
            let a = if p.side == "left" { "start" } else { "end" };
            let dx = if p.side == "left" { 1.2 } else { -1.2 };
            s += &text(&p.name, [p.x_mm + dx, p.y_mm - 0.45], 1.27, a, "#808");
        }
    }
    for w in &page.wires {
        s += &line(&w.points, "#070", 0.2);
    }
    for j in &page.junctions {
        s += &format!("<circle cx='{}' cy='{}' r='0.5' fill='#070'/>", j[0], y(j[1]));
    }
    for n in &page.no_connects {
        s += &line(&[[n[0] - 0.8, n[1] - 0.8], [n[0] + 0.8, n[1] + 0.8]], "#00a", 0.2);
        s += &line(&[[n[0] - 0.8, n[1] + 0.8], [n[0] + 0.8, n[1] - 0.8]], "#00a", 0.2);
    }
    for l in &page.labels {
        let e = l.extent();
        s += &format!(
            "<rect x='{}' y='{}' width='{}' height='{}' fill='none' stroke='#c60' stroke-width='0.1'/>",
            e[0],
            y(e[3]),
            e[2] - e[0],
            e[3] - e[1]
        );
        let name = &l.text;
        let (at, a) = match l.side.as_str() {
            "left" => ([e[0] + 0.3, l.y_mm - 0.45], "start"),
            "right" => ([e[2] - 0.3, l.y_mm - 0.45], "end"),
            "up" => ([l.x_mm, e[3] - 1.0], "middle"),
            _ => ([l.x_mm, e[1] + 0.3], "middle"),
        };
        s += &text(name, at, 1.27, a, "#c60");
    }
    s + "</svg>"
}

fn dump(name: &str, index: &SchematicIndex) {
    let Ok(dir) = std::env::var("RUNMAT_SCHEMATIC_SVG") else {
        return;
    };
    for page in &index.sheets {
        let file = format!("{name}{}.svg", page.path.replace(['/', '[', ']'], "_"));
        std::fs::write(PathBuf::from(&dir).join(file), svg(page)).unwrap();
    }
}

#[test]
fn the_showcase_schematic_is_drawn_to_the_bar() {
    let files = project("daq");
    let index = drawn(&files, "/p/daq.circuit");
    dump("daq", &index);
    for page in &index.sheets {
        check_page(page);
    }
    // The channel: signal flow left to right, the ADC right of the amplifier, real wires.
    let ch = index.sheets.iter().find(|s| s.path == "/ch[0]").expect("a channel page");
    let x_of = |leaf: &str| {
        ch.symbols
            .iter()
            .find(|s| s.path.ends_with(&format!("/{leaf}")))
            .map(|s| s.x_mm)
            .unwrap()
    };
    assert!(x_of("RS") < x_of("U") && x_of("U") < x_of("RO") && x_of("RO") < x_of("ADC"));
    let wired: std::collections::BTreeSet<&str> = ch.wires.iter().map(|w| w.net.as_str()).collect();
    for net in ["INF", "FB", "AMP", "AIN"] {
        assert!(wired.iter().any(|n| n.ends_with(net)), "{net} is wired: {wired:?}");
    }
    assert!(!ch.junctions.is_empty(), "T joins get junction dots");
    assert!(ch.labels.iter().any(|l| l.kind == "power" && l.side == "up"));
    assert!(ch.labels.iter().any(|l| l.kind == "power" && l.side == "down"));
    assert!(ch.labels.iter().any(|l| l.kind == "hier"), "ports leave as hierarchical labels");
    // The top sheet: the connector feeds the channels, the channels feed the FPGA.
    let root = index.sheets.iter().find(|s| s.path == "/").unwrap();
    let j2 = root.symbols.iter().find(|s| s.designator == "J2").unwrap().x_mm;
    let ch0 = root.sheet_symbols.iter().find(|s| s.path == "/ch[0]").unwrap().x_mm;
    let fpga = root
        .symbols
        .iter()
        .find(|s| s.designator == "U1" && s.unit.as_deref() == Some("A"))
        .unwrap()
        .x_mm;
    assert!(j2 < ch0 && ch0 < fpga, "J2 {j2} → ch {ch0} → U1A {fpga}");
    assert!(
        root.wires.iter().any(|w| w.net.ends_with("SDO0")),
        "the channels' SDO is wired to the FPGA"
    );
    // Same design, same bytes.
    assert_eq!(index, drawn(&files, "/p/daq.circuit"));
}

#[test]
fn the_io_schematic_is_drawn_to_the_bar() {
    let files = project("io");
    let index = drawn(&files, "/p/io.circuit");
    dump("io", &index);
    for page in &index.sheets {
        check_page(page);
    }
}

/// Annotating records the places; an added part goes to free space and nothing else moves.
#[test]
fn recorded_places_stick_through_an_edit() {
    let mut files = project("daq");
    let circuit = "/p/daq.circuit";
    let lock = "/p/daq.circuit.lock";
    // Start from a lock with no places, so annotating records them.
    let bare: String = files[lock]
        .lines()
        .filter(|l| !l.starts_with("place "))
        .map(|l| format!("{l}\n"))
        .collect();
    files.insert(lock.into(), bare);
    let request = |command: &str, values: serde_json::Value| EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap(),
        selection: Vec::new(),
        options: CompileOptions::default(),
    };
    let out = edit(circuit, &files[circuit], &files, &request("schematic.annotate", serde_json::json!({})))
        .expect("annotates");
    let written = out.files.iter().find(|f| f.path == lock).expect("the lock records places");
    assert!(written.text.contains("\nplace channel/RS "), "{}", written.text);
    files.insert(lock.into(), written.text.clone());
    let before = drawn(&files, circuit);
    let placed = |index: &SchematicIndex| -> BTreeMap<String, (f64, f64, f64)> {
        index
            .sheets
            .iter()
            .flat_map(|s| s.symbols.iter().map(move |y| (format!("{}|{}", s.path, y.key), (y.x_mm, y.y_mm, y.rotation))))
            .collect()
    };
    // One edit: a pull-down on the amplifier's input and a new part on the top sheet.
    let text = files[circuit]
        .replace(
            "        component CREF = { part = \"C0805-10U\" }",
            "        component CREF = { part = \"C0805-10U\" }\n        component RP = { part = \"R0402-100K\" }",
        )
        .replace(
            "        connect = [net.VREF, ADC.REF, \"CREF.1\"]",
            "        connect = [net.VREF, ADC.REF, \"CREF.1\"]\n        connect = [net.INF, \"RP.1\"]\n        connect = [net.GND, \"RP.2\"]",
        );
    assert_ne!(text, files[circuit]);
    files.insert(circuit.into(), text);
    let after = drawn(&files, circuit);
    let (a, b) = (placed(&before), placed(&after));
    for (k, v) in &a {
        assert_eq!(b.get(k), Some(v), "{k} moved");
    }
    assert_eq!(b.len(), a.len() + 3, "the new part on each channel page");
    for page in &after.sheets {
        check_page(page);
    }
    // A move sticks too.
    let moved = edit(
        circuit,
        &files[circuit],
        &files,
        &EditRequest {
            selection: vec!["component:J2".into()],
            ..request("schematic.move", serde_json::json!({ "dx": 2.54, "dy": -5.08 }))
        },
    )
    .expect("moves");
    let lock_text = &moved.files.iter().find(|f| f.path == lock).unwrap().text;
    files.insert(lock.into(), lock_text.clone());
    let again = drawn(&files, circuit);
    let j2 = |i: &SchematicIndex| {
        let s = i.sheets[0].symbols.iter().find(|s| s.designator == "J2").unwrap();
        (s.x_mm, s.y_mm)
    };
    let (x0, y0) = j2(&after);
    let (x1, y1) = j2(&again);
    assert!((x1 - x0 - 2.54).abs() < 1e-6 && (y1 - y0 + 5.08).abs() < 1e-6, "{x0},{y0} → {x1},{y1}");
}

/// The layout is part of every compile, so the showcase schematic (without its board) compiles
/// well inside the Desktop's 0.5 s edit budget (an optimised build; debug builds get more room).
#[test]
fn the_layout_is_fast() {
    let mut files = project("daq");
    files.remove("/p/daq.circuit.layout");
    let run = || compile("/p/daq.circuit", &files["/p/daq.circuit"], &files, &CompileOptions::default());
    run();
    let start = std::time::Instant::now();
    let runs: usize = std::env::var("RUNMAT_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    for _ in 0..runs {
        std::hint::black_box(run());
    }
    let each = start.elapsed().as_secs_f64() / runs as f64;
    let budget = if cfg!(debug_assertions) { 2.0 } else { 0.1 };
    eprintln!("schematic compile: {:.1} ms", each * 1e3);
    assert!(each < budget, "the compile took {each:.3} s");
}
