//! The schematic auto-layout on the showcase (`tests/projects/daq`) and the small io board:
//! wires never cross a symbol, texts never overlap, every pin is wired, labelled or given a
//! power port, crosses mark only no-connect pins, signal flows left to right, and recorded
//! places stick through an edit. Set `RUNMAT_SCHEMATIC_SVG` to a folder to write each page as SVG.

use runmat_circuit::runtime::{compile, edit, CompileOptions, EditRequest, Files};
use runmat_circuit::schematic::{SchematicIndex, SchematicSheet};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn project(name: &str) -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/projects")
        .join(name);
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
    let index = out
        .index
        .unwrap_or_else(|| panic!("compiles: {:?}", out.diagnostics));
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
    s.body[0]
        .iter()
        .fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| {
            [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ]
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
        bodies.push((
            s.path.clone(),
            [s.x_mm, s.y_mm, s.x_mm + s.w_mm, s.y_mm + s.h_mm],
        ));
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
        for (k, t) in runmat_circuit::schematic::symbol_texts(s)
            .into_iter()
            .enumerate()
        {
            texts.push((format!("{} text {k}", s.designator), t));
        }
    }
    for l in &page.labels {
        texts.push((
            format!("label {} at {},{}", l.net, l.x_mm, l.y_mm),
            l.extent(),
        ));
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
                // Crossed only when marked no-connect; other open pins are left for ERC.
                assert_eq!(
                    ncs.contains(&k),
                    p.no_connect,
                    "{}: {}.{} no-connect cross",
                    page.path,
                    s.designator,
                    p.name
                );
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
        let d: Vec<String> = pts
            .iter()
            .map(|p| format!("{},{}", p[0], y(p[1])))
            .collect();
        format!(
            "<polyline points='{}' fill='none' stroke='{c}' stroke-width='{w}'/>",
            d.join(" ")
        )
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
        s += &text(
            &runmat_circuit::schematic::designator_text(sym),
            sym.designator_at,
            1.524,
            &sym.text_anchor,
            "#000",
        );
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
        s += &text(
            &sh.path,
            [sh.x_mm, sh.y_mm + sh.h_mm + 0.8],
            1.6,
            "start",
            "#808",
        );
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
        s += &format!(
            "<circle cx='{}' cy='{}' r='0.5' fill='#070'/>",
            j[0],
            y(j[1])
        );
    }
    for n in &page.no_connects {
        s += &line(
            &[[n[0] - 0.8, n[1] - 0.8], [n[0] + 0.8, n[1] + 0.8]],
            "#00a",
            0.2,
        );
        s += &line(
            &[[n[0] - 0.8, n[1] + 0.8], [n[0] + 0.8, n[1] - 0.8]],
            "#00a",
            0.2,
        );
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
        let more = if page.page > 1 {
            format!("p{}", page.page)
        } else {
            String::new()
        };
        let file = format!(
            "{name}{}{more}.svg",
            page.path.replace(['/', '[', ']'], "_")
        );
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
    let ch = index
        .sheets
        .iter()
        .find(|s| s.path == "/ch[0]")
        .expect("a channel page");
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
        assert!(
            wired.iter().any(|n| n.ends_with(net)),
            "{net} is wired: {wired:?}"
        );
    }
    assert!(!ch.junctions.is_empty(), "T joins get junction dots");
    assert!(ch
        .labels
        .iter()
        .any(|l| l.kind == "power" && l.side == "up"));
    assert!(ch
        .labels
        .iter()
        .any(|l| l.kind == "power" && l.side == "down"));
    assert!(
        ch.labels.iter().any(|l| l.kind == "hier"),
        "ports leave as hierarchical labels"
    );
    // The top sheet: the connector feeds the channels, the channels feed the FPGA.
    let root = index.sheets.iter().find(|s| s.path == "/").unwrap();
    let j2 = root
        .symbols
        .iter()
        .find(|s| s.designator == "J2")
        .unwrap()
        .x_mm;
    let ch0 = root
        .sheet_symbols
        .iter()
        .find(|s| s.path == "/ch[0]")
        .unwrap()
        .x_mm;
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
    // The FPGA's unused banks (U1B–U1E, every pin no_erc) are page 2 of the top sheet, so the
    // top sheet's frame holds only what is wired.
    let spare = index
        .sheets
        .iter()
        .find(|s| s.path == "/" && s.page == 2)
        .expect("the top sheet's second page");
    let units = |page: &SchematicSheet| -> Vec<String> {
        page.symbols
            .iter()
            .filter(|s| s.designator == "U1")
            .filter_map(|s| s.unit.clone())
            .collect()
    };
    assert_eq!(units(spare), ["B", "C", "D", "E"]);
    assert_eq!(spare.symbols.len(), 4, "only the unused units");
    assert_eq!(spare.holds.as_deref(), Some("unused units"));
    assert!(units(root).contains(&"A".to_owned()) && !units(root).contains(&"B".to_owned()));
    let row = |page: &SchematicSheet, field: &str| {
        page.title_block
            .iter()
            .find(|(f, _)| f == field)
            .map(|(_, v)| v.clone())
            .unwrap()
    };
    assert_eq!(row(spare, "Sheet"), "main, page 2 (unused units)");
    let count = index.sheets.len();
    assert_eq!(row(root, "Page"), format!("1 of {count}"));
    assert_eq!(row(spare, "Page"), format!("2 of {count}"));
    assert!(
        root.width_mm <= 420.0,
        "the top sheet fits A3 without the banks: {}",
        root.frame
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

/// The quadcopter's flight controller (`runmat-cad/wasm/tests/drone`) as the automatic layout
/// draws it, with the lock's recorded places dropped.
fn flight_controller() -> SchematicIndex {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-cad/wasm/tests/drone/electrical/flight_controller");
    let files: Files = ["circuit", "circuit.lock"]
        .iter()
        .map(|ext| {
            let f = format!("flight_controller.{ext}");
            let text = std::fs::read_to_string(dir.join(&f)).unwrap();
            let text = if *ext == "circuit.lock" {
                text.lines()
                    .filter(|l| !l.starts_with("place "))
                    .map(|l| format!("{l}\n"))
                    .collect()
            } else {
                text
            };
            (format!("/p/{f}"), text)
        })
        .collect();
    drawn(&files, "/p/flight_controller.circuit")
}

#[test]
fn the_flight_controller_power_sheet_reads_as_drawn_by_hand() {
    let index = flight_controller();
    dump("fc", &index);
    for page in &index.sheets {
        check_page(page);
    }
    let pwr = index
        .sheets
        .iter()
        .find(|s| s.path == "/pwr")
        .expect("the power page");
    let at = |d: &str| {
        let s = pwr
            .symbols
            .iter()
            .find(|s| s.designator == d)
            .unwrap_or_else(|| panic!("{d} drawn"));
        (s.x_mm, s.y_mm)
    };
    // Power flows left to right: the buck, its inductor, the LDO.
    let (u2, l1, u3) = (at("U2"), at("L1"), at("U3"));
    assert!(
        u2.0 < l1.0 && l1.0 < u3.0,
        "U2 {u2:?} → L1 {l1:?} → U3 {u3:?}"
    );
    // The switch node is a wire from SW through L1 (a local power net is no rail).
    assert!(
        pwr.wires.iter().any(|w| w.net.ends_with("/SW")),
        "SW is wired"
    );
    assert!(
        !pwr.labels.iter().any(|l| l.text == "SW"),
        "SW gets no port"
    );
    // The feedback divider stands in one column beside U2, R6 above R7.
    let (r6, r7) = (at("R6"), at("R7"));
    assert!(
        (r6.0 - r7.0).abs() < EPS && r6.1 > r7.1,
        "R6 {r6:?} over R7 {r7:?}"
    );
    assert!(
        r6.0 < u2.0 && u2.0 - r6.0 < 40.0,
        "the divider sits by U2's FB pin: {r6:?} {u2:?}"
    );
    // The input caps by the buck, the output caps by the inductor, the LDO's caps by the LDO.
    for (cap, host) in [
        ("C12", u2),
        ("C13", u2),
        ("C15", l1),
        ("C16", l1),
        ("C18", u3),
        ("C19", u3),
    ] {
        let c = at(cap);
        assert!(
            (c.0 - host.0).abs() < 25.0 && c.1 < host.1,
            "{cap} {c:?} under its stage {host:?}"
        );
    }
    // The page's content stays clear of its title block (bottom right, y up).
    let title = [pwr.width_mm - 10.0 - 130.0, 10.0 + 6.0 * 6.0];
    for s in &pwr.symbols {
        let b = body_box(s);
        assert!(
            b[2] < title[0] || b[1] > title[1],
            "{} reaches the title block",
            s.designator
        );
    }
}

/// The flight controller's files with the lock's recorded places, and `edit` applied to the
/// circuit (old, new) pairs.
fn flight_controller_edited(edits: &[(&str, &str)]) -> (SchematicIndex, SchematicIndex) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../runmat-cad/wasm/tests/drone/electrical/flight_controller");
    let mut files: Files = ["circuit", "circuit.lock"]
        .iter()
        .map(|ext| {
            let f = format!("flight_controller.{ext}");
            (
                format!("/p/{f}"),
                std::fs::read_to_string(dir.join(&f)).unwrap(),
            )
        })
        .collect();
    let circuit = "/p/flight_controller.circuit";
    assert!(
        files["/p/flight_controller.circuit.lock"].contains("\nplace power/U3 "),
        "the lock records the power sheet"
    );
    let before = drawn(&files, circuit);
    let mut text = files[circuit].clone();
    for (old, new) in edits {
        assert_eq!(text.matches(old).count(), 1, "{old}");
        text = text.replace(old, new);
    }
    files.insert(circuit.into(), text);
    (before, drawn(&files, circuit))
}

/// An agent's edit adds a GPS supply to the recorded power sheet: an LDO fed from +5 V with its
/// input and output caps. The caps stand under the LDO, input cap on the input side, output cap
/// on the output side; the new block keeps clear of every drawn symbol, port and text, stays in
/// the page's frame, off its title block, and nothing recorded moves.
#[test]
fn a_new_regulator_lands_with_its_caps_on_a_recorded_sheet() {
    let (before, after) = flight_controller_edited(&[
        (
            "        component C19 = { part = \"C0402-1U\" }\n",
            "        component C19 = { part = \"C0402-1U\" }\n        component U6 = { part = \"LDO-3V3-SOT23\" }\n        component C24 = { part = \"C0402-1U\" }\n        component C25 = { part = \"C0402-1U\" }\n",
        ),
        (
            "        net \"+3V3\" = { class = supply, global = true }\n        net SW",
            "        net \"+3V3\" = { class = supply, global = true }\n        net \"+3V3_GPS\" = { class = supply, global = true }\n        net SW",
        ),
        (", U3.VIN, U3.EN, \"C18.1\"]", ", U3.VIN, U3.EN, \"C18.1\", U6.VIN, U6.EN, \"C24.1\"]"),
        (
            "        connect = [\"net.+3V3\", U3.VOUT, \"C19.1\"]\n",
            "        connect = [\"net.+3V3\", U3.VOUT, \"C19.1\"]\n        connect = [\"net.+3V3_GPS\", U6.VOUT, \"C25.1\"]\n",
        ),
        ("U3.GND, \"C18.2\", \"C19.2\",", "U3.GND, \"C18.2\", \"C19.2\", U6.GND, \"C24.2\", \"C25.2\","),
    ]);
    let page = |i: &SchematicIndex| {
        i.sheets
            .iter()
            .find(|s| s.path == "/pwr")
            .cloned()
            .expect("the power page")
    };
    let (old, pwr) = (page(&before), page(&after));
    for s in &old.symbols {
        let now = pwr
            .symbols
            .iter()
            .find(|n| n.key == s.key)
            .expect("still drawn");
        assert_eq!(
            (now.x_mm, now.y_mm, now.rotation),
            (s.x_mm, s.y_mm, s.rotation),
            "{} moved",
            s.designator
        );
    }
    assert_eq!(
        (pwr.frame.as_str(), pwr.width_mm),
        (old.frame.as_str(), old.width_mm),
        "the frame grew"
    );
    dump("fc-gps", &after);
    check_page(&pwr);
    let sym = |d: &str| {
        pwr.symbols
            .iter()
            .find(|s| s.designator == d)
            .unwrap_or_else(|| panic!("{d} drawn"))
    };
    let pin = |d: &str, p: &str| {
        let q = sym(d)
            .pins
            .iter()
            .find(|q| q.name == p)
            .unwrap_or_else(|| panic!("{d}.{p}"));
        (q.x_mm, q.y_mm)
    };
    let (u6, c24, c25) = (
        body_box(sym("U6")),
        body_box(sym("C24")),
        body_box(sym("C25")),
    );
    // As far under U6 as U3's own caps stand under U3: a row a port's height below the IC.
    let (u3, c18) = (body_box(sym("U3")), body_box(sym("C18")));
    let row = u3[1] - c18[3];
    for (cap, b) in [("C24", c24), ("C25", c25)] {
        let below = u6[1] - b[3];
        assert!(
            below > 0.0 && below <= row + EPS,
            "{cap} {b:?} stands under U6 {u6:?} as C18 stands under U3 ({row} mm)"
        );
        assert!(
            b[0] > u6[0] - 15.0 && b[2] < u6[2] + 15.0,
            "{cap} {b:?} is under U6 {u6:?}"
        );
    }
    let (vin, vout) = (pin("U6", "VIN"), pin("U6", "VOUT"));
    assert!(
        vin.0 < vout.0,
        "U6 reads input to output: VIN {vin:?}, VOUT {vout:?}"
    );
    assert!(
        c24[2] <= c25[0],
        "the input cap C24 {c24:?} stands on the input side of the output cap C25 {c25:?}"
    );
    // Everything the new parts draw (bodies, texts, ports) keeps clear of the old drawing and
    // of the title block, inside the frame.
    let new_boxes: Vec<[f64; 4]> = {
        let old_boxes = runmat_circuit::schematic::page_boxes(&old);
        runmat_circuit::schematic::page_boxes(&pwr)
            .into_iter()
            .filter(|b| {
                !old_boxes
                    .iter()
                    .any(|o| o.iter().zip(b).all(|(x, y)| (x - y).abs() < EPS))
            })
            .collect()
    };
    use runmat_circuit::schematic::{title_height_mm, FRAME_MM, TITLE_W_MM};
    let title = [
        pwr.width_mm - FRAME_MM - TITLE_W_MM - 2.0,
        0.0,
        pwr.width_mm,
        FRAME_MM + title_height_mm(5) + 2.0,
    ];
    for b in &new_boxes {
        assert!(
            !overlaps(b, &title, 0.0),
            "{b:?} reaches the title block {title:?}"
        );
        assert!(
            b[0] >= 10.0
                && b[2] <= pwr.width_mm - 10.0
                && b[1] >= 10.0
                && b[3] <= pwr.height_mm - 10.0,
            "{b:?} leaves the frame"
        );
    }
    for s in old.symbols.iter() {
        let b = body_box(s);
        for d in ["U6", "C24", "C25"] {
            assert!(
                !overlaps(&body_box(sym(d)), &b, 0.0),
                "{d} sits on {}",
                s.designator
            );
        }
    }
    // On the sheet's grid: whole 1.27 mm steps from the recorded U3.
    let grid = |v: f64| ((v / 1.27).round() * 1.27 - v).abs() < 1e-3;
    for d in ["U6", "C24", "C25"] {
        let (s, u3) = (sym(d), sym("U3"));
        assert!(
            grid(s.x_mm - u3.x_mm) && grid(s.y_mm - u3.y_mm),
            "{d} is on the grid"
        );
    }
}

/// An agent's edit as it came: an LDO from +5 V onto GPS_VCC, a supply net the connectors
/// sheet declares (J5's pin), with its input cap on +5 V and its output cap on GPS_VCC. The caps
/// share a net with a pin of the new LDO, so each stands beside that pin, in a row under the
/// LDO as C18 and C19 stand under U3, input cap on the input side, clear of every symbol.
#[test]
fn an_agent_added_regulator_on_an_existing_supply_keeps_its_caps_beside_its_pins() {
    let (before, after) = flight_controller_edited(&[
        (
            "        component C19 = { part = \"C0402-1U\" }\n",
            "        component C19 = { part = \"C0402-1U\" }\n        component U6 = { part = \"LDO-3V3-SOT23\" }\n        component C24 = { part = \"C0402-1U\" }\n        component C25 = { part = \"C0402-1U\" }\n",
        ),
        (", U3.VIN, U3.EN, \"C18.1\"]", ", U3.VIN, U3.EN, \"C18.1\", U6.VIN, U6.EN, \"C24.1\"]"),
        (
            "        connect = [\"net.+3V3\", U3.VOUT, \"C19.1\"]\n",
            "        connect = [\"net.+3V3\", U3.VOUT, \"C19.1\"]\n        connect = [net.GPS_VCC, U6.VOUT, \"C25.1\"]\n",
        ),
        ("U3.GND, \"C18.2\", \"C19.2\",", "U3.GND, \"C18.2\", \"C19.2\", U6.GND, \"C24.2\", \"C25.2\","),
    ]);
    let page = |i: &SchematicIndex| {
        i.sheets
            .iter()
            .find(|s| s.path == "/pwr")
            .cloned()
            .expect("the power page")
    };
    let (old, pwr) = (page(&before), page(&after));
    dump("fc-gps-vcc", &after);
    for s in &old.symbols {
        let now = pwr
            .symbols
            .iter()
            .find(|n| n.key == s.key)
            .expect("still drawn");
        assert_eq!(
            (now.x_mm, now.y_mm, now.rotation),
            (s.x_mm, s.y_mm, s.rotation),
            "{} moved",
            s.designator
        );
    }
    check_page(&pwr);
    let sym = |d: &str| {
        pwr.symbols
            .iter()
            .find(|s| s.designator == d)
            .unwrap_or_else(|| panic!("{d} drawn"))
    };
    let (u6, c24, c25) = (
        body_box(sym("U6")),
        body_box(sym("C24")),
        body_box(sym("C25")),
    );
    let (u3, c18) = (body_box(sym("U3")), body_box(sym("C18")));
    let row = u3[1] - c18[3];
    for (cap, b) in [("C24", c24), ("C25", c25)] {
        let below = u6[1] - b[3];
        assert!(
            below > 0.0 && below <= row + EPS,
            "{cap} {b:?} stands under U6 {u6:?} as C18 stands under U3 ({row} mm)"
        );
        assert!(
            b[0] > u6[0] - 15.0 && b[2] < u6[2] + 15.0,
            "{cap} {b:?} is under U6 {u6:?}"
        );
    }
    assert!(
        c24[2] <= c25[0],
        "the input cap C24 {c24:?} stands on the input side of the output cap C25 {c25:?}"
    );
    for s in pwr
        .symbols
        .iter()
        .filter(|s| !["U6", "C24", "C25"].contains(&s.designator.as_str()))
    {
        for d in ["U6", "C24", "C25"] {
            assert!(
                !overlaps(&body_box(sym(d)), &body_box(s), 0.0),
                "{d} sits on {}",
                s.designator
            );
        }
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
    let out = edit(
        circuit,
        &files[circuit],
        &files,
        &request("schematic.annotate", serde_json::json!({})),
    )
    .expect("annotates");
    let written = out
        .files
        .iter()
        .find(|f| f.path == lock)
        .expect("the lock records places");
    assert!(
        written.text.contains("\nplace channel/RS "),
        "{}",
        written.text
    );
    files.insert(lock.into(), written.text.clone());
    let before = drawn(&files, circuit);
    let placed = |index: &SchematicIndex| -> BTreeMap<String, (f64, f64, f64)> {
        index
            .sheets
            .iter()
            .flat_map(|s| {
                s.symbols.iter().map(move |y| {
                    (
                        format!("{}|{}", s.path, y.key),
                        (y.x_mm, y.y_mm, y.rotation),
                    )
                })
            })
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
            ..request(
                "schematic.move",
                serde_json::json!({ "dx": 2.54, "dy": -5.08 }),
            )
        },
    )
    .expect("moves");
    let lock_text = &moved.files.iter().find(|f| f.path == lock).unwrap().text;
    files.insert(lock.into(), lock_text.clone());
    let again = drawn(&files, circuit);
    let j2 = |i: &SchematicIndex| {
        let s = i.sheets[0]
            .symbols
            .iter()
            .find(|s| s.designator == "J2")
            .unwrap();
        (s.x_mm, s.y_mm)
    };
    let (x0, y0) = j2(&after);
    let (x1, y1) = j2(&again);
    assert!(
        (x1 - x0 - 2.54).abs() < 1e-6 && (y1 - y0 + 5.08).abs() < 1e-6,
        "{x0},{y0} → {x1},{y1}"
    );
}

/// The layout is part of every compile, so the showcase schematic (without its board) compiles
/// well inside the Desktop's 0.5 s edit budget (an optimised build; debug builds get more room).
#[test]
fn the_layout_is_fast() {
    let mut files = project("daq");
    files.remove("/p/daq.circuit.layout");
    let run = || {
        compile(
            "/p/daq.circuit",
            &files["/p/daq.circuit"],
            &files,
            &CompileOptions::default(),
        )
    };
    run();
    let start = std::time::Instant::now();
    let runs: usize = std::env::var("RUNMAT_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    for _ in 0..runs {
        std::hint::black_box(run());
    }
    let each = start.elapsed().as_secs_f64() / runs as f64;
    let budget = if cfg!(debug_assertions) { 2.0 } else { 0.1 };
    eprintln!("schematic compile: {:.1} ms", each * 1e3);
    assert!(each < budget, "the compile took {each:.3} s");
}
