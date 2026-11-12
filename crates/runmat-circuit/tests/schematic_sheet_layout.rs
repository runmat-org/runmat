//! The drawn sheet as a reader sees it: notes inside the frame (beside their subject or in a
//! notes block), parts with a declared library symbol drawn by it, child sheets' power ports
//! as entries, a sheet size that holds when content strays into the margin, and the layout
//! check that reports what lands off the sheet or over other text.

use runmat_circuit::runtime::{self, rule_check, CompileOptions, Files};
use runmat_circuit::schematic::{check, title_height_mm, SchematicSheet, FRAME_MM, TITLE_W_MM};

const LIBRARY: &str = r#"
symbol breaker = {
    body = [line([-2.54, 0], [1.27, 1.27]), circle([-2.54, 0], 0.4), circle([2.54, 0], 0.4)],
    pins = [
        { name = LINE, at = [-5.08, 0], dir = right },
        { name = LOAD, at = [5.08, 0], dir = left }
    ]
}

symbol meter = {
    body = [circle([0, 0], 3.81), text([0, 0], "kWh")],
    pins = [
        { name = IN, at = [-7.62, 0], dir = right },
        { name = OUT, at = [7.62, 0], dir = left }
    ]
}

part "BKR-20A" = {
    manufacturer = "Sample Catalog",
    description = "Breaker, 1 pole, 20 A",
    category = breaker,
    package = none,
    symbol = breaker,
    pins = [
        { name = LINE, type = passive },
        { name = LOAD, type = passive }
    ]
}

part "METER" = {
    manufacturer = "Sample Catalog",
    description = "Meter",
    category = meter,
    package = none,
    symbol = meter,
    pins = [
        { name = IN, type = passive },
        { name = OUT, type = passive }
    ]
}

part "RCPT" = {
    manufacturer = "Sample Catalog",
    description = "Receptacle",
    category = receptacle,
    package = none,
    pins = [
        { name = HOT, type = passive },
        { name = NEU, type = passive }
    ]
}
"#;

fn circuit(body: &str) -> String {
    format!("runmat 0.7\nstd 1\n{LIBRARY}\n{body}")
}

fn compile(text: &str) -> runmat_circuit::index::CircuitIndex {
    let path = "/p/service.circuit";
    let files: Files = [(path.to_owned(), text.to_owned())].into_iter().collect();
    let out = runtime::compile(path, text, &files, &CompileOptions::default());
    out.index
        .unwrap_or_else(|| panic!("compiles: {:?}", out.diagnostics))
}

/// Whether a box (mm) lies inside the page's frame line and off its title block.
fn on_sheet(page: &SchematicSheet, b: [f64; 4]) -> bool {
    let (w, h) = (page.width_mm, page.height_mm);
    let inside =
        b[0] >= FRAME_MM && b[1] >= FRAME_MM && b[2] <= w - FRAME_MM && b[3] <= h - FRAME_MM;
    let title = [
        w - FRAME_MM - TITLE_W_MM,
        FRAME_MM,
        w - FRAME_MM,
        FRAME_MM + title_height_mm(page.title_block.len()),
    ];
    let over = b[0] < title[2] && title[0] < b[2] && b[1] < title[3] && title[1] < b[3];
    inside && !over
}

const SERVICE: &str = r#"
circuit service {
    root = one_line
    sheet one_line {
        text notes = "NOTES"
        text drop = "1. Utility drop: 120/240 V, 1 phase, 3 wire, to the meter on wall A."
        text se = "2. Service entrance: 4/0 Al XHHW-2 (SER) in 2 in PVC to the 200 A main."
        text ground = "3. GEC: 4 AWG Cu to the footing electrode E1; 6 AWG Cu to the rod E2."
        text bond = "4. JP1 bonds the neutral bar to the enclosure at the panel only."
        text load = "5. Load (NEC 220.82): 128 A on the 200 A service; see DESIGN-NOTES."
        text cb1_load = "Range, kitchen east wall"
        component M1 = { part = "METER" }
        component CB1 = { part = "BKR-20A" }
        component X1 = { part = "RCPT" }
        net L1 = { power_flag = true }
        net N = { global = true, power_flag = true }
        connect = [net.L1, M1.IN]
        connect = [net.SE, M1.OUT, CB1.LINE]
        connect = [net.C1, CB1.LOAD, X1.HOT]
        connect = [net.N, X1.NEU]
    }
}
"#;

#[test]
fn unplaced_notes_go_beside_their_subject_or_into_a_block_inside_the_sheet() {
    let index = compile(&circuit(SERVICE));
    let page = &index.schematic.sheets[0];
    assert_eq!(page.frame, "A4", "a few parts and six notes fit A4");
    assert_eq!(page.texts.len(), 7);
    for t in &page.texts {
        let w = t.text.chars().count() as f64 * t.size_mm * 0.72;
        let b = [
            t.x_mm,
            t.y_mm - t.size_mm / 5.0,
            t.x_mm + w,
            t.y_mm + t.size_mm * 0.8,
        ];
        assert!(
            on_sheet(page, b),
            "note {} at {:?} on the {} sheet",
            t.name,
            b,
            page.frame
        );
    }
    // The block keeps source order, one line apart.
    let block: Vec<_> = page.texts.iter().filter(|t| t.name != "cb1_load").collect();
    for pair in block.windows(2) {
        assert!((pair[0].x_mm - pair[1].x_mm).abs() < 1e-6, "{pair:?}");
        assert!(
            (pair[0].y_mm - pair[1].y_mm - 2.54).abs() < 1e-6,
            "{pair:?}"
        );
    }
    // The note named after CB1 sits right beside it.
    let cb1 = page.symbols.iter().find(|s| s.designator == "CB1").unwrap();
    let note = page.texts.iter().find(|t| t.name == "cb1_load").unwrap();
    let pins_right = cb1.pins.iter().map(|p| p.x_mm).fold(f64::MIN, f64::max);
    assert!(
        note.x_mm > cb1.x_mm
            && note.x_mm - pins_right < 15.0
            && (note.y_mm - cb1.y_mm).abs() < 10.0,
        "note at ({}, {}) beside CB1 at ({}, {})",
        note.x_mm,
        note.y_mm,
        cb1.x_mm,
        cb1.y_mm
    );
    let found = check::page_findings(page);
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn a_part_with_a_declared_symbol_draws_its_art_and_pins() {
    let index = compile(&circuit(SERVICE));
    let page = &index.schematic.sheets[0];
    let m1 = page.symbols.iter().find(|s| s.designator == "M1").unwrap();
    assert_eq!(m1.art, "symbol");
    assert!(!m1.strokes.is_empty(), "the meter's circle is drawn");
    assert_eq!(m1.art_texts.len(), 1);
    assert_eq!(m1.art_texts[0].text, "kWh");
    // The art text sits at the symbol's origin; the pins at the declaration's tips.
    let (ox, oy) = (m1.x_mm, m1.y_mm);
    assert!((m1.art_texts[0].at[0] - ox).abs() < 1e-6 && (m1.art_texts[0].at[1] - oy).abs() < 1e-6);
    let mut tips: Vec<f64> = m1
        .pins
        .iter()
        .map(|p| ((p.x_mm - ox).abs() * 100.0).round() / 100.0)
        .collect();
    tips.sort_by(f64::total_cmp);
    assert_eq!(tips, vec![7.62, 7.62]);
    assert!(
        m1.pins.iter().all(|p| p.hidden),
        "pin names hide on symbol art"
    );
    // A part without a symbol keeps the generated box.
    let x1 = page.symbols.iter().find(|s| s.designator == "X1").unwrap();
    assert_eq!(x1.art, "box");
}

#[test]
fn a_child_sheets_power_ports_are_drawn_as_entries_and_as_ports_on_its_page() {
    let text = circuit(
        r#"
circuit branches {
    root = feeds
    sheet feeds {
        net C1 = { power_flag = true }
        net N = { global = true, power_flag = true }
        instance c1 = { sheet = c1 }
        connect = [net.C1, instance.c1.HOT]
    }
    sheet c1 {
        port HOT : power
        component X1 = { part = "RCPT" }
        connect = [port.HOT, X1.HOT]
        connect = [net.N, X1.NEU]
    }
}
"#,
    );
    let index = compile(&text);
    let feeds = index
        .schematic
        .sheets
        .iter()
        .find(|s| s.path == "/")
        .unwrap();
    let block = &feeds.sheet_symbols[0];
    let entries: Vec<_> = block.pins.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(entries, vec!["HOT"]);
    let child = index
        .schematic
        .sheets
        .iter()
        .find(|s| s.path == "/c1")
        .unwrap();
    assert!(
        child
            .labels
            .iter()
            .any(|l| l.text == "HOT" && l.kind == "hier"),
        "{:?}",
        child.labels
    );
}

#[test]
fn a_recorded_page_that_strays_into_the_margin_keeps_its_sheet() {
    let base = circuit(SERVICE);
    let index = compile(&base);
    let page = &index.schematic.sheets[0];
    assert_eq!(page.frame, "A4");
    // Record the places, then move M1 a little over the top margin (0.2 mm past it).
    let places = runmat_circuit::schematic::places(&index.schematic);
    let m1 = page.symbols.iter().find(|s| s.designator == "M1").unwrap();
    let top = m1
        .pins
        .iter()
        .map(|p| p.y_mm)
        .fold(f64::MIN, f64::max)
        .max(m1.designator_at[1] + 1.6);
    let lift = (page.height_mm - 25.4 - top) + 0.2;
    let mut lock = runmat_circuit::ids::LockSection {
        places: places.clone(),
        ..Default::default()
    };
    let key = &m1.key;
    let p = lock.places.get_mut(key).unwrap();
    p.y -= (lift * 1e6).round() as i64;
    let moved = relaid(&base, &lock);
    let page = &moved.sheets[0];
    assert_eq!(page.frame, "A4", "0.2 mm into the margin keeps A4");
    let m1 = page.symbols.iter().find(|s| s.designator == "M1").unwrap();
    assert!(m1.y_mm > 150.0, "M1 stays near the top: {}", m1.y_mm);
    let found = check::page_findings(page);
    assert!(found.iter().all(|f| f.code != "off_sheet"), "{found:?}");
}

/// Lays the circuit out again with `lock`'s places, read from its lock file.
fn relaid(
    text: &str,
    lock: &runmat_circuit::ids::LockSection,
) -> runmat_circuit::schematic::SchematicIndex {
    let path = "/p/service.circuit";
    let files: Files = [
        (path.to_owned(), text.to_owned()),
        ("/p/service.circuit.lock".to_owned(), lock.to_text()),
    ]
    .into_iter()
    .collect();
    let out = runtime::compile(path, text, &files, &CompileOptions::default());
    out.index.expect("compiles").schematic
}

#[test]
fn texts_drawn_over_each_other_and_items_off_the_frame_are_reported() {
    let index = compile(&circuit(SERVICE));
    let mut page = index.schematic.sheets[0].clone();
    // Put one note on another, and another past the right edge.
    let (a, b) = (page.texts[1].clone(), page.texts[2].clone());
    page.texts[2].x_mm = a.x_mm + 1.0;
    page.texts[2].y_mm = a.y_mm;
    page.texts[3].x_mm = page.width_mm - 12.0;
    let found = check::page_findings(&page);
    let codes: Vec<_> = found.iter().map(|f| (f.code, f.items.clone())).collect();
    assert!(
        codes.contains(&(
            "text_overlap",
            vec![format!("note {}", a.name), format!("note {}", b.name)]
        )),
        "{codes:?}"
    );
    assert!(
        codes
            .iter()
            .any(|(c, items)| *c == "off_sheet"
                && items[0] == format!("note {}", page.texts[3].name)),
        "{codes:?}"
    );
    // The rule check lists them as `schematic` findings too.
    let path = "/p/service.circuit";
    let text = circuit(SERVICE);
    let files: Files = [(path.to_owned(), text.clone())].into_iter().collect();
    let out = rule_check(path, &text, &files, &CompileOptions::default());
    assert!(
        out.erc.iter().all(|v| v.check != "schematic"),
        "the clean page has none: {:?}",
        out.erc
    );
}
