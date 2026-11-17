//! The ECAD commands that change what a component is, where it sits and which rules apply, run
//! through the shared dispatcher from the schematic and from the board: Rules, Net class,
//! Footprint, Change part, Swap pins/gates, Place, Align, Fitted and Net length.

use runmat_circuit::runtime::{CompileOptions, EditError, EditOutput, EditRequest, Files, Session};
use serde_json::{json, Value};

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";

fn files() -> Files {
    ["io.circuit", "io.circuit.layout", "io.circuit.lock"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(format!(
                "{}/tests/projects/io/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            (format!("/p/{name}"), text)
        })
        .collect()
}

fn run_on(
    files: &Files,
    from: &str,
    command: &str,
    selection: &[&str],
    values: Value,
) -> Result<EditOutput, EditError> {
    let mut session = Session::new();
    let out = session.compile(from, &files[from], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let request = EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    };
    session.edit(from, &files[from], files, &request)
}

fn run(
    from: &str,
    command: &str,
    selection: &[&str],
    values: Value,
) -> Result<EditOutput, EditError> {
    run_on(&files(), from, command, selection, values)
}

/// `files` after the edit, checked to compile clean.
fn applied_on(
    files: &Files,
    from: &str,
    command: &str,
    selection: &[&str],
    values: Value,
) -> (Files, String) {
    let out = run_on(files, from, command, selection, values)
        .unwrap_or_else(|e| panic!("{command}: {}", e.message));
    assert!(
        !out.files.is_empty(),
        "{command} changed nothing: {}",
        out.summary
    );
    let mut next = files.clone();
    for f in out.files {
        next.insert(f.path, f.text);
    }
    let mut session = Session::new();
    let compiled = session.compile(CIRCUIT, &next[CIRCUIT], &next, &CompileOptions::default());
    assert!(
        compiled.diagnostics.is_empty(),
        "{command}: {:?}",
        compiled.diagnostics
    );
    (next, out.summary)
}

fn applied(from: &str, command: &str, selection: &[&str], values: Value) -> (Files, String) {
    applied_on(&files(), from, command, selection, values)
}

fn line<'t>(text: &'t str, start: &str) -> &'t str {
    text.lines()
        .map(str::trim)
        .find(|l| l.starts_with(start))
        .unwrap_or_else(|| panic!("no line starting {start}"))
}

fn has_line(text: &str, start: &str) -> bool {
    text.lines().any(|l| l.trim().starts_with(start))
}

#[test]
fn rules_add_edit_and_delete_a_rule_from_either_view() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, _) = applied(
            from,
            "board.rules",
            &[],
            json!({ "kind": "width", "applies": ":net-class(power)", "value": "0.5 mm", "priority": 2 }),
        );
        assert_eq!(
            line(&next[LAYOUT], "rule width_1"),
            r#"rule width_1 = { select = ":net-class(power)", priority = 2, track_width = 0.5 mm }"#
        );
        // Edited from its row: the same name, a new value.
        let (next, _) = applied_on(
            &next,
            from,
            "board.rules",
            &["rule:width_1"],
            json!({ "kind": "width", "applies": ":net-class(power)", "value": "20 mil" }),
        );
        assert!(
            line(&next[LAYOUT], "rule width_1").contains("track_width = 0.508 mm"),
            "{}",
            line(&next[LAYOUT], "rule width_1")
        );
        assert_eq!(next[LAYOUT].matches("rule width_1").count(), 1);
        let (next, summary) =
            applied_on(&next, from, "circuit.delete", &["rule:width_1"], json!({}));
        assert!(!has_line(&next[LAYOUT], "rule width_1"), "{summary}");
    }
    // A clearance is a pair rule against everything; a via takes drill/diameter.
    let (next, _) = applied(
        LAYOUT,
        "board.rules",
        &[],
        json!({ "kind": "clearance", "applies": ":net-class(analog)", "value": "0.3 mm" }),
    );
    assert!(line(&next[LAYOUT], "rule clearance_1")
        .contains(r#"between = [":net-class(analog)", "*"]"#));
    let (next, _) = applied(
        LAYOUT,
        "board.rules",
        &[],
        json!({ "kind": "via", "applies": ":net-class(power)", "value": "0.3 mm/0.7 mm", "name": "power_via" }),
    );
    let via = line(&next[LAYOUT], "rule power_via");
    assert!(
        via.contains("via_drill = 0.3 mm") && via.contains("via_diameter = 0.7 mm"),
        "{via}"
    );
    let e = run(
        LAYOUT,
        "board.rules",
        &[],
        json!({ "kind": "width", "applies": "track:nope(", "value": "1 mm" }),
    )
    .unwrap_err();
    assert!(e.message.contains("isn't a selector"), "{}", e.message);
    let e = run(
        LAYOUT,
        "board.rules",
        &[],
        json!({ "kind": "width", "applies": "track" }),
    )
    .unwrap_err();
    assert!(e.message.contains("value"), "{}", e.message);
}

#[test]
fn a_mask_rule_grows_the_mask_openings_it_selects() {
    let (next, _) = applied(
        LAYOUT,
        "board.rules",
        &[],
        json!({ "kind": "mask", "applies": "pad", "value": "0.2 mm" }),
    );
    assert!(line(&next[LAYOUT], "rule mask_1").contains("mask_expansion = 0.2 mm"));
}

#[test]
fn net_class_declares_nets_and_classes_as_needed() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, summary) = applied(
            from,
            "circuit.net-class",
            &["net:SHIELD"],
            json!({ "class": "analog" }),
        );
        assert!(
            has_line(&next[CIRCUIT], "net SHIELD = { class = analog }"),
            "{summary}"
        );
        // A channel's net: its sheet declares it once for every channel.
        let (next, _) = applied(
            from,
            "circuit.net-class",
            &["net:/ch[0]/INF"],
            json!({ "class": "hs" }),
        );
        assert!(has_line(&next[CIRCUIT], "net INF = { class = hs }"));
        assert!(has_line(&next[CIRCUIT], "net_class hs"));
        let (next, _) = applied_on(
            &next,
            from,
            "circuit.net-class",
            &["net:/ch[0]/INF"],
            json!({ "class": "default" }),
        );
        assert!(!has_line(&next[CIRCUIT], "net INF = { class"));
    }
    let e = run(
        CIRCUIT,
        "circuit.net-class",
        &["component:C900"],
        json!({ "class": "power" }),
    )
    .unwrap_err();
    assert!(e.message.contains("select the nets"), "{}", e.message);
}

#[test]
fn footprint_takes_the_library_part_with_the_same_value() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, summary) = applied(
            from,
            "circuit.footprint",
            &["component:R1"],
            json!({ "footprint": "R0603" }),
        );
        assert!(
            has_line(&next[CIRCUIT], r#"component RF = { part = "R0603-10K" }"#),
            "{summary}"
        );
    }
    let e = run(
        CIRCUIT,
        "circuit.footprint",
        &["component:C900"],
        json!({ "footprint": "C0402" }),
    )
    .unwrap_err();
    assert!(
        e.message
            .contains("no C0805-10U equivalent in C_0402_1005Metric")
            && e.message.contains("C_0805"),
        "{}",
        e.message
    );
}

#[test]
fn change_part_maps_pins_by_name() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, _) = applied(
            from,
            "circuit.change-part",
            &["component:C900"],
            json!({ "part": "C0402-100N" }),
        );
        assert!(has_line(
            &next[CIRCUIT],
            r#"component C900 = { part = "C0402-100N" }"#
        ));
    }
    let e = run(
        CIRCUIT,
        "circuit.change-part",
        &["component:U1"],
        json!({ "part": "R0402-10K" }),
    )
    .unwrap_err();
    assert!(e.message.contains("has no pin"), "{}", e.message);
}

#[test]
fn swap_pins_trades_connections_and_pads() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, summary) = applied(
            from,
            "circuit.swap-pins",
            &["component:C900"],
            json!({ "swap": "pins", "pins": "1, 2" }),
        );
        assert!(
            has_line(&next[CIRCUIT], r#"connect = ["net.+5V", "C900.2"]"#),
            "{summary}"
        );
        assert!(has_line(&next[CIRCUIT], r#"connect = [net.GND, "C900.1"]"#));
        let fp = line(&next[LAYOUT], "footprint C900");
        assert!(
            fp.contains(r#""1" = GND"#) && fp.contains(r#""2" = "+5V""#),
            "{fp}"
        );
    }
    // Gates of the quad op-amp, in every channel.
    let (next, _) = applied(
        CIRCUIT,
        "circuit.swap-pins",
        &["component:U1"],
        json!({ "swap": "gates", "gates": "B, C" }),
    );
    assert!(has_line(
        &next[CIRCUIT],
        "connect = [net.TIE_B, U.C.OUT, \"U.C.IN-\"]"
    ));
    let u1 = line(&next[LAYOUT], "footprint U1 ");
    assert!(
        u1.contains(r#""8" = "/ch[0]/TIE_B""#) && u1.contains(r#""7" = "/ch[0]/TIE_C""#),
        "{u1}"
    );
    let e = run(
        CIRCUIT,
        "circuit.swap-pins",
        &["component:U1"],
        json!({ "swap": "pins", "pins": "A.IN+, A.IN-" }),
    )
    .unwrap_err();
    assert!(
        e.message.contains("aren't interchangeable"),
        "{}",
        e.message
    );
    let e = run(
        CIRCUIT,
        "circuit.swap-pins",
        &["component:U1"],
        json!({ "swap": "gates", "gates": "A, P" }),
    )
    .unwrap_err();
    assert!(
        e.message.contains("aren't interchangeable"),
        "{}",
        e.message
    );
}

#[test]
fn fitted_writes_the_variants_override() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, summary) = applied(
            from,
            "circuit.fitted",
            &["component:C900"],
            json!({ "fitting": "not_fitted" }),
        );
        assert!(
            line(&next[CIRCUIT], "variant lite").contains(r#""/C900" = not_fitted"#),
            "{summary}"
        );
        let (next, _) = applied_on(
            &next,
            from,
            "circuit.fitted",
            &["component:C900"],
            json!({ "fitting": "alternate", "part": "C0402-100N" }),
        );
        assert!(
            line(&next[CIRCUIT], "variant lite").contains("C0402-100N"),
            "{}",
            line(&next[CIRCUIT], "variant lite")
        );
        let (next, _) = applied_on(
            &next,
            from,
            "circuit.fitted",
            &["component:C900"],
            json!({ "fitting": "fitted" }),
        );
        assert!(!line(&next[CIRCUIT], "variant lite").contains("C900"));
    }
    let e = run(
        CIRCUIT,
        "circuit.fitted",
        &["component:C900"],
        json!({ "fitting": "not_fitted", "variant": "full" }),
    )
    .unwrap_err();
    assert!(e.message.contains("no variant `full`"), "{}", e.message);
}

fn at(layout: &str, designator: &str) -> (f64, f64) {
    let l = line(layout, &format!("footprint {designator} "));
    let rest = &l[l.find("at = [").expect("at") + 6..];
    let xy: Vec<f64> = rest[..rest.find(']').unwrap()]
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    (xy[0], xy[1])
}

#[test]
fn place_brings_staged_footprints_inside_the_outline() {
    let mut staged = files();
    let text = staged[LAYOUT].replace(
        "footprint R900 = { package = R_0402_1005Metric_IPCN, at = [88, 10]",
        "footprint R900 = { package = R_0402_1005Metric_IPCN, at = [130, 10]",
    );
    staged.insert(LAYOUT.into(), text);
    for from in [CIRCUIT, LAYOUT] {
        let (next, _) = applied_on(
            &staged,
            from,
            "board.place",
            &[],
            json!({ "mode": "from_schematic", "side": "top" }),
        );
        let (x, y) = at(&next[LAYOUT], "R900");
        assert!(
            (0.0..107.0).contains(&x) && (0.0..49.0).contains(&y),
            "R900 at {x}, {y}"
        );
        assert_eq!(
            at(&next[LAYOUT], "C900"),
            (97.0, 24.0),
            "a placed part stays"
        );
    }
    let e = run(
        LAYOUT,
        "board.place",
        &[],
        json!({ "mode": "from_schematic" }),
    )
    .unwrap_err();
    assert!(e.message.contains("on the board already"), "{}", e.message);
    // Single: the picked part, on the bottom.
    let (next, _) = applied(
        LAYOUT,
        "board.place",
        &[],
        json!({ "mode": "single", "components": ["component:C900"], "side": "bottom" }),
    );
    assert!(line(&next[LAYOUT], "footprint C900").contains("side = bottom"));
    assert_eq!(
        at(&next[LAYOUT], "C900"),
        (97.0, 24.0),
        "a placed part keeps its spot"
    );
}

#[test]
fn align_lines_components_up_and_distributes_them() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, _) = applied(
            from,
            "board.align",
            &["component:C900", "component:R900"],
            json!({ "align": "left" }),
        );
        let (c, r) = (at(&next[LAYOUT], "C900"), at(&next[LAYOUT], "R900"));
        assert_eq!(c.0, r.0, "same package, same left edge: {c:?} {r:?}");
        assert_eq!(r, (88.0, 10.0), "the leftmost stays");
    }
    let (next, _) = applied(
        LAYOUT,
        "board.align",
        &[],
        json!({ "align": "distribute_x", "components": ["component:R50", "component:R1", "component:R4"] }),
    );
    let mut xs: Vec<f64> = ["R50", "R1", "R4"]
        .iter()
        .map(|d| at(&next[LAYOUT], d).0)
        .collect();
    xs.sort_by(f64::total_cmp);
    assert!(
        (xs[2] - xs[1] - (xs[1] - xs[0])).abs() < 1e-3 && xs[2] > xs[0],
        "evenly spread: {xs:?}"
    );
    let e = run(
        LAYOUT,
        "board.align",
        &["component:C900"],
        json!({ "align": "left" }),
    )
    .unwrap_err();
    assert!(e.message.contains("two or more"), "{}", e.message);
}

#[test]
fn net_length_reports_routed_and_unrouted_length() {
    for from in [CIRCUIT, LAYOUT] {
        let out = run(
            from,
            "circuit.net-length",
            &["net:+5V", "net:AIN0"],
            json!({}),
        )
        .unwrap();
        assert!(out.files.is_empty());
        let report = out.report.expect("a table");
        assert_eq!(report["columns"][1], "Routed");
        let rows = report["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["cells"][0], "+5V");
        assert!(rows[0]["cells"][1].as_str().unwrap().ends_with(" mm"));
    }
    let e = run(CIRCUIT, "circuit.net-length", &[], json!({})).unwrap_err();
    assert!(e.message.contains("select the nets"), "{}", e.message);
}
