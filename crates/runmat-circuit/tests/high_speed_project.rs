//! The high-speed commands on a real project: the USB demo board (`tests/projects/usb`) with a
//! 90 Ω differential class on its D+/D- pair, on a four-layer stack with 0.1 mm prepreg. Diff pair takes the
//! class's solved width and gap, Tune length reaches a delay target with each pattern, and
//! Tune group brings a matched group in; the board stays DRC clean.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::board::parse_board;
use serde_json::{json, Value};
use std::path::PathBuf;

const NAME: &str = "usb_bridge";
const LAYOUT: &str = "/p/usb_bridge.circuit.layout";

fn files() -> Files {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/usb");
    let mut files = Files::new();
    for x in [".circuit", ".circuit.layout", ".circuit.lock"] {
        files.insert(format!("/p/{NAME}{x}"), std::fs::read_to_string(dir.join(format!("{NAME}{x}"))).unwrap());
    }
    files
}

fn request(command: &str, values: Value) -> EditRequest {
    EditRequest { command: command.into(), values: values.as_object().cloned().unwrap_or_default(), selection: vec![], options: CompileOptions::default() }
}

fn run(files: &mut Files, command: &str, values: Value) -> String {
    let out = runtime::edit(LAYOUT, &files[LAYOUT], files, &request(command, values)).unwrap_or_else(|e| panic!("{command}: {} {:?}", e.message, e.diagnostics));
    for f in out.files {
        files.insert(f.path.clone(), f.text.clone());
    }
    out.summary
}

fn index(files: &Files) -> runmat_circuit::index::BoardIndex {
    let out = runtime::compile(LAYOUT, &files[LAYOUT], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    out.index.unwrap().board.unwrap()
}

const STACK: &str = r#"    stackup rigid {
        finish = ENIG
        layer "F.Mask" = { kind = mask, thickness = 0.02, material = solder_mask }
        layer "F.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
        layer "prepreg 1" = { kind = prepreg, thickness = 0.1, material = fr4 }
        layer "In1.Cu" = { kind = copper, thickness = 0.0152, weight = 0.5, material = copper }
        layer "core" = { kind = core, thickness = 1.285, material = fr4 }
        layer "In2.Cu" = { kind = copper, thickness = 0.0152, weight = 0.5, material = copper }
        layer "prepreg 2" = { kind = prepreg, thickness = 0.1, material = fr4 }
        layer "B.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
        layer "B.Mask" = { kind = mask, thickness = 0.02, material = solder_mask }
    }
"#;

/// The USB board on four layers, its pair in a 90 Ω differential class, `extra` lines added.
fn hs_board(extra: &[&str]) -> Files {
    let mut files = files();
    let mut text = files[LAYOUT].clone();
    // A four-layer 1.6 mm build with 0.1 mm prepreg under each outer layer.
    let (from, to) = (text.find("    stackup rigid {").unwrap(), text.find("    outline =").unwrap());
    text.replace_range(from..to, STACK);
    let mut lines = vec!["    net_class usb = { diff_impedance = 90, diff_gap = 0.15 mm }".to_owned()];
    lines.extend(extra.iter().map(|l| format!("    {l}")));
    let anchor = "    net_class supply";
    let at = text.find(anchor).expect("the supply class line");
    text.insert_str(at, &format!("{}\n", lines.join("\n")));
    for n in ["USB_DN", "USB_DP"] {
        text = text.replacen(&format!("    net {n}\n"), &format!("    net {n} : usb\n"), 1);
    }
    files.insert(LAYOUT.into(), text);
    files
}

#[test]
fn diff_pair_routes_at_the_classs_solved_width_and_gap_and_checks_clean() {
    let mut files = hs_board(&[]);
    let b = index(&files);
    let usb = b.controlled.iter().find(|c| c.class == "usb").expect("the usb class row");
    let top = &usb.layers[0];
    assert_eq!((top.layer.as_str(), top.kind.as_str()), ("F.Cu", "microstrip"));
    assert!((top.diff_impedance.unwrap() - 90.0).abs() < 1.0, "{top:?}");
    let (w, s) = (top.width_mm.unwrap(), top.gap_mm.unwrap());
    let summary = run(&mut files, "board.diff-pair", json!({ "pair": "net:USB_DP" }));
    assert!(summary.starts_with(&format!("Routed USB on F.Cu at {w:.3} mm width, {s:.3} mm gap")), "{summary} vs {w} {s}");
    let b = index(&files);
    assert!(b.drc.is_empty(), "{:#?}", b.drc);
    assert!(b.unrouted.is_empty());
    // Every pair track sits at the solved width.
    let board = parse_board(LAYOUT, &files[LAYOUT]).board.unwrap();
    let dp = board.net("USB_DP").unwrap();
    assert!(board.tracks.values().filter(|t| t.net == Some(dp)).all(|t| (t.width as f64 * 1e-6 - w).abs() < 1e-6));
}

#[test]
fn tune_length_reaches_a_delay_target_with_each_pattern_and_tune_group_matches_a_group() {
    let mut files = hs_board(&[]);
    run(&mut files, "board.diff-pair", json!({ "pair": "net:USB_DP" }));
    let length = |files: &Files, net: &str| index(files).nets.iter().find(|r| r.name == net).map(|r| r.length_mm).unwrap();
    let before = length(&files, "USB_DN");
    assert!(before > 5.0, "{before}");
    // Delay target: about 1.5x the route's delay at 5.9 ps/mm, given in seconds.
    let target_ps = (before * 1.5 * 5.9).round();
    let summary = run(&mut files, "board.tune-length", json!({ "net": "net:USB_DN", "basis": "delay", "target_delay": target_ps * 1e-12, "pattern": "accordion" }));
    assert!(summary.starts_with("Tuned USB_DN from ") && summary.contains(" ps to ") && summary.contains("accordion"), "{summary}");
    // Its partner with sawtooth teeth to the same delay, then the board is clean, skew included.
    let summary = run(&mut files, "board.tune-length", json!({ "net": "net:USB_DP", "basis": "delay", "target_delay": target_ps * 1e-12, "tolerance_delay": 1e-12, "pattern": "sawtooth" }));
    assert!(summary.contains("sawtooth"), "{summary}");
    let b = index(&files);
    assert!(b.drc.is_empty(), "{:#?}", b.drc);

    // A delay group on the pair, 20 ps beyond the pair now: Tune group brings both members in.
    let mut grouped = files.clone();
    grouped.insert(LAYOUT.into(), {
        let tuned = &files[LAYOUT];
        let class_line = "    net_class usb = { diff_impedance = 90, diff_gap = 0.15 mm }\n";
        let rule = format!("    rule usb_len = {{ select = \":net(USB_DP, USB_DN)\", match_group = usb, match_target = {} ps, match_tolerance = 1 ps }}\n", target_ps + 20.0);
        tuned.replacen(class_line, &format!("{class_line}{rule}"), 1)
    });
    let b = index(&grouped);
    let g = b.length_groups.iter().find(|g| g.name == "usb").expect("the usb group row");
    assert_eq!((g.unit.as_str(), g.written), ("ps", true));
    assert!(g.members.iter().all(|m| m.status == "short"), "{g:?}");
    assert_eq!(b.drc.iter().filter(|r| r.code == "length-match").count(), 2, "{:#?}", b.drc);
    let summary = run(&mut grouped, "board.tune-group", json!({ "group": "usb", "pattern": "trombone" }));
    assert!(summary.starts_with("Tuned usb: Tuned USB_DN") && !summary.contains("not tuned"), "{summary}");
    let b = index(&grouped);
    let g = b.length_groups.iter().find(|g| g.name == "usb").unwrap();
    assert!(g.members.iter().all(|m| m.status == "matched"), "{g:?}");
    assert!(b.drc.is_empty(), "{:#?}", b.drc);
    let err = runtime::edit(LAYOUT, &grouped[LAYOUT], &grouped, &request("board.tune-group", json!({ "group": "usb" }))).unwrap_err();
    assert!(err.message.contains("no member of usb is short"), "{}", err.message);
}
