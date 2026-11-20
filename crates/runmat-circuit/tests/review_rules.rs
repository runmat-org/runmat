//! The I²C and USB-C design-review rules on the showcase DAQ board (`tests/projects/daq`) and on
//! small designs that break each rule.
//!
//! The DAQ board has an SPI clock net named SCLK and a power-only USB-C port whose D+/D− pins are
//! waived with `no_erc`. Neither is a design-review finding.

use runmat_circuit::elab::{elaborate, ElabOptions, Netlist, Severity};
use runmat_circuit::fixture::standard_pool;
use runmat_circuit::ir::*;
use runmat_circuit::library::Pool;
use runmat_circuit::review::{self, ReviewOptions};
use runmat_circuit::runtime::{compile, CompileOptions, Files};
use std::path::PathBuf;

fn daq_files() -> Files {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projects/daq");
    ["daq.circuit", "daq.circuit.layout", "daq.circuit.lock"]
        .iter()
        .map(|name| {
            (
                format!("/p/{name}"),
                std::fs::read_to_string(here.join(name)).expect("the showcase is checked in"),
            )
        })
        .collect()
}

/// The review rows of `path`, as `code: message`.
fn review_rows(files: &Files, path: &str) -> Vec<(String, String, String)> {
    let out = compile(path, &files[path], files, &CompileOptions::default());
    let index = out.index.expect("compiles");
    index
        .review
        .iter()
        .map(|r| (r.code.clone(), r.severity.clone(), r.message.clone()))
        .collect()
}

#[test]
fn the_daq_board_has_no_i2c_or_usb_findings_and_no_review_errors() {
    let files = daq_files();
    for path in ["/p/daq.circuit", "/p/daq.circuit.layout"] {
        let rows = review_rows(&files, path);
        let bus: Vec<_> = rows
            .iter()
            .filter(|(code, _, _)| code.starts_with("i2c.") || code.starts_with("usb-c.data"))
            .collect();
        assert!(bus.is_empty(), "{path}: {bus:#?}");
        let errors: Vec<_> = rows.iter().filter(|(_, sev, _)| sev == "error").collect();
        assert!(errors.is_empty(), "{path}: review errors {errors:#?}");
    }
}

#[test]
fn the_daq_board_without_its_usb_data_waivers_reports_the_open_pins() {
    let mut files = daq_files();
    let circuit = "/p/daq.circuit";
    let text = files[circuit].replace("no_erc = [J1.DP, J1.DN, ", "no_erc = [");
    assert_ne!(text, files[circuit], "the waiver is in the showcase");
    files.insert(circuit.into(), text);
    let rows = review_rows(&files, circuit);
    assert!(
        rows.iter()
            .any(|(code, sev, msg)| code == "usb-c.data-connected"
                && sev == "error"
                && msg.contains("not connected")),
        "{rows:#?}"
    );
}

fn elab(d: &Design, pool: &Pool) -> Netlist {
    let nl = elaborate(d, pool, &ElabOptions::default(), None);
    assert!(nl.diagnostics.is_empty(), "{:#?}", nl.diagnostics);
    nl
}

/// An MCU and an I²C sensor on a supply, with the bus on nets whose names don't say I²C, and an
/// SPI-style clock named SCLK between two GPIOs.
fn i2c_design() -> Design {
    let mut d = Design::new("bus");
    let mut b = DesignBuilder::new(&mut d, "main");
    b.global("GND", None);
    b.component("U1", "MCU-QFN32");
    b.component("U2", "TMP117");
    b.wire("+3V3", &[("U1", "VDD"), ("U2", "V+")]);
    b.wire("GND", &[("U1", "VSS"), ("U2", "GND"), ("U2", "ADD0")]);
    b.wire("BUS_A", &[("U1", "SCL"), ("U2", "SCL")]);
    b.wire("BUS_B", &[("U1", "SDA"), ("U2", "SDA")]);
    b.wire("SCLK", &[("U1", "PA0"), ("U1", "PA1")]);
    d
}

#[test]
fn an_i2c_line_is_found_by_the_parts_pins_not_by_its_net_name() {
    let pool = standard_pool();
    let nl = elab(&i2c_design(), &pool);
    let f = review::i2c(&nl, &pool, &ReviewOptions::default());
    let mut lines: Vec<&str> = f
        .iter()
        .filter(|x| x.rule == "pull-up")
        .map(|x| x.items[0].as_str())
        .collect();
    lines.sort();
    assert_eq!(lines, vec!["BUS_A", "BUS_B"], "{f:#?}");
    assert!(f.iter().all(|x| x.severity == Severity::Error));
    assert!(
        !f.iter().any(|x| x.items.iter().any(|i| i == "SCLK")),
        "an SPI clock named SCLK is not an I²C line: {f:#?}"
    );
}

fn usb_design(waive: bool) -> Design {
    let mut d = Design::new("usb");
    let mut b = DesignBuilder::new(&mut d, "main");
    b.global("GND", None);
    b.component("J1", "USB4105");
    b.component("R1", "R0402-10K");
    b.component("R2", "R0402-10K");
    b.wire("VBUS", &[("J1", "VBUS")]);
    b.wire(
        "GND",
        &[("J1", "GND"), ("J1", "SHIELD"), ("R1", "2"), ("R2", "2")],
    );
    b.wire("CC1", &[("J1", "CC1"), ("R1", "1")]);
    b.wire("CC2", &[("J1", "CC2"), ("R2", "1")]);
    for p in ["SBU1", "SBU2"] {
        b.no_erc("J1", p);
    }
    if waive {
        b.no_erc("J1", "DP");
        b.no_erc("J1", "DN");
    }
    d
}

#[test]
fn usb_data_pins_are_reported_open_unless_waived() {
    let pool = standard_pool();
    let o = ReviewOptions::default();
    let open = review::usb_c(&elab(&usb_design(false), &pool), &pool, &o);
    assert!(
        open.iter()
            .any(|x| x.rule == "data-connected" && x.severity == Severity::Error),
        "{open:#?}"
    );
    let waived = review::usb_c(&elab(&usb_design(true), &pool), &pool, &o);
    assert!(
        !waived.iter().any(|x| x.rule.starts_with("data")),
        "a power-only port waives D+/D−: {waived:#?}"
    );
}
