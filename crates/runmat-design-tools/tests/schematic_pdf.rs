//! `design.export` with `format: "pdf"` on a circuit: the schematic as one PDF set, a page per
//! schematic page at its paper size, each with its frame and title block.

use base64::Engine;
use runmat_design_tools::{call, Files};
use serde_json::json;

const CIRCUIT: &str = r#"runmat 0.7
std 1

circuit reg {
    root = main

    sheet supply {
        port VOUT : out
        component U1 = { part = "LDO-3V3-SOT23" }
        component C1 = { part = "C0603-1U-50V" }
        net GND = { global = true }
        connect = [port.VOUT, U1.VOUT, "C1.1"]
        connect = [net.GND, U1.GND, "C1.2"]
        no_erc = [U1.NC, U1.VIN, U1.EN]
        text note1 = "3.3 V rail"
    }

    sheet main {
        instance power = { sheet = supply }
        component R1 = { part = "R0402-10K" }
        net GND = { global = true }
        connect = [instance.power.VOUT, "R1.1"]
        connect = [net.GND, "R1.2"]
    }
}
"#;

#[test]
fn a_circuit_exports_every_schematic_page_with_its_title_block_in_one_pdf() {
    let files: Files = [("/p/reg.circuit".to_owned(), CIRCUIT.to_owned())].into_iter().collect();
    let out = call("design.export", json!({ "path": "/p/reg.circuit", "format": "pdf" }), &files).unwrap();
    let list = out["files"].as_array().unwrap();
    assert_eq!(list.len(), 1, "{list:?}");
    assert_eq!(list[0]["name"], "reg-schematic.pdf");
    let pdf = base64::engine::general_purpose::STANDARD.decode(list[0]["content"].as_str().unwrap()).unwrap();
    assert!(pdf.starts_with(b"%PDF-1.4") && pdf.ends_with(b"%%EOF\n"));
    let text = String::from_utf8_lossy(&pdf).into_owned();
    assert!(text.contains("/Type /Pages /Kids [4 0 R 6 0 R] /Count 2"), "two pages");
    assert_eq!(text.matches("/Type /Page /Parent").count(), 2);
    // Each page's title block: the design, the sheet, the page of the set.
    for words in ["(reg)", "(main)", "(supply)", "(1 of 2)", "(2 of 2)", "(Page)", "(A4)"] {
        assert!(text.contains(words), "{words}");
    }
    // The drawing itself: designators, a pin name, the sheet symbol's path, the note.
    for words in ["(U1)", "(C1)", "(R1)", "(VOUT)", "(/power)", "(3.3 V rail)"] {
        assert!(text.contains(words), "{words}");
    }
    // Every xref offset points at its object.
    let xref = text.rfind("startxref\n").unwrap();
    let at: usize = text[xref + 10..].lines().next().unwrap().parse().unwrap();
    assert!(pdf[at..].starts_with(b"xref"), "the xref offset counts bytes");
}
