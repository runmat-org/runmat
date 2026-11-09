//! The circuit library: catalog parts embedded as `part` declarations, and `library.place`.

use runmat_circuit::runtime::{self, CompileOptions, EditRequest, Files};
use runmat_circuit::syntax::{parse_circuit, print_circuit};
use serde_json::{json, Value};

const CIRCUIT: &str = "runmat 0.7\nstd 1\n\ncircuit board {\n    root = main\n\n    sheet main {\n        component R1 = { part = \"R0402-10K\" }\n        net GND = { global = true }\n        connect = [net.GND, \"R1.2\"]\n    }\n\n    sheet empty {\n    }\n}\n";

fn buck() -> Value {
    json!({
        "mpn": "BUCK-60V-3A-QFN",
        "manufacturer": "Generic",
        "description": "60 V 3 A synchronous buck, QFN-16 3 x 3 mm",
        "category": "regulator",
        "package": "QFN16_3mm_P0.5_EP1.7",
        "datasheet": "https://example.com/datasheets/buck-60v-3a-qfn.pdf",
        "params": { "vin_max": "60 V", "iout_max": "3 A" },
        "pins": [
            { "name": "VIN", "pads": ["1", "2"], "type": "power_in" },
            { "name": "EN", "pads": ["3"], "type": "input" },
            { "name": "PG", "pads": ["4"], "type": "open_collector" },
            { "name": "FB", "pads": ["5"], "type": "input" },
            { "name": "SS", "pads": ["6"], "type": "passive" },
            { "name": "AGND", "pads": ["7"], "type": "power_in" },
            { "name": "VCC", "pads": ["8"], "type": "power_out" },
            { "name": "BST", "pads": ["9"], "type": "passive" },
            { "name": "SW", "pads": ["10", "11", "12"], "type": "power_out" },
            { "name": "PGND", "pads": ["13", "14", "15", "16", "17"], "type": "power_in" }
        ]
    })
}

fn place(text: &str, values: Value, selection: &[&str]) -> Result<String, String> {
    let files: Files = [("/p/board.circuit".to_string(), text.to_string())].into();
    let request = EditRequest {
        command: "library.place".into(),
        values: values.as_object().cloned().unwrap(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    };
    let out = runtime::edit("/p/board.circuit", text, &files, &request).map_err(|e| e.message)?;
    assert_eq!(out.files.len(), 1);
    Ok(out.files[0].text.clone())
}

#[test]
fn placing_a_catalog_part_embeds_it_and_adds_the_next_designator() {
    let text = place(CIRCUIT, json!({ "part": buck() }), &[]).unwrap();
    assert!(text.contains("part \"BUCK-60V-3A-QFN\" = {"), "{text}");
    assert!(text.contains("        component R1 = { part = \"R0402-10K\" }\n        component U1 = { part = \"BUCK-60V-3A-QFN\" }\n"), "{text}");
    // The edited file reads, prints back the same, and elaborates the part on its package.
    let design = parse_circuit("board.circuit", &text).design.expect("reads");
    let printed = print_circuit(&design).unwrap();
    let reread = parse_circuit("board.circuit", &printed)
        .design
        .expect("the printed design reads");
    assert_eq!(print_circuit(&reread).unwrap(), printed);
    assert_eq!(reread.parts, design.parts);
    let part = &design.parts["BUCK-60V-3A-QFN"];
    assert_eq!(part.pins[8].pads, ["10", "11", "12"]);
    let files: Files = [("/p/board.circuit".to_string(), text.clone())].into();
    let out = runtime::compile(
        "/p/board.circuit",
        &text,
        &files,
        &CompileOptions::default(),
    );
    assert!(out.index.is_some(), "{:?}", out.diagnostics);
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| d.message.contains("BUCK-60V-3A-QFN") && d.message.contains("pool")),
        "{:?}",
        out.diagnostics
    );
    // A second placement reuses the declaration.
    let again = place(&text, json!({ "part": buck() }), &[]).unwrap();
    assert_eq!(again.matches("part \"BUCK-60V-3A-QFN\" = {").count(), 1);
    assert!(again.contains("component U2 = { part = \"BUCK-60V-3A-QFN\" }"));
}

#[test]
fn a_built_in_part_is_placed_without_a_declaration_on_the_selected_empty_sheet() {
    let resistor = json!({ "mpn": "R0402-10K", "manufacturer": "Generic", "category": "resistor", "package": "R0402",
        "pins": [{ "name": "1", "pads": ["1"] }, { "name": "2", "pads": ["2"] }] });
    let text = place(CIRCUIT, json!({ "part": resistor }), &["sheet:/empty"]).unwrap();
    assert!(!text.contains("part \"R0402-10K\" = "), "{text}");
    assert!(
        text.contains("    sheet empty {\n        component R2 = { part = \"R0402-10K\" }\n    }"),
        "{text}"
    );
}

#[test]
fn a_bad_record_or_a_taken_designator_is_refused() {
    let mut bad = buck();
    bad["pins"][0]["pads"] = json!(["99"]);
    assert!(place(CIRCUIT, json!({ "part": bad }), &[])
        .unwrap_err()
        .contains("pad `99`"));
    let taken = place(CIRCUIT, json!({ "part": buck(), "designator": "R1" }), &[]).unwrap_err();
    assert!(taken.contains("R1 is already"), "{taken}");
    let named = place(CIRCUIT, json!({ "part": buck(), "designator": "U9" }), &[]).unwrap();
    assert!(named.contains("component U9 = "));
}

#[test]
fn a_declaration_with_an_unknown_field_or_package_is_a_diagnostic() {
    let with = |body: &str| {
        CIRCUIT.replace(
            "    root = main\n",
            &format!("    root = main\n    part \"X1\" = {{ {body} }}\n"),
        )
    };
    let read = parse_circuit("b.circuit", &with("manufacturer = \"M\", package = \"SOT23-5\", colour = red, pins = [{ name = A, pad = \"1\" }]"));
    assert!(
        read.diagnostics
            .iter()
            .any(|d| d.message.contains("no `colour`")),
        "{:?}",
        read.diagnostics
    );
    let read = parse_circuit(
        "b.circuit",
        &with("manufacturer = \"M\", package = \"BLOB-3\", pins = [{ name = A, pad = \"1\" }]"),
    );
    assert!(
        read.diagnostics
            .iter()
            .any(|d| d.message.contains("footprint generator")),
        "{:?}",
        read.diagnostics
    );
    let read = parse_circuit(
        "b.circuit",
        &with("manufacturer = \"M\", package = \"SOT23-5\", pins = [{ name = A, pad = \"1\" }]"),
    );
    assert!(read.diagnostics.is_empty(), "{:?}", read.diagnostics);
}

#[test]
fn a_part_on_a_footprint_file_opened_in_place_compiles_and_previews() {
    use runmat_circuit::library::{catalog::CatalogPart, footprint, preview::preview_with};
    let kicad_mod =
        runmat_circuit::kicad::footprint::write_mod(&footprint::from_string("SOT23-5").unwrap());
    let ldo = json!({ "mpn": "LDO_3V3", "manufacturer": "", "prefix": "U", "package": "libs/Power.pretty/SOT-23-5.kicad_mod",
        "pins": [{ "name": "VIN", "pads": ["1"], "type": "power_in" }, { "name": "GND", "pads": ["2"], "type": "power_in" },
                 { "name": "EN", "pads": ["3"], "type": "input" }, { "name": "NC", "pads": ["4"], "type": "no_connect" },
                 { "name": "VOUT", "pads": ["5"], "type": "power_out" }] });
    let text = place(CIRCUIT, json!({ "part": ldo.clone() }), &[]).unwrap();
    assert!(
        text.contains("package = \"libs/Power.pretty/SOT-23-5.kicad_mod\""),
        "{text}"
    );
    let deps = runtime::dependencies("/p/board.circuit", &text);
    assert!(
        deps.contains(&"/p/libs/Power.pretty/SOT-23-5.kicad_mod".to_string()),
        "{deps:?}"
    );
    let mut files: Files = [("/p/board.circuit".to_string(), text.clone())].into();
    let missing = runtime::compile(
        "/p/board.circuit",
        &text,
        &files,
        &CompileOptions::default(),
    );
    assert!(
        missing
            .diagnostics
            .iter()
            .any(|d| d.message.contains("cannot be read")),
        "{:?}",
        missing.diagnostics
    );
    files.insert(
        "/p/libs/Power.pretty/SOT-23-5.kicad_mod".into(),
        kicad_mod.clone(),
    );
    let out = runtime::compile(
        "/p/board.circuit",
        &text,
        &files,
        &CompileOptions::default(),
    );
    assert!(out.index.is_some(), "{:?}", out.diagnostics);
    let part: CatalogPart = serde_json::from_value(ldo).unwrap();
    let preview = preview_with(&part, &|p| {
        (p == "libs/Power.pretty/SOT-23-5.kicad_mod").then(|| kicad_mod.clone())
    })
    .unwrap();
    assert_eq!(preview.footprint.pads.len(), 5);
    assert_eq!(preview.symbol.pins.len(), 5);
}
