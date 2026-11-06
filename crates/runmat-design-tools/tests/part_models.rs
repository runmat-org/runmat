//! SPICE models on parts (PCB-29): a part declared with a `spice` model simulates with no
//! `model` in the simulation section, and a section's own `model` takes precedence over the
//! part's. Runs the flight controller's 5 V buck tolerance study on the averaged buck model.

use runmat_design_tools::{call, Files};
use serde_json::{json, Value};

const FC: &str = include_str!("../../runmat-cad/wasm/tests/drone/electrical/flight_controller/flight_controller.circuit");
const MODELS: &str = include_str!("../../runmat-cad/wasm/tests/drone/electrical/flight_controller/fc_models.lib");

const SECTION_BUCK: &str = "        model \"BUCK-36V-2A\" = { library = \"fc_models.lib\", subckt = BUCK36V_AVG, pins = [VIN, EN, BST, SW, FB, GND] }\n";
const SECTION_LDO: &str = "        model \"LDO-3V3-SOT23\" = { library = \"fc_models.lib\", subckt = LDO33, pins = [VIN, EN, VOUT, GND] }\n";

/// The regulators as catalog parts that carry their SPICE models.
fn declarations(buck_model: &str) -> String {
    format!(
        r#"    part "BUCK-36V-2A" = {{
        manufacturer = Generic,
        description = "36 V 2 A synchronous buck, 500 kHz, 0.8 V reference, SOT-23-6",
        category = regulator,
        package = "SOT23-6",
        pins = [
            {{ name = BST, pad = "1", type = passive }},
            {{ name = GND, pad = "2", type = power_in }},
            {{ name = FB, pad = "3", type = input }},
            {{ name = EN, pad = "4", type = input }},
            {{ name = VIN, pad = "5", type = power_in }},
            {{ name = SW, pad = "6", type = power_out }}
        ],
        spice = {{ model = {buck_model}, library = "fc_models.lib", nodes = [VIN, EN, BST, SW, FB, GND] }}
    }}
    part "LDO-3V3-SOT23" = {{
        manufacturer = Generic,
        description = "3.3 V 300 mA low-dropout regulator, 40 V input, SOT-23-5",
        category = regulator,
        package = "SOT23-5",
        pins = [
            {{ name = VIN, pad = "1", type = power_in }},
            {{ name = GND, pad = "2", type = power_in }},
            {{ name = EN, pad = "3", type = input }},
            {{ name = NC, pad = "4", type = no_connect }},
            {{ name = VOUT, pad = "5", type = power_out }}
        ],
        spice = {{ model = LDO33, library = "fc_models.lib", nodes = [VIN, EN, VOUT, GND] }}
    }}
"#
    )
}

/// The flight controller with its regulators declared as `buck_model` parts; `section_models`
/// keeps the tolerance study's own `model` lines. The study runs 40 Monte Carlo runs.
fn circuit(buck_model: &str, section_models: bool) -> String {
    let mut text = FC.replace(
        "    root = main\n",
        &format!("    root = main\n{}", declarations(buck_model)),
    );
    assert!(
        text.contains("part \"BUCK-36V-2A\" = {"),
        "the flight controller's root line changed"
    );
    if !section_models {
        text = text.replace(SECTION_BUCK, "").replace(SECTION_LDO, "");
        assert!(
            !text.contains("subckt = BUCK36V_AVG"),
            "the tolerance study's model lines changed"
        );
    }
    text.replace("runs = 2000, seed = 11", "runs = 40, seed = 11")
}

fn run(text: String) -> Value {
    let files: Files = [
        ("/fc/flight_controller.circuit".to_string(), text),
        ("/fc/fc_models.lib".to_string(), MODELS.to_string()),
    ]
    .into();
    call(
        "design.simulate",
        json!({ "path": "/fc/flight_controller.circuit", "simulation": "buck_tolerance" }),
        &files,
    )
    .unwrap_or_else(|e| panic!("design.simulate: {e:?}"))
}

fn v5(out: &Value) -> f64 {
    let m = out["measures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"] == "v5")
        .expect("the v5 measure");
    m["value"]
        .as_f64()
        .unwrap_or_else(|| panic!("v5 has a value: {m}"))
}

#[test]
fn a_part_with_a_spice_model_simulates_with_no_model_in_the_section() {
    let out = run(circuit("BUCK36V_AVG", false));
    let v = v5(&out);
    assert!(
        (4.9..5.1).contains(&v),
        "the averaged buck regulates 5 V from the part's own model: {v}"
    );
}

#[test]
fn a_sections_model_takes_precedence_over_the_parts() {
    // The part names a subcircuit the library lacks; the section's own model is the one used.
    let out = run(circuit("NO_SUCH_BUCK", true));
    let v = v5(&out);
    assert!((4.9..5.1).contains(&v), "{v}");
}

#[test]
fn a_parts_model_library_resolves_relative_to_the_circuit() {
    // A KiCad library's model file, rebased from the symbol library's folder: ../models/….
    let text = circuit("BUCK36V_AVG", false).replace(
        "library = \"fc_models.lib\"",
        "library = \"../models/fc_models.lib\"",
    );
    let files: Files = [
        ("/fc/boards/flight_controller.circuit".to_string(), text),
        (
            "/fc/models/fc_models.lib".to_string(),
            MODELS.to_string(),
        ),
    ]
    .into();
    let out = call(
        "design.simulate",
        json!({ "path": "/fc/boards/flight_controller.circuit", "simulation": "buck_tolerance" }),
        &files,
    )
    .unwrap_or_else(|e| panic!("design.simulate: {e:?}"));
    let v = v5(&out);
    assert!((4.9..5.1).contains(&v), "{v}");
}
