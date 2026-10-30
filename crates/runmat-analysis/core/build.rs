//! Generates `ANALYSIS_IDENTITIES` from the capability ledger (docs/simulation/ledger.json).
//!
//! The ledger is the one place that says whether a family runs a procedure in this release or
//! lists it as planned with a pending id. Every ledger capability with an `identity` becomes one
//! row, in ledger order: `release` rows run, `pending` rows carry their capability id.
use std::{env, fs, path::PathBuf};

fn family(label: &str) -> &'static str {
    match label {
        "structural" => "STRUCTURAL",
        "thermal" => "THERMAL",
        "electromagnetic" => "EM",
        "acoustic" => "ACOUSTIC",
        "fluid" => "FLUID",
        "thermomechanical" => "TM",
        "electrothermal" => "ET",
        "conjugate_heat_transfer" => "CHT",
        "fluid_structure_interaction" => "FSI",
        other => panic!("ledger: unknown family {other:?}"),
    }
}

fn procedure(label: &str) -> &'static str {
    match label {
        "static" => "Static",
        "modal" => "ModalExtraction",
        "prestressed_modal" => "PrestressedModalExtraction",
        "buckling" => "EigenvalueBuckling",
        "transient" => "TransientResponse",
        "harmonic" => "HarmonicResponse",
        "prestressed_harmonic" => "PrestressedHarmonicResponse",
        "response_spectrum" => "ModalResponseSpectrum",
        "random_response" => "RandomResponse",
        "nonlinear" => "Nonlinear",
        "adaptive" => "Adaptive",
        other => panic!("ledger: unknown procedure {other:?}"),
    }
}

fn space(label: &str) -> &'static str {
    match label {
        "direct" => "Direct",
        "modal_reduced" => "ModalReduced",
        other => panic!("ledger: unknown solution space {other:?}"),
    }
}

fn main() {
    let ledger = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../docs/simulation/ledger.json");
    println!("cargo:rerun-if-changed={}", ledger.display());
    let text = fs::read_to_string(&ledger).expect("docs/simulation/ledger.json");
    let ledger: serde_json::Value = serde_json::from_str(&text).expect("ledger.json is JSON");
    let mut rows = String::new();
    for capability in ledger["capabilities"].as_array().expect("ledger capabilities") {
        let Some(identity) = capability.get("identity") else {
            continue;
        };
        let id = capability["id"].as_str().expect("capability id");
        let status = match capability["status"].as_str() {
            Some("release") => "Runs".to_string(),
            Some("pending") => format!("Pending({id:?})"),
            other => panic!("ledger: {id} has status {other:?}"),
        };
        rows.push_str(&format!(
            "    row({}, P::{}, {}, {}),\n",
            family(identity["family"].as_str().expect("identity family")),
            procedure(identity["procedure"].as_str().expect("identity procedure")),
            space(identity["solution_space"].as_str().expect("identity solution space")),
            status
        ));
    }
    let source = format!(
        "/// The one closed registry of analysis identities the language can author, generated from\n\
         /// docs/simulation/ledger.json. `runmat check`, the native execution gate and the docs\n\
         /// status matrix all read this table; a combination that is absent is not a procedure of\n\
         /// that family.\n\
         pub const ANALYSIS_IDENTITIES: &[AnalysisIdentity] = &[\n{rows}];\n"
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("analysis_identities.rs"),
        source,
    )
    .unwrap();
}
