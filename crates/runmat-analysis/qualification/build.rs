//! Generates the requirement register and the scope inventory from the capability ledger
//! (docs/simulation/ledger.json), the one source for requirement ids, obligations, envelopes and
//! scope claims. Envelope blocks are ordered cross products, expanded here in ledger order.
use serde_json::Value;
use std::{env, fs, path::PathBuf};

/// `snake_case` wire name to the Rust variant name (the inverse of `requirement::snake_name`).
fn variant(wire: &str) -> String {
    wire.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn text(value: &Value) -> String {
    format!("{:?}", value.as_str().expect("ledger string"))
}

fn texts(value: &Value) -> String {
    let items: Vec<String> = value
        .as_array()
        .expect("ledger string list")
        .iter()
        .map(text)
        .collect();
    format!("&[{}]", items.join(", "))
}

fn strings(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .expect("ledger list")
        .iter()
        .map(|item| item.as_str().expect("ledger string"))
        .collect()
}

fn main() {
    let path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../docs/simulation/ledger.json");
    println!("cargo:rerun-if-changed={}", path.display());
    let ledger: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("docs/simulation/ledger.json"))
            .expect("ledger.json is JSON");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    let mut source = String::from(
        "use crate::requirement::{Envelope, EvidenceKind, ExecutionEnvelope, Formulation, MeshOrder, Obligation, Requirement, Stage};\n\
         pub(crate) const ALL: &[&[Requirement]] = &[REQUIREMENTS];\n\
         const REQUIREMENTS: &[Requirement] = &[\n",
    );
    for requirement in ledger["requirements"]
        .as_array()
        .expect("ledger requirements")
    {
        let stages: Vec<String> = strings(&requirement["stages"])
            .iter()
            .map(|stage| format!("Stage::{}", variant(stage)))
            .collect();
        source.push_str(&format!(
            "Requirement {{ id: {}, title: {}, stages: &[{}], acceptance: {}, owners: {}, dependencies: {}, obligations: &[\n",
            text(&requirement["id"]),
            text(&requirement["title"]),
            stages.join(", "),
            text(&requirement["acceptance"]),
            texts(&requirement["owners"]),
            texts(&requirement["dependencies"]),
        ));
        for obligation in requirement["obligations"].as_array().expect("obligations") {
            let mut envelopes = Vec::new();
            for block in obligation["envelopes"].as_array().expect("envelope blocks") {
                for formulation in strings(&block["formulations"]) {
                    for mesh in strings(&block["meshes"]) {
                        for execution in strings(&block["executions"]) {
                            envelopes.push(format!(
                                "Envelope {{ formulation: Formulation::{}, mesh: MeshOrder::{}, execution: ExecutionEnvelope::{} }}",
                                variant(formulation),
                                variant(mesh),
                                variant(execution)
                            ));
                        }
                    }
                }
            }
            source.push_str(&format!(
                "    Obligation {{ id: {}, criterion: {}, evidence_kind: EvidenceKind::{}, required_envelopes: &[\n        {}\n    ] }},\n",
                text(&obligation["id"]),
                text(&obligation["criterion"]),
                variant(obligation["evidence_kind"].as_str().expect("evidence kind")),
                envelopes.join(",\n        ")
            ));
        }
        source.push_str("] },\n");
    }
    source.push_str("];\n");
    fs::write(out.join("requirements.rs"), source).unwrap();

    // The scope inventory, in the tab-separated form `scope` parses.
    let mut inventory = String::from("id\tsource\tcapability\tmaps\n");
    for claim in ledger["claims"].as_array().expect("ledger claims") {
        let field = |key: &str| claim[key].as_str().expect("claim field").to_string();
        inventory.push_str(
            &[
                field("id"),
                field("source"),
                field("capability"),
                field("maps"),
            ]
            .join("\t"),
        );
        inventory.push('\n');
    }
    fs::write(out.join("inventory.tsv"), inventory).unwrap();
}
