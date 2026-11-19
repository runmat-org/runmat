//! `assembly.component-configuration` writes the configuration the component uses, and a
//! configured component's label (Outline, BOM, STEP product name) reads its inputs in display
//! units.

use crate::edit::edit_json;
use crate::runtime::{compile_assembly, Files};
use runmat_cad::assembly::model::ConfigurationSelection;
use runmat_cad::ids::InputName;
use runmat_cad::syntax::{parse_assembly_in, print_assembly};
use runmat_cad::value::ValueExpr;
use serde_json::{json, Value};

const ARM: &str = "runmat 0.7\nstd 1\n\npart arm {\n    input length : length = 80 mm\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.length, 0 mm])\n        l2 = line(start = [input.length, 0 mm], end = [input.length, 10 mm])\n        l3 = line(start = [input.length, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 5 mm)\n    output arm = { body = feature.body.body }\n    configuration long = { length = 120 mm }\n}\n";

const RIG: &str = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part arm as arm\n}\n";
const PATH: &str = "/p/rig.assembly";

fn files() -> Files {
    [("/p/arm.part".to_owned(), ARM.to_owned())].into()
}

fn uuid(text: &str) -> String {
    let (index, _, _) = compile_assembly(PATH, text, &files()).unwrap_or_else(|d| panic!("{d:?}"));
    arm(&index)["uuid"].as_str().unwrap().to_owned()
}

/// The `arm` component's Outline row (row 0 is the assembly itself).
fn arm(index: &Value) -> Value {
    index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["path"] == "arm")
        .cloned()
        .unwrap_or_else(|| panic!("{index}"))
}

fn run(text: &str, configuration: &str) -> Value {
    let request = json!({ "command": "assembly.component-configuration", "values": { "configuration": configuration }, "selection": [format!("occurrence:{}", uuid(text))] });
    serde_json::from_str(&edit_json(PATH, text, &files(), &request.to_string())).unwrap()
}

#[test]
fn a_component_takes_another_configuration_of_its_file() {
    let out = run(RIG, "long");
    assert_eq!(out["ok"], true, "{out}");
    let text = out["files"][0]["text"].as_str().unwrap().to_owned();
    assert!(
        text.contains("configuration = { select = named, name = long }"),
        "{text}"
    );
    let (index, _, problems) = compile_assembly(PATH, &text, &files()).unwrap();
    assert!(
        problems.iter().all(|d| d.severity != "error"),
        "{problems:?}"
    );
    let row = arm(&index);
    assert!(row.to_string().contains("long"), "{row}");
    // Back to the default: the line goes.
    let back = run(&text, "default");
    assert_eq!(back["ok"], true, "{back}");
    assert!(
        !back["files"][0]["text"]
            .as_str()
            .unwrap()
            .contains("configuration ="),
        "{back}"
    );
    // A name the file doesn't have is refused with the ones it has.
    let wrong = run(RIG, "short");
    assert_eq!(wrong["ok"], false);
    assert!(
        wrong["message"].as_str().unwrap().contains("default, long"),
        "{wrong}"
    );
}

#[test]
fn a_configured_component_reads_its_inputs_in_display_units() {
    let parsed = parse_assembly_in(
        PATH,
        RIG,
        &crate::runtime::module_scope(PATH, RIG, &files()),
    );
    let mut document = parsed.document.unwrap();
    document.occurrences[0].configuration = ConfigurationSelection::Inputs {
        base: None,
        values: [(
            InputName::new("length"),
            ValueExpr::Quantity {
                value: 88.0,
                unit: "mm".into(),
            },
        )]
        .into(),
    };
    let text = print_assembly(&document, &parsed.uses).unwrap();
    let (index, _, _) = compile_assembly(PATH, &text, &files()).unwrap_or_else(|d| panic!("{d:?}"));
    // The Outline row reads the label the BOM and STEP product names read.
    assert_eq!(
        arm(&index)["configuration"],
        "length 88 mm",
        "{}",
        arm(&index)
    );
}

#[test]
fn an_input_written_in_inches_reads_as_written_in_properties() {
    let text = RIG.replace(
        "use part arm as arm",
        "use part arm as arm {\n        configuration = { select = inputs, values = { length = 3.5 in } }\n    }",
    );
    let (index, _, problems) =
        compile_assembly(PATH, &text, &files()).unwrap_or_else(|d| panic!("{d:?}"));
    assert!(
        problems.iter().all(|d| d.severity != "error"),
        "{problems:?}"
    );
    // Properties' Configuration reads this label; the Outline's detail reads the same text.
    assert_eq!(
        arm(&index)["configuration"],
        "length 3.5 in",
        "{}",
        arm(&index)
    );
}

#[test]
fn a_component_whose_part_does_not_compile_says_so_on_the_assembly() {
    // `distance = thickness` names an input without `input.`, so the part doesn't compile.
    let broken: Files = [(
        "/p/arm.part".to_owned(),
        ARM.replace("distance = 5 mm", "distance = length"),
    )]
    .into();
    let (index, _, problems) =
        compile_assembly(PATH, RIG, &broken).unwrap_or_else(|d| panic!("{d:?}"));
    // The index (what the language server and the Desktop's status read) says the same thing,
    // and nothing calls the part's file missing.
    let index_text = index["diagnostics"].to_string();
    assert!(
        index_text.contains("doesn't compile, so it draws nothing")
            && !index_text.contains("is not in the project"),
        "{index_text}"
    );
    assert!(
        problems
            .iter()
            .all(|d| !d.message.contains("is not in the project")),
        "{problems:?}"
    );
    let top: Vec<_> = problems
        .iter()
        .filter(|d| d.path == PATH && d.severity == "error")
        .collect();
    assert_eq!(top.len(), 1, "{problems:?}");
    assert!(
        top[0]
            .message
            .contains("doesn't compile, so it draws nothing")
            && top[0].message.contains("write `input.length`"),
        "{}",
        top[0].message
    );
    // Its occurrences are placed without their model, and their rows say why (the Outline's
    // mark, and the Desktop's "draws nothing" note over the Model view).
    let rows: Vec<&Value> = index["occurrences"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| {
            o["definition"]["file"]
                .as_str()
                .is_some_and(|f| f.ends_with("arm.part"))
        })
        .collect();
    assert!(!rows.is_empty(), "{}", index["occurrences"]);
    for row in rows {
        let problem = row["problem"].as_str().unwrap_or_default();
        assert!(
            problem.contains("doesn't compile") && problem.contains("input.length"),
            "{row}"
        );
    }
}
