//! One wrong-unit diagnostic in every language: the same wrong-unit cases written in a `.part`,
//! a `.circuit.layout` and a `.fea` file report the shared message for their field, underline
//! the value, say which language raised them, and offer the same quick fixes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lsp_types::{CodeActionOrCommand, Diagnostic, NumberOrString, Range, Url};
use runmat_types::physical::{
    parse_unit_expression, unit_fixes, unit_mismatch_message, DynamicDimension, FoundUnit,
};

use super::design::{design_dialect, DesignDocumentAnalysis, DesignFiles, DesignValues};
use super::fea::FeaDocumentAnalysis;
use super::position::position_to_offset;

const LAYOUT: &str =
    include_str!("../../../runmat-circuit/tests/projects/io/io.circuit.layout");
const CIRCUIT: &str = include_str!("../../../runmat-circuit/tests/projects/io/io.circuit");
const LOCK: &str = include_str!("../../../runmat-circuit/tests/projects/io/io.circuit.lock");

/// A wrong-unit case: the value as written, what it must measure, and the message's tail.
struct Case {
    value: &'static str,
    expected: &'static str,
}

const LENGTH_CASES: &[Case] = &[
    // Mass for a length.
    Case {
        value: "10 kg",
        expected: "m",
    },
    // An angle unit for a length.
    Case {
        value: "10 deg",
        expected: "m",
    },
];

fn dimension(unit: &str) -> DynamicDimension {
    DynamicDimension::from_unit(parse_unit_expression(unit).unwrap())
}

/// The shared message for `case` at the field `subject`.
fn expected_message(subject: &str, case: &Case) -> String {
    let found = match case.value.split_once(' ') {
        Some((_, unit)) => FoundUnit::Authored(unit, dimension(unit)),
        None => FoundUnit::Missing,
    };
    unit_mismatch_message(subject, dimension(case.expected), found).message
}

/// The one diagnostic with `message`, its underlined text and its quick fixes' texts.
fn report(
    uri: &Url,
    source: &str,
    diagnostics: &[Diagnostic],
    actions: impl Fn(Range) -> Vec<CodeActionOrCommand>,
    message: &str,
) -> (String, String, Vec<String>) {
    let d = diagnostics
        .iter()
        .find(|d| d.message == message)
        .unwrap_or_else(|| panic!("`{message}` not in {diagnostics:#?}"));
    let start = position_to_offset(source, &d.range.start);
    let end = position_to_offset(source, &d.range.end);
    let code = match &d.code {
        Some(NumberOrString::String(code)) => code.clone(),
        other => panic!("diagnostic code {other:?}"),
    };
    let fixes = actions(d.range)
        .into_iter()
        .filter_map(|a| match a {
            CodeActionOrCommand::CodeAction(a) => a.edit?.changes?.remove(uri)?.pop(),
            _ => None,
        })
        .map(|e| e.new_text)
        .collect();
    (source[start..end].to_string(), code, fixes)
}

fn design(path: &str, text: &str, files: &DesignFiles) -> DesignDocumentAnalysis {
    let uri = Url::parse(&format!("file://{path}")).unwrap();
    let dialect = design_dialect(&uri).expect("a design file");
    DesignDocumentAnalysis::analyze(&uri, dialect, text, files, &DesignValues::new())
}

fn check_part(case: &Case) {
    let path = "/p/plate.part";
    let text = format!(
        "runmat 0.7\nstd 1\n\npart plate {{\n    sketch s1 on origin.xy {{\n        l1 = line(start = [0 mm, 0 mm], end = [20 mm, 0 mm])\n        l2 = line(start = [20 mm, 0 mm], end = [20 mm, 10 mm])\n        l3 = line(start = [20 mm, 10 mm], end = [0 mm, 10 mm])\n        l4 = line(start = [0 mm, 10 mm], end = [0 mm, 0 mm])\n    }}\n    feature body = extrude(profile = sketch.s1, distance = {})\n}}\n",
        case.value
    );
    let uri = Url::parse(&format!("file://{path}")).unwrap();
    let analysis = design(path, &text, &DesignFiles::new());
    let message = expected_message("distance", case);
    let (underlined, code, fixes) = report(
        &uri,
        &text,
        &analysis.diagnostics(),
        |range| analysis.code_actions(range),
        &message,
    );
    assert_eq!(underlined, case.value, ".part span");
    assert_eq!(code, "cad.value.unit");
    assert_eq!(fixes, unit_fixes(case.value, case.expected), ".part fixes");
}

fn check_layout(case: &Case) {
    let path = "/p/io.circuit.layout";
    let text = LAYOUT.replacen("clearance = 0.15", &format!("clearance = {}", case.value), 1);
    assert_ne!(text, LAYOUT);
    let files = DesignFiles::from([
        ("/p/io.circuit".to_string(), CIRCUIT.to_string()),
        ("/p/io.circuit.lock".to_string(), LOCK.to_string()),
        (path.to_string(), text.clone()),
    ]);
    let uri = Url::parse(&format!("file://{path}")).unwrap();
    let analysis = design(path, &text, &files);
    let message = expected_message("clearance", case);
    let (underlined, code, fixes) = report(
        &uri,
        &text,
        &analysis.diagnostics(),
        |range| analysis.code_actions(range),
        &message,
    );
    assert_eq!(underlined, case.value, ".circuit.layout span");
    assert_eq!(code, "circuit.value.unit");
    assert_eq!(fixes, unit_fixes(case.value, case.expected), ".circuit.layout fixes");
}

fn check_fea(field: &str, subject: &str, case: &Case, code_prefix: &str) {
    let temporary = tempfile::tempdir().unwrap();
    let main_path = temporary.path().join("main.fea");
    let size = if field == "target_size" { case.value } else { "0.1 m" };
    let modulus = if field == "youngs_modulus" { case.value } else { "200 GPa" };
    let source = format!(
        r#"material steel {{
    elastic {{
        youngs_modulus = {modulus}
        poisson_ratio = 0.3
    }}
}}
geometry part {{
    construct {{
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }}
    region body : solid {{
        select result
        expect exactly_one
    }}
}}
mesh volume on geometry.part {{
    tetrahedra {{
        order = linear
        target_size = {size}
        maximum_deviation = 0.01 m
    }}
}}
"#
    );
    std::fs::write(&main_path, &source).unwrap();
    let roots: Vec<PathBuf> = vec![temporary.path().to_path_buf()];
    let analysis = FeaDocumentAnalysis::from_checked(
        FeaDocumentAnalysis::prepare_with_overlays(&source, &main_path, &roots, &[])
            .unwrap()
            .check(Vec::new(), Vec::new())
            .unwrap(),
        BTreeMap::new(),
    );
    let uri = Url::from_file_path(&main_path).unwrap();
    let message = expected_message(subject, case);
    let (underlined, code, fixes) = report(
        &uri,
        &source,
        &analysis.diagnostics(&source),
        |range| analysis.code_actions(&uri, &source, range),
        &message,
    );
    assert_eq!(underlined, case.value, ".fea span");
    assert!(code.starts_with(code_prefix), "{code}");
    assert_eq!(fixes, unit_fixes(case.value, case.expected), ".fea fixes");
}

#[test]
fn a_wrong_unit_reads_the_same_in_every_language() {
    for case in LENGTH_CASES {
        check_part(case);
        check_layout(case);
        check_fea("target_size", "target size", case, "fea.");
    }
}

#[test]
fn a_length_for_a_force_per_area_reads_the_same_in_part_and_fea() {
    let case = Case {
        value: "200 mm",
        expected: "Pa",
    };
    check_fea("youngs_modulus", "youngs modulus", &case, "fea.");
    // A .part property of kind pressure takes the same message.
    let path = "/p/plate.part";
    let text = "runmat 0.7\nstd 1\n\npart plate {\n    input load : pressure = 200 mm\n}\n";
    let uri = Url::parse(&format!("file://{path}")).unwrap();
    let analysis = design(path, text, &DesignFiles::new());
    let message = expected_message("load", &case);
    let (underlined, code, fixes) = report(
        &uri,
        text,
        &analysis.diagnostics(),
        |range| analysis.code_actions(range),
        &message,
    );
    assert_eq!(underlined, case.value, ".part input span");
    assert_eq!(code, "cad.value.unit");
    assert_eq!(fixes, unit_fixes(case.value, case.expected));
}

#[test]
fn a_missing_unit_reads_the_same_in_part_and_fea() {
    // A board length's bare number is millimetres by design, so only .part and .fea require one.
    let case = Case {
        value: "10",
        expected: "m",
    };
    check_part(&case);
    check_fea("target_size", "target size", &case, "fea.");
}
