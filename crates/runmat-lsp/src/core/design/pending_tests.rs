use super::super::{design_dependencies, DesignDocumentAnalysis, DesignFiles, DesignValues};
use super::PENDING_CODE;
use lsp_types::{DiagnosticSeverity, Url};
use runmat_design_syntax::Dialect;

const PART: &str = "runmat 0.7\nstd 1\n\npart battery {\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 8 mm)\n    }\n    feature body = extrude(profile = sketch.s, distance = 5 mm)\n}\n";
const TOP: &str = "runmat 0.7\nstd 1\n\nassembly drone {\n    use part battery from \"./mechanical/body\" as battery\n}\n";
const CIRCUIT: &str = "runmat 0.7\nstd 1\n\nuse sheet buck from \"../power\" as reg\n\ncircuit ctl {\n    sheet main {\n        instance r = { sheet = reg }\n    }\n}\n";

fn analyze(path: &str, text: &str, files: &DesignFiles, pending: &[&str]) -> DesignDocumentAnalysis {
    let uri = Url::parse(&format!("file://{path}")).unwrap();
    let dialect = Dialect::for_path(path).unwrap();
    let pending: Vec<String> = pending.iter().map(|p| p.to_string()).collect();
    DesignDocumentAnalysis::analyze(&uri, dialect, text, files, &DesignValues::new()).with_pending(&pending)
}

fn messages(a: &DesignDocumentAnalysis) -> Vec<String> {
    a.diagnostics().into_iter().map(|d| d.message).collect()
}

#[test]
fn an_assembly_resolves_a_subfolder_import_from_the_supplied_folder_files() {
    // The dependency the host lists is the folder; it supplies the folder's design files.
    assert!(design_dependencies(Dialect::Assembly, "/d/drone.assembly", TOP).contains(&"/d/mechanical/body/".to_string()));
    let files = DesignFiles::from([("/d/mechanical/body/battery.part".to_string(), PART.to_string())]);
    let a = analyze("/d/drone.assembly", TOP, &files, &[]);
    let m = messages(&a);
    assert!(m.iter().all(|m| !m.contains("holds no designs") && !m.contains("no `part battery`")), "{m:?}");
}

#[test]
fn a_folder_still_loading_says_so_on_its_use_line_and_holds_the_compile_problems() {
    let a = analyze("/d/drone.assembly", TOP, &DesignFiles::new(), &["/d/mechanical/body/"]);
    let d = a.diagnostics();
    assert_eq!(d.len(), 1, "{:?}", messages(&a));
    assert_eq!(d[0].severity, Some(DiagnosticSeverity::INFORMATION));
    assert_eq!(d[0].message, "`./mechanical/body` isn't loaded yet; this line is checked again when it arrives");
    assert_eq!(d[0].range.start.line, 4);
    assert!(matches!(&d[0].code, Some(lsp_types::NumberOrString::String(c)) if c == PENDING_CODE));
    // Without the pending list, the same files report the folder missing.
    let missing = messages(&analyze("/d/drone.assembly", TOP, &DesignFiles::new(), &[]));
    assert!(missing.iter().any(|m| m.contains("holds no designs")), "{missing:?}");
}

#[test]
fn a_pending_file_no_use_line_names_marks_the_first_line() {
    let files = DesignFiles::from([("/d/mechanical/body/battery.part".to_string(), PART.to_string())]);
    let a = analyze("/d/drone.assembly", TOP, &files, &["/d/mechanical/+hardware/"]);
    let d = a.diagnostics();
    assert_eq!(d.len(), 1, "{:?}", messages(&a));
    assert_eq!(d[0].range.start.line, 0);
    assert!(d[0].message.starts_with("`/d/mechanical/+hardware/` isn't loaded yet"), "{}", d[0].message);
}

#[test]
fn a_circuit_import_still_loading_says_so_on_its_use_line() {
    let a = analyze("/p/ctl/ctl.circuit", CIRCUIT, &DesignFiles::new(), &["/p/power/"]);
    let m = messages(&a);
    assert!(m.contains(&"`../power` isn't loaded yet; this line is checked again when it arrives".to_string()), "{m:?}");
    assert!(m.iter().all(|m| m.contains("isn't loaded yet")), "{m:?}");
}

#[test]
fn syntax_errors_stay_while_files_load() {
    let broken = TOP.replace("as battery", "as battery (");
    let m = messages(&analyze("/d/drone.assembly", &broken, &DesignFiles::new(), &["/d/mechanical/body/"]));
    assert!(m.iter().any(|m| !m.contains("isn't loaded yet")), "{m:?}");
}

#[test]
fn an_assembly_resolves_a_circuit_imported_from_a_subfolder() {
    let io = |f: &str| std::fs::read_to_string(format!("{}/../runmat-circuit/tests/projects/io/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let top = "runmat 0.7\nstd 1\n\nassembly drone {\n    use circuit io from \"./electrical/io\" as board {\n        placement = { placement = offset, offset = { translation = [25 mm, 0 mm, 10 mm] } }\n    }\n}\n";
    let mut files = DesignFiles::new();
    for f in ["io.circuit", "io.circuit.lock", "io.circuit.layout"] {
        files.insert(format!("/d/electrical/io/{f}"), io(f));
    }
    let listed = design_dependencies(Dialect::Assembly, "/d/drone.assembly", top);
    assert!(listed.contains(&"/d/electrical/io/".to_string()), "{listed:?}");
    let m = messages(&analyze("/d/drone.assembly", top, &files, &[]));
    assert!(m.iter().all(|m| !m.contains("not in the project") && !m.contains("holds no designs") && !m.contains("no `circuit io`")), "{m:?}");
}

#[test]
fn a_note_about_a_whole_declaration_marks_its_first_line() {
    let a = analyze("/d/drone.assembly", TOP, &DesignFiles::new(), &[]);
    let mut b = a.clone();
    b.problems = vec![super::super::Problem {
        span: runmat_types::Span { start: TOP.find("assembly drone").unwrap(), end: TOP.len() - 1 },
        severity: DiagnosticSeverity::INFORMATION,
        code: Some("asm.unlocked".into()),
        message: "1 item(s) of `/d/drone.assembly` have no lock entry".into(),
    }];
    let d = b.diagnostics();
    assert_eq!((d[0].range.start.line, d[0].range.end.line), (3, 3));
    assert_eq!(d[0].range.end.character as usize, "assembly drone {".len());
}
