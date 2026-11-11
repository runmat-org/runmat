use super::*;
use crate::syntax::{parse_assembly_in, parse_drawing_in, print_assembly};

const SEAT: &str = "runmat 0.7\nstd 1\n\npart seat_frame {\n}\n";

fn part(name: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart {name} {{\n}}\n")
}

fn files() -> Vec<(String, String)> {
    vec![
        ("/p/runmat.toml".into(), String::new()),
        ("/p/skeleton.part".into(), part("skeleton")),
        ("/p/cabin/seat_frame.part".into(), SEAT.into()),
        ("/p/cabin/galley.part".into(), part("galley")),
        (
            "/p/cabin/bulk.m".into(),
            "function x = bulk()\nx = 1;\nend\n".into(),
        ),
        ("/p/+hardware/rivet.part".into(), part("rivet")),
        ("/p/+hardware/+fasteners/nut.part".into(), part("nut")),
    ]
}

fn scope_with(extra: &[(&str, &str)]) -> ModuleScope {
    let mut all = files();
    all.extend(extra.iter().map(|(p, t)| (p.to_string(), t.to_string())));
    ModuleScope::from_files(all.iter().map(|(p, t)| (p.as_str(), t.as_str())))
}

const CABIN: &str = r#"runmat 0.7
std 1

use "../+hardware" as hw

assembly cabin {
    use part seat_frame as seat
    use part galley as galley
    use part rivet from hardware as rivet
    use part nut from hardware.fasteners as nut
    use part skeleton from ".." as skeleton
    use hw.part.rivet as rivet2
}
"#;

#[test]
fn symbols_resolve_through_folders_aliases_and_packages() {
    let scope = scope_with(&[("/p/cabin/cabin.assembly", CABIN)]);
    let parsed = parse_assembly_in("/p/cabin/cabin.assembly", CABIN, &scope);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let doc = parsed.document.unwrap();
    let files: Vec<&str> = doc
        .occurrences
        .iter()
        .map(|o| o.source.definition.file.as_str())
        .collect();
    assert_eq!(
        files,
        [
            "/p/cabin/seat_frame.part",
            "/p/cabin/galley.part",
            "/p/+hardware/rivet.part",
            "/p/+hardware/+fasteners/nut.part",
            "/p/skeleton.part",
            "/p/+hardware/rivet.part",
        ]
    );
    // `.m` stems are in the folder's scope too (report 12 §3.2), for value calls.
    assert!(scope.scope.folders["/p/cabin"].functions.contains("bulk"));
}

#[test]
fn printing_writes_the_module_form_and_reads_back() {
    let scope = scope_with(&[("/p/cabin/cabin.assembly", CABIN)]);
    let doc = parse_assembly_in("/p/cabin/cabin.assembly", CABIN, &scope)
        .document
        .unwrap();
    let text = print_assembly(&doc, &[]).unwrap();
    assert!(text.contains("use part seat_frame as seat"), "{text}");
    assert!(
        text.contains("use part rivet from hardware as rivet"),
        "{text}"
    );
    assert!(
        text.contains("use part nut from hardware.fasteners as nut"),
        "{text}"
    );
    assert!(
        text.contains("use part skeleton from \"..\" as skeleton"),
        "{text}"
    );
    assert!(!text.contains(".part\""), "no file paths: {text}");
    let again = parse_assembly_in("/p/cabin/cabin.assembly", &text, &scope);
    assert!(
        again.diagnostics.is_empty(),
        "{:?}\n{text}",
        again.diagnostics
    );
    assert_eq!(again.document.unwrap().occurrences, doc.occurrences);
}

#[test]
fn unknown_and_duplicate_designs_are_errors_naming_the_files() {
    let text = "runmat 0.7\nstd 1\n\nassembly cabin {\n    use part seet_frame as seat\n}\n";
    let scope = scope_with(&[("/p/cabin/cabin.assembly", text)]);
    let parsed = parse_assembly_in("/p/cabin/cabin.assembly", text, &scope);
    let m = &parsed.diagnostics[0].message;
    assert!(
        m.contains("no `part seet_frame` in this folder") && m.contains("seat_frame"),
        "{m}"
    );
    assert!(
        parsed.diagnostics[0].span.start > 0,
        "the error sits on the occurrence"
    );

    let text = "runmat 0.7\nstd 1\n\nassembly cabin {\n    use part galley as g\n}\n";
    let other =
        "runmat 0.7\nstd 1\n\npart galley_b {\n    output galley = { body = feature.x.body }\n}\n";
    let scope = scope_with(&[
        ("/p/cabin/cabin.assembly", text),
        ("/p/cabin/galley_b.part", other),
    ]);
    let parsed = parse_assembly_in("/p/cabin/cabin.assembly", text, &scope);
    let m = &parsed.diagnostics[0].message;
    assert!(
        m.contains("declared twice")
            && m.contains("/p/cabin/galley.part")
            && m.contains("/p/cabin/galley_b.part"),
        "{m}"
    );
    assert_eq!(scope.problems().len(), 1);
}

#[test]
fn use_cycles_between_folders_are_errors() {
    let a = "runmat 0.7\nstd 1\n\nassembly a {\n    use part b from \"../b\" as b\n}\n";
    let b = "runmat 0.7\nstd 1\n\nassembly b {\n    use assembly a from \"../a\" as a\n}\n";
    let scope = scope_with(&[
        ("/p/a/a.assembly", a),
        ("/p/b/b.assembly", b),
        ("/p/b/b.part", &part("b")),
    ]);
    let cycles: Vec<String> = scope.problems().iter().map(problem_text).collect();
    assert_eq!(
        cycles,
        ["folders use each other in a cycle: /p/a → /p/b → /p/a"]
    );
}

#[test]
fn interface_frames_and_drawing_models_name_designs() {
    let text = r#"runmat 0.7
std 1

assembly top {
    use part skeleton as skeleton
    use assembly cabin from "./cabin" as cabin {
        placement = { placement = attach, own = { anchor = origin }, to = { frame = interface, part = skeleton, name = cabin } }
    }
}
"#;
    let cabin = "runmat 0.7\nstd 1\n\nassembly cabin {\n}\n";
    let scope = scope_with(&[
        ("/p/top.assembly", text),
        ("/p/cabin/cabin.assembly", cabin),
    ]);
    let parsed = parse_assembly_in("/p/top.assembly", text, &scope);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let doc = parsed.document.unwrap();
    let json = serde_json::to_string(&doc.occurrences[1].placement).unwrap();
    assert!(json.contains("\"file\":\"/p/skeleton.part\""), "{json}");
    let printed = print_assembly(&doc, &[]).unwrap();
    assert!(
        printed.contains("frame = interface, part = skeleton"),
        "{printed}"
    );
    assert!(
        printed.contains("use assembly cabin from \"./cabin\" as cabin"),
        "{printed}"
    );

    let drawing = r#"runmat 0.7
std 1

drawing seat {
    format a3 = { size = { size = iso, name = A3 }, title_block = [] }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = seat_frame, from = "./cabin" }, at = [150, 165], display = hidden_lines)
}
"#;
    let scope = scope_with(&[("/p/seat.drawing", drawing)]);
    let parsed = parse_drawing_in("/p/seat.drawing", drawing, &scope);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    assert_eq!(
        parsed.document.unwrap().views[0].model.file.as_str(),
        "/p/cabin/seat_frame.part"
    );
}

#[test]
fn dependencies_list_the_folders_a_file_can_name() {
    let deps = dependencies("/p/cabin/cabin.assembly", CABIN);
    for f in [
        "/p/cabin/",
        "/p/",
        "/p/+hardware/",
        "/p/cabin/+hardware/",
        "/p/+hardware/+fasteners/",
    ] {
        assert!(deps.contains(&f.to_string()), "{f} in {deps:?}");
    }
}

#[test]
fn module_text_is_relative_or_a_package() {
    assert_eq!(module_text("/p/a.assembly", "/p/b.part"), None);
    assert_eq!(
        module_text("/p/a.assembly", "/p/cabin/b.part").as_deref(),
        Some("./cabin")
    );
    assert_eq!(
        module_text("/p/wing/a.assembly", "/p/cabin/b.part").as_deref(),
        Some("../cabin")
    );
    assert_eq!(
        module_text("/p/wing/a.assembly", "/p/b.part").as_deref(),
        Some("..")
    );
    assert_eq!(
        module_text("/p/wing/a.assembly", "/p/+hw/b.part").as_deref(),
        Some("hw")
    );
    assert_eq!(
        module_text("a.assembly", "+hw/+x/b.part").as_deref(),
        Some("hw.x")
    );
}

/// A package is searched for in the referring folder, then each folder above it, as MATLAB's
/// path is searched in order (report 12 §3.2: the first on the path wins). A `+hardware` in
/// `cabin/` shadows the project's for designs in `cabin/`; other folders still reach the
/// project's.
#[test]
fn a_nearer_package_shadows_an_outer_one() {
    let scope = scope_with(&[
        ("/p/cabin/+hardware/rivet.part", &part("rivet")),
        ("/p/wing/rib.part", &part("rib")),
    ]);
    assert_eq!(
        scope.locate("/p/cabin/cabin.assembly", "part", "rivet", Some("hardware")),
        Ok("/p/cabin/+hardware/rivet.part".to_string())
    );
    assert_eq!(
        scope.locate("/p/wing/wing.assembly", "part", "rivet", Some("hardware")),
        Ok("/p/+hardware/rivet.part".to_string())
    );
    // The two `rivet`s are in different folders: no duplicate.
    assert!(scope.problems().is_empty(), "{:?}", scope.problems());
}

/// An override's replacement names its design the same way an occurrence does: by name in this
/// folder, or `from` a folder or package. It binds to the file and prints back by name.
#[test]
fn a_replacement_names_its_design_by_symbol() {
    let text = "runmat 0.7\nstd 1\n\nassembly cabin {\n    use assembly row as row {\n        override seat {\n            replace = part(galley)\n        }\n        override bolt {\n            replace = part(rivet, from = hardware, revision = \"B\")\n        }\n    }\n}\n";
    let row = "runmat 0.7\nstd 1\n\nassembly row {\n    use part seat_frame as seat\n    use part seat_frame as bolt\n}\n";
    let scope = scope_with(&[
        ("/p/cabin/cabin.assembly", text),
        ("/p/cabin/row.assembly", row),
    ]);
    let parsed = parse_assembly_in("/p/cabin/cabin.assembly", text, &scope);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let doc = parsed.document.unwrap();
    let replaced: Vec<String> = doc.occurrences[0]
        .overrides
        .iter()
        .filter_map(|o| match &o.set {
            crate::assembly::OverrideKind::Replace { source } => {
                Some(source.definition.file.0.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        replaced,
        ["/p/cabin/galley.part", "/p/+hardware/rivet.part"]
    );
    let printed = print_assembly(&doc, &[]).unwrap();
    assert!(printed.contains("replace = part(galley)\n"), "{printed}");
    assert!(
        printed.contains("replace = part(rivet, from = hardware, revision = \"B\")\n"),
        "{printed}"
    );
    let again = parse_assembly_in("/p/cabin/cabin.assembly", &printed, &scope);
    assert_eq!(
        again.document.unwrap().occurrences,
        doc.occurrences,
        "{printed}"
    );
}

/// The sandbox puts a launched project at the workspace root (`/cabin/cabin.assembly`,
/// `/skeleton.part`): the root is the folder `/`, and the dependencies a host lists are
/// absolute.
#[test]
fn a_project_at_the_workspace_root_resolves_and_lists_absolute_folders() {
    let at_root = |p: &str| p.strip_prefix("/p").unwrap().to_string();
    let mut all: Vec<(String, String)> =
        files().into_iter().map(|(p, t)| (at_root(&p), t)).collect();
    all.push(("/cabin/cabin.assembly".into(), CABIN.into()));
    let scope = ModuleScope::from_files(all.iter().map(|(p, t)| (p.as_str(), t.as_str())));
    assert_eq!(folder_of("/skeleton.part"), "/");
    assert_eq!(scope.root.as_deref(), Some("/"));
    let parsed = parse_assembly_in("/cabin/cabin.assembly", CABIN, &scope);
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let files: Vec<String> = parsed
        .document
        .unwrap()
        .occurrences
        .iter()
        .map(|o| o.source.definition.file.as_str().to_string())
        .collect();
    assert_eq!(
        files,
        [
            "/cabin/seat_frame.part",
            "/cabin/galley.part",
            "/+hardware/rivet.part",
            "/+hardware/+fasteners/nut.part",
            "/skeleton.part",
            "/+hardware/rivet.part"
        ]
    );
    let top = "runmat 0.7\nstd 1\n\nassembly top {\n    use assembly cabin from \"./cabin\" as cabin\n    use part skeleton as skeleton\n}\n";
    assert_eq!(dependencies("/top.assembly", top), ["/", "/cabin/"]);
    assert_eq!(
        module_text("/top.assembly", "/+hardware/rivet.part").as_deref(),
        Some("hardware")
    );
    assert_eq!(
        module_text("/top.assembly", "/cabin/galley.part").as_deref(),
        Some("./cabin")
    );
}

/// The file form is refused (report 12 §3.5): an occurrence, an interface frame or a drawing
/// model that names a design by its file path is an error that says how to name the design.
/// A STEP file keeps its path.
#[test]
fn a_design_named_by_its_file_path_is_an_error_that_says_to_name_the_design() {
    let bracket = part("bracket");
    let text = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket from \"parts/bracket.part\" as b\n}\n";
    let scope = scope_with(&[
        ("/p/rig.assembly", text),
        ("/p/parts/bracket.part", &bracket),
    ]);
    let parsed = parse_assembly_in("/p/rig.assembly", text, &scope);
    let m = &parsed.diagnostics[0].message;
    assert_eq!(m, "name the design: write `use part bracket as …` with `from \"./parts\"`, not the file \"parts/bracket.part\"");
    assert!(
        parsed.diagnostics[0].span.start > 0,
        "the error sits on the occurrence"
    );

    let drawing = "runmat 0.7\nstd 1\n\ndrawing d {\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n    view front = standard(orientation = front, sheet = \"1\", model = { file = \"bracket.part\" }, at = [150, 165], display = hidden_lines)\n}\n";
    let scope = scope_with(&[("/p/d.drawing", drawing), ("/p/bracket.part", &bracket)]);
    let parsed = parse_drawing_in("/p/d.drawing", drawing, &scope);
    let messages: Vec<&str> = parsed
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        messages.contains(
            &"name the design: write `model = { part = bracket }`, not the file \"bracket.part\""
        ),
        "{messages:?}"
    );

    // The symbol form of the same assembly binds.
    let text =
        "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket from \"./parts\" as b\n}\n";
    let scope = scope_with(&[
        ("/p/rig.assembly", text),
        ("/p/parts/bracket.part", &bracket),
    ]);
    assert!(parse_assembly_in("/p/rig.assembly", text, &scope)
        .diagnostics
        .is_empty());
}
