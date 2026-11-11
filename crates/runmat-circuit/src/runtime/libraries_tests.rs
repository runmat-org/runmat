use super::*;

const HEAD: &str = "runmat 0.7\nstd 1\n\n";

fn lib(uses: &[&str], body: &str) -> String {
    let uses: String = uses.iter().map(|u| format!("use \"{u}\"\n")).collect();
    format!("{HEAD}{uses}\n{body}\n")
}

fn part(mpn: &str, package: &str) -> String {
    format!("part \"{mpn}\" = {{ manufacturer = \"M\", package = {package}, pins = [{{ name = A, pad = \"1\" }}, {{ name = B, pad = \"2\" }}] }}")
}

fn files(entries: &[(&str, String)]) -> Files {
    entries.iter().map(|(p, t)| (p.to_string(), t.clone())).collect()
}

fn design(path: &str, f: &Files) -> Design {
    let parsed = parse_circuit(path, &f[path]);
    parsed.design.unwrap_or_else(|| panic!("{path}: {:?}", parsed.diagnostics))
}

fn resolved(path: &str, f: &Files) -> Result<(Design, Origins), LibraryError> {
    resolve(path, &f[path], &design(path, f), f)
}

#[test]
fn a_library_brings_its_declarations_and_those_it_uses_into_scope() {
    let f = files(&[
        ("/p/lib/base.circuit", lib(&[], &part("R-BASE", "\"R0402\""))),
        ("/p/lib/a.circuit", lib(&["base.circuit"], &part("R-A", "\"R0603\""))),
        // `b` reaches `base` too: the same file, not a second declaration.
        ("/p/lib/b.circuit", lib(&["./base.circuit"], &part("R-B", "\"R0805\""))),
        ("/p/top.circuit", lib(&["lib/a.circuit", "lib/b.circuit"], "")),
    ]);
    let (scope, origins) = resolved("/p/top.circuit", &f).unwrap();
    assert_eq!(scope.parts.keys().collect::<Vec<_>>(), ["R-A", "R-B", "R-BASE"]);
    assert_eq!(origins[&("part".to_string(), "R-BASE".to_string())], "/p/lib/base.circuit");
    assert_eq!(library_dependencies("/p/top.circuit", &design("/p/top.circuit", &f)), ["/p/lib/a.circuit", "/p/lib/b.circuit"]);
}

#[test]
fn a_name_declared_in_two_files_names_both() {
    let f = files(&[
        ("/p/a.circuit", lib(&[], &part("R1", "\"R0402\""))),
        ("/p/b.circuit", lib(&[], &part("R1", "\"R0603\""))),
        ("/p/top.circuit", lib(&["a.circuit", "b.circuit"], "")),
    ]);
    let e = resolved("/p/top.circuit", &f).unwrap_err();
    assert_eq!(e.message, "part `R1` is declared in both /p/a.circuit and /p/b.circuit; rename one or remove one");
    // A file's own declaration and a used one of the same name, too.
    let f = files(&[("/p/a.circuit", lib(&[], &part("R1", "\"R0402\""))), ("/p/top.circuit", lib(&["a.circuit"], &part("R1", "\"R0603\"")))]);
    assert!(resolved("/p/top.circuit", &f).unwrap_err().message.contains("both /p/top.circuit and /p/a.circuit"));
}

#[test]
fn a_cycle_lists_the_chain_and_a_missing_file_names_its_path() {
    let f = files(&[
        ("/p/a.circuit", lib(&["b.circuit"], "")),
        ("/p/b.circuit", lib(&["a.circuit"], "")),
        ("/p/top.circuit", lib(&["a.circuit"], "")),
    ]);
    let e = resolved("/p/top.circuit", &f).unwrap_err();
    assert_eq!(e.file, "/p/b.circuit");
    assert_eq!(e.message, "these libraries use each other in a cycle: /p/a.circuit → /p/b.circuit → /p/a.circuit");
    let text = &f["/p/b.circuit"];
    assert_eq!(&text[e.start..e.end], "\"a.circuit\"");

    let f = files(&[("/p/top.circuit", lib(&["lib/gone.circuit"], ""))]);
    let e = resolved("/p/top.circuit", &f).unwrap_err();
    assert_eq!(e.message, "the library \"lib/gone.circuit\" can't be read: no file /p/lib/gone.circuit");

    let f = files(&[
        ("/p/board.circuit", format!("{HEAD}circuit board {{\n    sheet main {{\n    }}\n}}\n")),
        ("/p/top.circuit", lib(&["board.circuit"], "")),
    ]);
    assert!(resolved("/p/top.circuit", &f).unwrap_err().message.contains("is a design"));
}

#[test]
fn a_library_part_names_a_footprint_from_another_library() {
    let lcc = "footprint lcc4 = {\n    pads = [\n        { number = \"1\", shape = rect, size = [1, 1], at = [-1, 0], layers = smd },\n        { number = \"2\", shape = rect, size = [1, 1], at = [1, 0], layers = smd }\n    ],\n    courtyard = rect([-2, -1], [2, 1])\n}";
    let f = files(&[
        ("/p/footprints/lcc4.circuit", lib(&[], lcc)),
        ("/p/parts.circuit", lib(&["footprints/lcc4.circuit"], &part("MOD-1", "lcc4"))),
    ]);
    let own = design("/p/parts.circuit", &f);
    let (scope, _) = resolved("/p/parts.circuit", &f).unwrap();
    assert_eq!(library_findings("/p/parts.circuit", &own, &scope, &f), Vec::<String>::new());
    // Without the footprint library, the part names what it can't find.
    let f = files(&[("/p/parts.circuit", lib(&[], &part("MOD-1", "lcc4")))]);
    let own = design("/p/parts.circuit", &f);
    assert_eq!(
        library_findings("/p/parts.circuit", &own, &own, &f),
        ["part `MOD-1`: no footprint `lcc4` is declared here or in a library this file uses"]
    );
}
