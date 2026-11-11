//! Parts with no footprint (`package = none`): a breaker and a receptacle wired in the field
//! compile on a schematic-only circuit, pass ERC, and print back as `package = none`.

use runmat_circuit::runtime::{check, CompileOptions, Files};
use runmat_circuit::syntax::{parse_circuit, print_circuit};

const BRANCH: &str = r#"runmat 0.7
std 1

symbol breaker = {
    body = [line([-1.27, 0], [1.27, 1.27])],
    pins = [
        { name = LINE, at = [-5.08, 0], dir = right },
        { name = LOAD, at = [5.08, 0], dir = left }
    ]
}

part "BKR-1P-20A" = {
    manufacturer = "Sample Catalog",
    description = "Breaker, 1 pole, 20 A",
    category = breaker,
    package = none,
    symbol = breaker,
    pins = [
        { name = LINE, type = passive },
        { name = LOAD, type = passive }
    ]
}

part "RCPT-20A" = {
    manufacturer = "Sample Catalog",
    description = "Duplex receptacle, 20 A, 125 V",
    category = receptacle,
    package = none,
    pins = [
        { name = HOT, type = passive },
        { name = NEU, type = passive },
        { name = GND, type = passive }
    ]
}

circuit branch {
    root = main
    sheet main {
        component CB1 = { part = "BKR-1P-20A" }
        component X1 = { part = "RCPT-20A" }
        net L1 = { power_flag = true }
        net N = { global = true, power_flag = true }
        net EG = { global = true, power_flag = true }
        connect = [net.L1, CB1.LINE]
        connect = [net.C1, CB1.LOAD, X1.HOT]
        connect = [net.N, X1.NEU]
        connect = [net.EG, X1.GND]
    }
}
"#;

#[test]
fn a_field_wired_part_compiles_on_a_schematic_only_circuit_and_passes_erc() {
    let path = "/p/electrical/branch.circuit";
    let out = check(path, BRANCH, &Files::new(), &CompileOptions::default());
    let errors: Vec<String> = out
        .diagnostics
        .iter()
        .filter(|d| format!("{:?}", d.severity).contains("Error"))
        .map(|d| d.message.clone())
        .collect();
    assert_eq!(errors, Vec::<String>::new());
    assert!(out.index.is_some(), "{:?}", out.diagnostics);
}

#[test]
fn package_none_prints_back_and_a_missing_package_says_how_to_write_it() {
    let parsed = parse_circuit("/p/b.circuit", BRANCH);
    let design = parsed
        .design
        .unwrap_or_else(|| panic!("{:?}", parsed.diagnostics));
    assert_eq!(design.parts["RCPT-20A"].package, "none");
    let printed = print_circuit(&design).unwrap();
    assert!(printed.contains("package = none,"), "{printed}");
    let again = parse_circuit("/p/b.circuit", &printed);
    assert_eq!(
        again
            .design
            .unwrap_or_else(|| panic!("{:?}", again.diagnostics))
            .parts["BKR-1P-20A"]
            .package,
        "none"
    );
    let out = check(
        "/p/b.circuit",
        &BRANCH.replace("    package = none,\n    symbol", "    symbol"),
        &Files::new(),
        &CompileOptions::default(),
    );
    let m: Vec<String> = out.diagnostics.iter().map(|d| d.message.clone()).collect();
    assert!(
        m.iter()
            .any(|m| m.contains("`package = none` for a part wired in the field")),
        "{m:?}"
    );
}
