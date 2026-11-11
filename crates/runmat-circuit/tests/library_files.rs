//! Library `.circuit` files: the `footprint`, `symbol` and `part` declarations read and print
//! back exactly, a generated package prints as a footprint declaration, and a design compiles
//! against libraries it reaches through `use`, gathered the way the browser host gathers them.

use runmat_circuit::library::footprint::from_string;
use runmat_circuit::runtime::{check, compile, dependencies, CompileOptions, Files};
use runmat_circuit::syntax::{footprint_decl, parse_circuit, print_circuit};

/// The spec's GPS module footprint, a symbol and two parts, as the printer writes them.
const LIBRARY: &str = r#"runmat 0.7
std 1

footprint gps_lcc6 = {
    description = "6-pad LCC GPS module, 10.1 x 9.7 mm",
    pads = [
        { number = "1", shape = rect, size = [0.8, 1.5], at = [-4.05, -2.54], layers = smd },
        { number = "2", shape = rect, size = [0.8, 1.5], at = [-4.05, 0], layers = smd },
        { number = "3", shape = rect, size = [0.8, 1.5], at = [-4.05, 2.54], layers = smd },
        { number = "4", shape = roundrect, size = [0.8, 1.5], ratio = 0.1, at = [4.05, 2.54], layers = smd, paste = false },
        { number = "5", shape = oval, size = [0.8, 1.5], at = [4.05, 0], rotation = 90, layers = smd, mask_expand = 0.05 },
        { number = "6", shape = circle, size = [1.6, 1.6], at = [0, 0], layers = through, drill = 1 },
        { number = "", shape = circle, size = [1, 1], at = [0, 3], layers = hole, drill = 1 }
    ],
    courtyard = rect([-5.3, -5.1], [5.3, 5.1]),
    silk = [line([-5.05, -4.85], [5.05, -4.85]), circle([-5.6, -4.6], 0.15, fill = true)],
    fab = [rect([-5.05, -4.85], [5.05, 4.85]), text([0, 0], "${REFERENCE}")],
    body = { height = 2.4, model = "models/gps_module.step" }
}

symbol buffer = {
    body = [polygon([[-2.54, -2.54], [-2.54, 2.54], [2.54, 0]])],
    pins = [
        { name = IN, at = [-5.08, 0], dir = right },
        { name = OUT, at = [5.08, 0], dir = left },
        { name = VCC, at = [0, 5.08], dir = down, length = 3.81 }
    ]
}

part "ADC-QFN24" = {
    manufacturer = "Sample Catalog",
    package = "QFN24_4mm_P0.5_EP2.6",
    pins = [
        { name = AIN, pads = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], type = input },
        { name = GND, pads = ["13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"], type = power_in }
    ]
}

part "GPS-M6" = {
    manufacturer = "Sample Catalog",
    package = gps_lcc6,
    symbol = buffer,
    body = "models/gps.part",
    pins = [
        { name = IN, pad = "1", type = input },
        { name = OUT, pad = "2", type = output },
        { name = VCC, pads = ["3", "6"], type = power_in }
    ],
    nc = ["4", "5"]
}
"#;

#[test]
fn a_library_reads_and_prints_back_exactly() {
    let parsed = parse_circuit("/p/lib.circuit", LIBRARY);
    let lib = parsed.design.unwrap_or_else(|| panic!("{:?}", parsed.diagnostics));
    assert!(lib.library);
    assert_eq!(lib.footprints["gps_lcc6"].pads.len(), 7);
    assert_eq!(lib.parts["GPS-M6"].footprint.as_deref(), Some("gps_lcc6"));
    assert_eq!(lib.symbols["buffer"].pins[2].angle, 270.0);
    let printed = print_circuit(&lib).unwrap();
    assert_eq!(printed, LIBRARY);
}

#[test]
fn a_generated_package_prints_as_a_footprint_declaration() {
    for s in ["QFN24_4mm_P0.5_EP2.6", "SOIC8", "R0402@A", "PinHeader2x05", "MountingHole_3.2mm_Pad6", "BGA256_16x16_P1", "SOT223", "DFN8_3x3mm_P0.65_EP1.6x2.4", "SMDConn4_P1_MP1.2x1.8"] {
        let mut generated = from_string(s).unwrap();
        let text = format!("runmat 0.7\nstd 1\n\n{}\n", footprint_decl::statement("detached", &generated).unwrap());
        let parsed = parse_circuit("/p/f.circuit", &text);
        let lib = parsed.design.unwrap_or_else(|| panic!("{s}: {:?}\n{text}", parsed.diagnostics));
        let read = &lib.footprints["detached"];
        generated.name = read.name.clone();
        generated.uuid = read.uuid;
        assert_eq!(read, &generated, "{s}\n{text}");
    }
}

#[test]
fn declaration_errors_name_the_footprint_and_pad() {
    let bad = |body: &str| {
        let text = format!("runmat 0.7\nstd 1\n\nfootprint f = {{\n    pads = [{body}],\n    courtyard = [line([0, 0], [1, 0]), line([1, 0], [1, 1])]\n}}\n");
        parse_circuit("/p/f.circuit", &text).diagnostics.into_iter().map(|d| d.message).collect::<Vec<_>>()
    };
    let m = bad("{ number = \"1\", shape = rect, size = [1, 1], at = [0, 0], layers = smd }");
    assert!(m.iter().any(|m| m.starts_with("footprint `f`: the courtyard is not closed; its outline ends at")), "{m:?}");
    let m = bad("{ number = \"1\", shape = rect, size = [1, 1], layers = through }");
    assert!(m.iter().any(|m| m == "footprint `f`: pad `1` goes through the board and needs a `drill`"), "{m:?}");
    let m = bad("{ number = \"1\", shape = star, size = [1, 1] }");
    assert!(m.iter().any(|m| m.contains("pad `1` has shape `star`")), "{m:?}");
}

/// A project in the standard layout: a footprint library, a parts library that uses it, and a
/// board that uses the parts library.
fn project() -> Files {
    let footprint = LIBRARY.split("symbol buffer").next().unwrap().to_string();
    let parts = format!(
        "runmat 0.7\nstd 1\n\nuse \"footprints/gps_lcc6.circuit\"\n\nsymbol buffer{}",
        LIBRARY.split("symbol buffer").nth(1).unwrap()
    );
    let board = r#"runmat 0.7
std 1

use "../library/parts.circuit"

circuit sensor_board {
    sheet main {
        component U1 = { part = "ADC-QFN24" }
        component U2 = { part = "GPS-M6" }
        net GND = { global = true, power_flag = true }
        net VCC = { global = true, power_flag = true }
        connect = [net.GND, U1.GND]
        connect = [net.VCC, U2.VCC]
        connect = [net.SIG, U2.OUT, U1.AIN]
        no_erc = [U2.IN]
    }
}
"#;
    [
        ("/p/electrical/library/footprints/gps_lcc6.circuit", footprint),
        ("/p/electrical/library/parts.circuit", parts),
        ("/p/electrical/sensor_board/sensor_board.circuit", board.to_string()),
    ]
    .into_iter()
    .map(|(p, t)| (p.to_string(), t))
    .collect()
}

/// The browser host's gather: the compiler names a file's dependencies, the host reads each
/// that exists and asks again for each file it read.
fn gather(path: &str, disk: &Files) -> Files {
    let mut files = Files::new();
    let mut queue = dependencies(path, &disk[path]);
    while let Some(next) = queue.pop() {
        if files.contains_key(&next) || next == path {
            continue;
        }
        if let Some(text) = disk.get(&next) {
            files.insert(next.clone(), text.clone());
            queue.extend(dependencies(&next, text));
        }
    }
    files
}

#[test]
fn a_board_compiles_against_the_libraries_its_use_lines_reach() {
    let disk = project();
    let board = "/p/electrical/sensor_board/sensor_board.circuit";
    let files = gather(board, &disk);
    assert!(files.contains_key("/p/electrical/library/parts.circuit"));
    assert!(files.contains_key("/p/electrical/library/footprints/gps_lcc6.circuit"), "a library's own libraries are gathered");
    let out = compile(board, &disk[board], &files, &CompileOptions::default());
    assert!(out.index.is_some(), "{:?}", out.diagnostics);
    // Each library compiles on its own, clean.
    for lib in ["/p/electrical/library/parts.circuit", "/p/electrical/library/footprints/gps_lcc6.circuit"] {
        let out = check(lib, &disk[lib], &gather(lib, &disk), &CompileOptions::default());
        assert_eq!(out.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>(), Vec::<String>::new(), "{lib}");
    }
}

#[test]
fn a_library_that_isnt_loaded_or_is_wrong_says_so_on_the_use_line() {
    let disk = project();
    let board = "/p/electrical/sensor_board/sensor_board.circuit";
    let out = check(board, &disk[board], &Files::new(), &CompileOptions::default());
    let d = &out.diagnostics[0];
    assert_eq!(d.message, "the library \"../library/parts.circuit\" can't be read: no file /p/electrical/library/parts.circuit");
    // A pad left without a pin is a finding on the part, in the library.
    let parts = "/p/electrical/library/parts.circuit";
    let text = disk[parts].replace(",\n    nc = [\"4\", \"5\"]", "");
    let out = check(parts, &text, &gather(parts, &disk), &CompileOptions::default());
    let m: Vec<String> = out.diagnostics.iter().map(|d| d.message.clone()).collect();
    assert!(m.iter().any(|m| m.starts_with("part `GPS-M6`: its footprint has 6 pads and its pins land on 4; pads `4`, `5` have no pin")), "{m:?}\n{text}");
}
