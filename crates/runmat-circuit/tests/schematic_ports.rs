//! A sheet's own ports on its page: a port the sheet declares is drawn as a port label on its
//! net, on the root sheet too, so a one-sheet filter shows where its input and output are.

use runmat_circuit::runtime::{self, CompileOptions, Files};

const PATH: &str = "/rc/lowpass.circuit";

const CIRCUIT: &str = r#"runmat 0.7
std 1

circuit lowpass {
    root = main
    sheet main {
        port IN : in
        port OUT : out
        component R1 = { part = "R0402-10K", value = 9.95 kohm }
        component C1 = { part = "C0402-100N" }
        net GND = { global = true, power_flag = true }
        connect = [port.IN, net.IN, "R1.1"]
        connect = [port.OUT, net.OUT, "R1.2", "C1.1"]
        connect = [net.GND, "C1.2"]
    }
}
"#;

#[test]
fn a_root_sheets_ports_are_drawn_as_port_labels_and_the_filter_reads_left_to_right() {
    let files: Files = [(PATH.to_owned(), CIRCUIT.to_owned())]
        .into_iter()
        .collect();
    let out = runtime::compile(PATH, CIRCUIT, &files, &CompileOptions::default());
    let index = out.index.expect("compiles");
    // IN's one pin reaches outside through the root port, GND's through its power flag: neither
    // is a lone pin.
    let lone: Vec<_> = index
        .erc
        .iter()
        .filter(|d| d.code == "single-pin-net")
        .collect();
    assert!(lone.is_empty(), "{lone:?}");
    assert_eq!(index.schematic.sheets.len(), 1);
    let page = &index.schematic.sheets[0];
    let label = |text: &str| {
        page.labels
            .iter()
            .find(|l| l.text == text)
            .unwrap_or_else(|| panic!("{text} labelled: {:?}", page.labels))
    };
    let (input, output, ground) = (label("IN"), label("OUT"), label("GND"));
    assert_eq!(
        (input.kind.as_str(), output.kind.as_str()),
        ("hier", "hier")
    );
    assert_eq!(ground.kind, "power");
    let at = |d: &str| {
        let s = page
            .symbols
            .iter()
            .find(|s| s.designator == d)
            .unwrap_or_else(|| panic!("{d} drawn"));
        (s.x_mm, s.y_mm)
    };
    let (r1, c1) = (at("R1"), at("C1"));
    // Input on the left, the series resistor, the shunt capacitor, the output on the right.
    assert!(
        input.x_mm < r1.0,
        "IN at {} left of R1 at {}",
        input.x_mm,
        r1.0
    );
    assert!(r1.0 < c1.0, "R1 at {} left of C1 at {}", r1.0, c1.0);
    let layout = || {
        let symbols: Vec<_> = page
            .symbols
            .iter()
            .map(|s| (&s.designator, s.x_mm, s.y_mm, s.rotation))
            .collect();
        let labels: Vec<_> = page
            .labels
            .iter()
            .map(|l| (&l.text, l.x_mm, l.y_mm, &l.side))
            .collect();
        format!(
            "symbols {symbols:?} labels {labels:?} wires {:?}",
            page.wires
        )
    };
    assert!(
        output.x_mm > c1.0,
        "OUT at {} right of C1 at {}: {}",
        output.x_mm,
        c1.0,
        layout()
    );
    // The capacitor goes down to ground (y grows up the page).
    assert!(
        ground.y_mm < c1.1,
        "GND at {} below C1 at {}: {}",
        ground.y_mm,
        c1.1,
        layout()
    );
    // Each label stands clear of the wires: no wire runs on past a label's connection point.
    for l in [input, output] {
        let beyond = page.wires.iter().flat_map(|w| w.points.iter()).any(|p| {
            (p[1] - l.y_mm).abs() < 1e-6
                && if l.side == "right" {
                    p[0] > l.x_mm + 1e-6
                } else {
                    p[0] < l.x_mm - 1e-6
                }
        });
        assert!(!beyond, "{} has a wire beyond it: {}", l.text, layout());
    }
}
