//! No-connect crosses on a schematic: a pin on no net gets a cross at its tip only when the
//! design marks it no-connect (a `no_erc` mark or a not-connected pin type). Any other pin on
//! no net is drawn open, and ERC warns about it so a forgotten wire stays visible.

use runmat_circuit::runtime::{self, CompileOptions, Files};

const PATH: &str = "/p/reg.circuit";

const CIRCUIT: &str = r#"runmat 0.7
std 1

circuit reg {
    root = main

    sheet main {
        component U1 = { part = "LDO-3V3-SOT23" }
        component C1 = { part = "C0603-1U-50V" }
        connect = [U1.VIN, "C1.1"]
        no_erc = [U1.GND]
    }
}
"#;

#[test]
fn only_marked_pins_get_a_cross_and_open_pins_are_left_to_erc() {
    let files: Files = [(PATH.to_owned(), CIRCUIT.to_owned())].into_iter().collect();
    let out = runtime::compile(PATH, CIRCUIT, &files, &CompileOptions::default());
    let index = out.index.expect("compiles");
    let page = &index.schematic.sheets[0];
    let u1 = page.symbols.iter().find(|s| s.designator == "U1").expect("U1 drawn");
    let crossed = |name: &str| {
        let p = u1.pins.iter().find(|p| p.name == name).expect(name);
        (p.no_connect, page.no_connects.contains(&[p.x_mm, p.y_mm]))
    };
    // NC is a not-connected pin type and GND carries a no_erc mark: both are crossed.
    assert_eq!(crossed("NC"), (true, true));
    assert_eq!(crossed("GND"), (true, true));
    // EN and VOUT are on no net and unmarked: drawn open, no cross.
    assert_eq!(crossed("EN"), (false, false));
    assert_eq!(crossed("VOUT"), (false, false));
    // A wired pin is never crossed.
    assert_eq!(crossed("VIN"), (false, false));
    assert_eq!(page.no_connects.len(), 2, "{:?}", page.no_connects);

    // ERC names each open pin and says how to accept it; marked pins stay quiet.
    let open: Vec<_> = index.erc.iter().filter(|d| d.code == "pin-unconnected").collect();
    let about = |pin: &str| open.iter().find(|d| d.items.iter().any(|i| i.starts_with(pin)));
    for pin in ["U1.EN", "U1.VOUT"] {
        let d = about(pin).unwrap_or_else(|| panic!("{pin} warned: {open:?}"));
        assert!(
            d.message.contains(&format!("pin {pin}")) && d.message.ends_with("is unconnected; mark it no-connect if intended"),
            "{}",
            d.message
        );
    }
    assert!(about("U1.NC").is_none() && about("U1.GND").is_none(), "{open:?}");
}
