use runmat_hir::lower;
use runmat_ignition::execute;
use runmat_parser::parse;
use std::convert::TryInto;

#[test]
fn logical_operators_and_short_circuit() {
    let ast =
        parse("a = 0 && (1/0); b = 1 || (1/0); c = 0 & 5; d = 0 | 5; e = ~0; f = ~5;").unwrap();
    let hir = lower(&ast).unwrap();
    let vars = execute(&hir).unwrap();
    let a: f64 = (&vars[0]).try_into().unwrap();
    let b: f64 = (&vars[1]).try_into().unwrap();
    let c: f64 = (&vars[2]).try_into().unwrap();
    let d: f64 = (&vars[3]).try_into().unwrap();
    let e: f64 = (&vars[4]).try_into().unwrap();
    let f: f64 = (&vars[5]).try_into().unwrap();
    assert_eq!(a, 0.0);
    assert_eq!(b, 1.0);
    assert_eq!(c, 0.0);
    assert_eq!(d, 1.0);
    assert_eq!(e, 1.0);
    assert_eq!(f, 0.0);
}
