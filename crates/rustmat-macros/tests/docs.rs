use rustmat_builtins::builtins;
use rustmat_macros::matlab_fn;

#[matlab_fn(name = "foo")]
/// line1
/// line2
fn foo() {}

#[matlab_fn(name = "bar")]
/// single line doc
fn bar() {}

#[test]
fn docs_captured() {
    let mut found_foo = false;
    let mut found_bar = false;
    for b in builtins() {
        if b.name == "foo" {
            assert_eq!(b.doc, " line1\n line2");
            found_foo = true;
        }
        if b.name == "bar" {
            assert_eq!(b.doc, " single line doc");
            found_bar = true;
        }
    }
    assert!(found_foo && found_bar);
}
