use runmat_parser::{Expr, Stmt};

mod parse;
use parse::parse;

#[test]
fn command_form_dynamic_member_with_ellipsis_errors() {
    // Command-form should not accept dynamic member s.(expr); with ellipsis continuation
    let res = parse("foo s.(1) ...\n 'arg'");
    assert!(res.is_err());
}

#[test]
fn mixed_dynamic_member_with_command_tokens_errors() {
    // Mixing command-form tokens around a dynamic member should not be accepted as command-form
    let res = parse("bar before s.(x) after");
    assert!(res.is_err());
}

#[test]
fn imports_and_metaclass_static_same_line() {
    let program =
        parse("import pkg.*; __register_test_classes(); v = ?Point.staticValue;").unwrap();
    assert_eq!(program.body.len(), 3);
    match &program.body[2] {
        Stmt::Assign(name, rhs, _, _) => {
            assert_eq!(name, "v");
            match rhs {
                Expr::Member(obj, field, _) => {
                    assert_eq!(field, "staticValue");
                    assert!(matches!(**obj, Expr::MetaClass(ref s, _) if s == "Point"));
                }
                _ => panic!("expected member on metaclass"),
            }
        }
        _ => panic!("expected assignment to metaclass static member"),
    }
}
