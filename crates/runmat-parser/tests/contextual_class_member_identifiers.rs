use runmat_lexer::{tokenize_detailed, Token};
use runmat_parser::{parse, ClassMember, Expr, Stmt, UnOp};

#[test]
fn class_member_words_remain_identifiers_outside_block_position() {
    for name in ["properties", "methods", "events", "enumeration"] {
        for source in [
            format!("{name}(obj);"),
            format!("result = {name}(obj);"),
            format!("result = obj.{name};"),
            format!("result = @{name};"),
            format!("result = @pkg.{name};"),
            format!("{name} = 1;"),
            format!("obj.{name} = 1;"),
            format!("[out, {name}] = f();"),
            format!("function {name} = f({name})\n{name} = {name} + 1;\nend"),
        ] {
            parse(&source).unwrap_or_else(|error| panic!("{source}: {error}"));
        }

        let source = format!("{name}'");
        let tokens = tokenize_detailed(&source);
        assert_eq!(
            tokens.last().map(|token| &token.token),
            Some(&Token::Transpose)
        );
        let program = parse(&source).unwrap();
        assert!(matches!(
            program.body.as_slice(),
            [Stmt::ExprStmt(Expr::Unary(UnOp::Transpose, _, _), _, _)]
        ));
    }
}

#[test]
fn class_block_dispatch_is_still_contextual() {
    let program = parse(
        "classdef C\nproperties\n p\nend\nmethods\n function y = events(obj)\n y = properties(obj);\n end\nend\nevents\n E\nend\nenumeration\n A\nend\nend",
    )
    .unwrap();
    let [Stmt::ClassDef { members, .. }] = program.body.as_slice() else {
        panic!("expected classdef")
    };
    assert!(matches!(
        members.as_slice(),
        [
            ClassMember::Properties { .. },
            ClassMember::Methods { .. },
            ClassMember::Events { .. },
            ClassMember::Enumeration { .. }
        ]
    ));
}

#[test]
fn non_contextual_keywords_remain_reserved() {
    for source in ["true = 1;", "end = 1;", "global = 1;"] {
        assert!(parse(source).is_err(), "{source}");
    }
}
