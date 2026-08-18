use runmat_parser::{parse, Expr, IntegerLiteralClass, Stmt};

fn assigned_literal(source: &str) -> runmat_parser::IntegerLiteral {
    let program = parse(source).expect(source);
    match &program.body[0] {
        Stmt::Assign(_, Expr::IntegerLiteral(literal, _), _, _) => literal.clone(),
        other => panic!("unexpected statement: {other:?}"),
    }
}

#[test]
fn parser_preserves_exact_radix_class_and_bits() {
    for (source, class, bits) in [
        ("x = 0x2A;", IntegerLiteralClass::UInt8, 42),
        ("x = 0x100;", IntegerLiteralClass::UInt16, 256),
        ("x = 0xFFs8;", IntegerLiteralClass::Int8, 255),
        (
            "x = 0xFFFFFFFFFFFFFFFFu64;",
            IntegerLiteralClass::UInt64,
            u64::MAX,
        ),
    ] {
        let literal = assigned_literal(source);
        assert_eq!(literal.class(), class);
        assert_eq!(literal.bits(), bits);
    }
}

#[test]
fn parser_rejects_invalid_radix_digits_suffixes_and_widths() {
    for source in [
        "x = 0xGG;",
        "x = 0b102;",
        "x = 0xFFu9;",
        "x = 0x100u8;",
        "x = 0b100000000s8;",
        "x = 0x10000000000000000;",
    ] {
        let error = parse(source).expect_err(source);
        assert_eq!(error.position, 4, "{source}");
        assert!(
            error.message.contains("literal"),
            "{source}: {}",
            error.message
        );
    }
}
