use runmat_parser::{parse_simple as parse, BinOp, Expr, LValue, Stmt};

#[test]
fn paren_slice_assignment_variants() {
    let program = parse("A(:, i) = v; A(1:3, :) = v").unwrap();
    assert_eq!(program.body.len(), 2);

    match &program.body[0] {
        Stmt::AssignLValue(LValue::Index(base, idxs), rhs, _) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "A"));
            assert_eq!(idxs.len(), 2);
            assert!(matches!(idxs[0], Expr::Colon));
            assert!(matches!(idxs[1], Expr::Ident(ref n) if n == "i"));
            assert!(matches!(rhs, Expr::Ident(ref n) if n == "v"));
        }
        _ => panic!("expected A(:,i)=v as index assignment"),
    }

    match &program.body[1] {
        Stmt::AssignLValue(LValue::Index(base, idxs), _rhs, _) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "A"));
            assert_eq!(idxs.len(), 2);
            assert!(matches!(idxs[0], Expr::Range(_, _, _)));
            assert!(matches!(idxs[1], Expr::Colon));
        }
        _ => panic!("expected A(1:3,:)=v as index assignment"),
    }
}

#[test]
fn logical_index_assignment() {
    let program = parse("A(A>5) = 0").unwrap();
    match &program.body[0] {
        Stmt::AssignLValue(LValue::Index(base, idxs), rhs, _) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "A"));
            assert_eq!(idxs.len(), 1);
            assert!(matches!(idxs[0], Expr::Binary(_, BinOp::Greater, _)));
            assert!(matches!(rhs, Expr::Number(ref n) if n == "0"));
        }
        _ => panic!("expected logical index assignment"),
    }
}

#[test]
fn cell_and_member_lvalue_assignment() {
    let program = parse("C{1,2} = v; s.field = w").unwrap();
    assert_eq!(program.body.len(), 2);
    match &program.body[0] {
        Stmt::AssignLValue(LValue::IndexCell(base, idxs), rhs, _) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "C"));
            assert_eq!(idxs.len(), 2);
            assert!(matches!(rhs, Expr::Ident(ref n) if n == "v"));
        }
        _ => panic!("expected C{{1,2}}=v as cell content assignment"),
    }
    match &program.body[1] {
        Stmt::AssignLValue(LValue::Member(base, name), rhs, _) => {
            assert!(matches!(**base, Expr::Ident(ref n) if n == "s"));
            assert_eq!(name, "field");
            assert!(matches!(rhs, Expr::Ident(ref n) if n == "w"));
        }
        _ => panic!("expected s.field=w as member assignment"),
    }
}

#[test]
fn deeply_nested_lvalue_assignment() {
    let program = parse("A(1).field{2}(3:5) = val").unwrap();
    match &program.body[0] {
        Stmt::AssignLValue(LValue::Index(base, idxs), rhs, _) => {
            // base should be IndexCell(Member(Index(Ident(A), [1]), 'field'), [2])
            match &**base {
                Expr::IndexCell(member, braces) => {
                    assert_eq!(braces.len(), 1);
                    assert!(matches!(braces[0], Expr::Number(ref n) if n == "2"));
                    match &**member {
                        Expr::Member(indexed, fname) => {
                            assert_eq!(fname, "field");
                            match &**indexed {
                                Expr::Index(id, paren_idxs) => {
                                    assert!(matches!(**id, Expr::Ident(ref n) if n == "A"));
                                    assert_eq!(paren_idxs.len(), 1);
                                    assert!(
                                        matches!(paren_idxs[0], Expr::Number(ref n) if n == "1")
                                    );
                                }
                                _ => panic!("member base not Index"),
                            }
                        }
                        _ => panic!("base not Member"),
                    }
                }
                _ => panic!("outer base not IndexCell"),
            }
            assert_eq!(idxs.len(), 1);
            assert!(matches!(idxs[0], Expr::Range(_, _, _)));
            assert!(matches!(rhs, Expr::Ident(ref n) if n == "val"));
        }
        _ => panic!("expected deeply nested lvalue assignment"),
    }
}

#[test]
fn multiple_assignment_kinds_sequence() {
    let program = parse("A=1; A(1)=2; A{1}=3; s.f = 4").unwrap();
    assert_eq!(program.body.len(), 4);
}

#[test]
fn dynamic_member_assignment_parses() {
    let program = parse("s.(idx) = value").unwrap();
    assert_eq!(program.body.len(), 1);
    match &program.body[0] {
        Stmt::AssignLValue(LValue::MemberDynamic(base, name_expr), rhs, suppressed) => {
            assert!(!*suppressed);
            assert!(matches!(**base, Expr::Ident(ref n) if n == "s"));
            assert!(matches!(**name_expr, Expr::Ident(ref n) if n == "idx"));
            assert!(matches!(rhs, Expr::Ident(ref n) if n == "value"));
        }
        other => panic!("expected dynamic member assignment, got {other:?}"),
    }
}
