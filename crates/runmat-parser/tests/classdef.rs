use runmat_parser::Stmt;

mod parse;
use parse::parse;

fn single_stmt(src: &str) -> Stmt {
    let program = parse(src).unwrap();
    assert_eq!(program.body.len(), 1);
    program.body.into_iter().next().unwrap()
}

#[test]
fn classdef_minimal_empty() {
    let src = "classdef A\nend";
    let stmt = single_stmt(src);
    match stmt {
        Stmt::ClassDef {
            name,
            super_class,
            members,
            ..
        } => {
            assert_eq!(name, "A");
            assert!(super_class.is_none());
            assert!(members.is_empty());
        }
        _ => panic!("expected classdef"),
    }
}

#[test]
fn classdef_with_super_and_all_blocks() {
    let src = r#"
classdef MyClass < handle
  properties
    a, b; c
  end
  methods
    function y = f(x)
      y = x;
    end
    function z = g(u,v)
      z = u;
    end
  end
  events
    Started, Stopped;
  end
  enumeration
    Red, Green; Blue
  end
  arguments
    x, y;
  end
end
"#;
    let stmt = single_stmt(src);
    match stmt {
        Stmt::ClassDef {
            name,
            super_class,
            members,
            ..
        } => {
            assert_eq!(name, "MyClass");
            assert_eq!(super_class.as_deref(), Some("handle"));
            assert!(!members.is_empty());
        }
        _ => panic!("expected classdef"),
    }
}

#[test]
fn classdef_multiple_properties_lines_and_commas() {
    let src = r#"
classdef C
  properties
    a, b;
    c
  end
end
"#;
    let stmt = single_stmt(src);
    match stmt {
        Stmt::ClassDef { .. } => {}
        _ => panic!("expected classdef"),
    }
}

#[test]
fn classdef_parses_class_level_attributes() {
    let src = "classdef (Sealed) A\nend";
    let stmt = single_stmt(src);
    match stmt {
        Stmt::ClassDef {
            attributes, name, ..
        } => {
            assert_eq!(name, "A");
            assert_eq!(attributes.len(), 1);
            assert_eq!(attributes[0].name, "Sealed");
        }
        _ => panic!("expected classdef"),
    }
}

#[test]
fn classdef_parses_dependent_accessor_method_names() {
    let src = r#"
classdef D
  properties(Dependent)
    p
  end
  methods
    function val = get.p(obj)
      val = 2;
    end
    function obj = set.p(obj, val)
      obj = obj;
    end
  end
end
"#;
    let stmt = single_stmt(src);
    let Stmt::ClassDef { members, .. } = stmt else {
        panic!("expected classdef");
    };
    let methods = members
        .into_iter()
        .find_map(|member| match member {
            runmat_parser::ClassMember::Methods { body, .. } => Some(body),
            _ => None,
        })
        .expect("methods block");
    let names: Vec<String> = methods
        .into_iter()
        .filter_map(|stmt| match stmt {
            Stmt::Function { name, .. } => Some(name),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["get.p".to_string(), "set.p".to_string()]);
}

#[test]
fn classdef_methods_two_functions() {
    let src = r#"
classdef C
  methods
    function y = f(x)
      y = x;
    end
    function z = g(u)
      z = u;
    end
  end
end
"#;
    let stmt = single_stmt(src);
    match stmt {
        Stmt::ClassDef { .. } => {}
        _ => panic!("expected classdef"),
    }
}

#[test]
fn classdef_super_constructor_stmt_parses_as_semantic_super_ctor_call() {
    let src = "classdef B < A\nmethods\nfunction obj = B(v)\nobj@A(v);\nend\nend\nend";
    let program = runmat_parser::parse(src).expect("parse");
    let Stmt::ClassDef { members, .. } = &program.body[0] else {
        panic!("expected classdef");
    };
    let methods_body = match &members[0] {
        runmat_parser::ClassMember::Methods { body, .. } => body,
        _ => panic!("expected methods block"),
    };
    let Stmt::Function { body, .. } = &methods_body[0] else {
        panic!("expected constructor function");
    };
    let Stmt::Assign(var, expr, ..) = &body[0] else {
        panic!("expected rewritten assignment");
    };
    assert_eq!(var, "obj");
    let runmat_parser::Expr::SuperConstructorCall {
        current_class,
        super_class,
        args,
        ..
    } = expr
    else {
        panic!("expected semantic super constructor call");
    };
    assert_eq!(current_class, "B");
    assert_eq!(super_class, "A");
    assert_eq!(args.len(), 1);
}

#[test]
fn classdef_qualified_super_constructor_stmt_parses_as_semantic_super_ctor_call() {
    let src = "classdef B < pkg.A\nmethods\nfunction obj = B(v)\nobj@pkg.A(v);\nend\nend\nend";
    let program = runmat_parser::parse(src).expect("parse");
    let Stmt::ClassDef { members, .. } = &program.body[0] else {
        panic!("expected classdef");
    };
    let methods_body = match &members[0] {
        runmat_parser::ClassMember::Methods { body, .. } => body,
        _ => panic!("expected methods block"),
    };
    let Stmt::Function { body, .. } = &methods_body[0] else {
        panic!("expected constructor function");
    };
    let Stmt::Assign(var, expr, ..) = &body[0] else {
        panic!("expected rewritten assignment");
    };
    assert_eq!(var, "obj");
    let runmat_parser::Expr::SuperConstructorCall {
        current_class,
        super_class,
        args,
        ..
    } = expr
    else {
        panic!("expected semantic super constructor call");
    };
    assert_eq!(current_class, "B");
    assert_eq!(super_class, "pkg.A");
    assert_eq!(args.len(), 1);
}

#[test]
fn classdef_events_and_enumerations_arguments() {
    let src = r#"
classdef C
  events
    Ready, Done
  end
  enumeration
    Red, Green
  end
  arguments
    a, b
  end
end
"#;
    let stmt = single_stmt(src);
    match stmt {
        Stmt::ClassDef { .. } => {}
        _ => panic!("expected classdef"),
    }
}

#[test]
fn classdef_missing_end_in_block_errors() {
    let src = r#"
classdef C
  properties
    a, b
  methods
    function y = f(x)
      y = x;
    end
  end
end
"#; // missing 'end' after properties
    let err = parse(src).unwrap_err();
    assert!(err.message.contains("expected 'end'"));
}

#[test]
fn classdef_enumeration_and_arguments_blocks_parse() {
    let src = r#"
        classdef MyEnum
            enumeration
                Red, Green, Blue
            end
        end

        classdef C
            properties
                x
            end
            methods
                function obj = C()
                    obj.x = 1;
                end
            end
            arguments(Input)
                a
                b
            end
        end
    "#;
    let ast = runmat_parser::parse(src).unwrap();
    // Ensure both classes are present
    assert!(matches!(ast.body[0], runmat_parser::Stmt::ClassDef { .. }));
    assert!(matches!(ast.body[1], runmat_parser::Stmt::ClassDef { .. }));
}
