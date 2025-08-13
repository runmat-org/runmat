use runmat_parser::{parse_simple as parse, Stmt};

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
        Stmt::ClassDef { name, super_class, members } => {
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
        Stmt::ClassDef { name, super_class, members } => {
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
    assert!(err.contains("expected 'end'"));
}


