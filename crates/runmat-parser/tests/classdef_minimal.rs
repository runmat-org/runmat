mod parse;
use parse::parse;

#[test]
fn classdef_minimal_structure() {
    let src = r#"
classdef MyClass < handle
  properties
    a, b;
  end
  methods
    function y = f(x)
      y = x;
    end
  end
  events
    Started
  end
  enumeration
    Red, Green
  end
  arguments
    x, y
  end
end
"#;
    // We only check that parsing succeeds and yields a single statement for now
    let program = parse(src).unwrap();
    assert_eq!(program.body.len(), 1);
}
