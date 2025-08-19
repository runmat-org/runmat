use runmat_parser::{parse_simple as parse, Stmt};

#[test]
fn varargin_must_be_last_param() {
    let err = parse("function y = f(varargin, x)\nend").unwrap_err();
    assert!(err.contains("varargin"));
}

#[test]
fn varargout_must_be_last_output() {
    let err = parse("function [varargout, y] = f(x)\nend").unwrap_err();
    assert!(err.contains("varargout"));
}

#[test]
fn simple_varargs_function_parses() {
    let program = parse("function varargout = f(varargin)\nend").unwrap();
    match &program.body[0] {
        Stmt::Function {
            name,
            params,
            outputs,
            ..
        } => {
            assert_eq!(name, "f");
            assert_eq!(params.as_slice(), ["varargin"]);
            assert_eq!(outputs.as_slice(), ["varargout"]);
        }
        _ => panic!("expected function stmt"),
    }
}
