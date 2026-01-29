mod parse;
use parse::parse;

#[test]
fn parse_various_valid_seeds() {
    let seeds = [
		"A=[1 2;3 4]; B=A(:,2);",
		"try; error('MATLAB:foo','msg'); catch e; id=getfield(e,'identifier'); end",
		"function y = f(x); y = x + 1; end; a = f(2);",
		"classdef C\n  properties\n    a\n  end\n  methods\n    function obj = setA(obj, v)\n      obj.a = v;\n    end\n  end\nend",
		"import pkg.*; x=1;",
	];
    for (i, src) in seeds.iter().enumerate() {
        let res = parse(src);
        assert!(res.is_ok(), "seed {} failed to parse: {:?}", i, res.err());
    }
}
