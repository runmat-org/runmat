use runmat_hir::lower;
use runmat_parser::parse;

#[test]
fn lower_various_valid_seeds() {
	let seeds = vec![
		"A=[1 2;3 4]; B=A(:,2);",
		"try; error('MATLAB:foo','msg'); catch e; id=getfield(e,'identifier'); end",
		"function y = f(x); y = x + 1; end; a = f(2);",
		"classdef C\n  properties\n    a\n  end\n  methods\n    function obj = setA(obj, v)\n      obj.a = v;\n    end\n  end\nend",
		"import pkg.*; x=1;",
	];
	for (i, src) in seeds.iter().enumerate() {
		let ast = parse(src).expect("parse");
		let res = lower(&ast);
		assert!(res.is_ok(), "seed {} failed to lower: {:?}", i, res.err());
	}
}


