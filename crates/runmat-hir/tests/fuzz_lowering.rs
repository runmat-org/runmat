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

#[test]
fn validate_classdefs_basic() {
    // Well-formed classdef: no error
    let src = "classdef C\n properties; a; end; methods; function obj=setA(obj,v); end; end; end";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    assert!(runmat_hir::validate_classdefs(&hir).is_ok());

    // Duplicate property should error
    let dup = "classdef D\n properties; a; a; end; end";
    let ast2 = parse(dup).unwrap();
    let hir2 = lower(&ast2).unwrap();
    assert!(runmat_hir::validate_classdefs(&hir2).is_err());
}

#[test]
fn collect_imports_list() {
    let src = "import pkg.sub.*; import top.mid.leaf; x=1;";
    let ast = parse(src).unwrap();
    let hir = lower(&ast).unwrap();
    let imports = runmat_hir::collect_imports(&hir);
    assert_eq!(imports.len(), 2);
    assert_eq!(imports[0].1, true);
    assert_eq!(imports[1].0, vec!["top".to_string(), "mid".to_string(), "leaf".to_string()]);
}


