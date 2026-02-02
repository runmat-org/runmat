use runmat_hir::{lower, LoweringContext};
use runmat_parser::parse;

#[test]
fn lower_various_valid_seeds() {
    let seeds = [
        "A=[1 2;3 4]; B=A(:,2);",
        "try; error('MATLAB:foo','msg'); catch e; id=getfield(e,'identifier'); end",
        "function y = f(x); y = x + 1; end; a = f(2);",
        "classdef C\nend",
        "import pkg.*; x=1;",
    ];
    for (i, src) in seeds.iter().enumerate() {
        println!("seed {i} input: {src}");
        let ast = parse(src).expect("parse");
        let res = lower(&ast, &LoweringContext::empty());
        assert!(res.is_ok(), "seed {i} failed to lower: {:?}", res.err());
    }
}

#[test]
fn validate_classdefs_basic() {
    // Well-formed classdef: no error during lowering
    let src = "classdef C\n properties; a; end; methods; function obj=setA(obj,v); end; end; end";
    let ast = parse(src).unwrap();
    let hir = lower(&ast, &LoweringContext::empty());
    assert!(hir.is_ok());

    // Duplicate property should error at lowering time now
    let dup = "classdef D\n properties; a; a; end; end";
    let ast2 = parse(dup).unwrap();
    let hir2 = lower(&ast2, &LoweringContext::empty());
    assert!(hir2.is_err());
}

#[test]
fn collect_imports_list() {
    let src = "import pkg.sub.*; import top.mid.leaf; x=1;";
    let ast = parse(src).unwrap();
    let hir = lower(&ast, &LoweringContext::empty()).unwrap().hir;
    let imports = runmat_hir::collect_imports(&hir);
    assert_eq!(imports.len(), 2);
    assert!(imports[0].1);
    assert_eq!(
        imports[1].0,
        vec!["top".to_string(), "mid".to_string(), "leaf".to_string()]
    );
}
