#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_division_comparison_shape_and_logic_are_exact() {
    execute_source(
        "a=uint64([0 intmax('uint64')]); if ~isequal(logical(a),logical([0 1])); error('logical conversion failed'); end; if length(a)~=2; error('length failed'); end; hi=intmax('uint64'); lo=hi-uint64(1); if ~isequal(lt(lo,hi),true); error('lt precision failed'); end; if ~isequal(le(int64(-1),uint64(0)),true); error('le precision failed'); end; q=ldivide(int16([2 4]),int16([5 9])); if ~isa(q,'int16') || ~isequal(q,int16([3 2])); error('ldivide failed'); end;",
    )
    .expect("compiled integer division, comparison, shape, and logical semantics");
}

#[test]
fn compiled_integer_sequence_counts_and_kronecker_product_are_exact() {
    execute_source(
        "a=kron(uint16([1 2]),uint16([3;4])); if ~isa(a,'uint16') || ~isequal(a,uint16([3 6;4 8])); error('kron failed'); end; x=linspace(0,1,uint8(3)); if ~isequal(x,[0 0.5 1]); error('linspace count failed'); end; y=logspace(0,1,uint8(3)); if length(y)~=3 || abs(y(1)-1)>1e-12 || abs(y(3)-10)>1e-12; error('logspace count failed'); end;",
    )
    .expect("compiled integer sequence counts and Kronecker semantics");
}

#[test]
fn compiled_floating_only_integer_extensions_are_gated_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "y=log(uint8(2));",
        "y=log10(uint8(10));",
        "y=log1p(uint8(1));",
        "y=log2(uint8(2));",
        "x=linsolve(int8(1),int8(1));",
        "x=linprog(int8(-1),int8(1),int8(1));",
        "[l,u]=lu(int8([1 0;0 1]));",
    ] {
        let error = execute_source(source).expect_err("integer extension must be gated");
        assert!(
            error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")),
            "{source}: {error}"
        );
    }
}

#[test]
fn compiled_logspace_pi_endpoint_and_logical_errors_match_contract() {
    execute_source("x=logspace(0,pi,4); if abs(x(4)-pi)>1e-12; error('pi endpoint failed'); end;")
        .expect("compiled logspace pi endpoint");
    for source in ["x=logical(NaN);", "x=logical(1+1i);"] {
        execute_source(source).expect_err("invalid logical conversion must reject");
    }
}
