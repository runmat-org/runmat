#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_numeric_transform_extensions_accept_every_exact_integer_class() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "x={constructor}([0 1 2 3]); y={constructor}([0 1 4 9]); q={constructor}([1 2]); v=pchip(x,y,q); if numel(v)~=2; error('pchip integer input'); end; p=pdf('Normal',{constructor}([0 1]),{constructor}(0),{constructor}(1)); if numel(p)~=2; error('pdf integer input'); end; d=pdist({constructor}([0;2;5])); if numel(d)~=3; error('pdist integer input'); end; d2=pdist2({constructor}([0;2]),{constructor}([1;3])); if any(size(d2)~=[2 2]); error('pdist2 integer input'); end; z=peaks({constructor}(3)); if any(size(z)~=[3 3]); error('peaks integer n'); end; s=periodogram({constructor}([0;1;0;1])); if isempty(s); error('periodogram integer signal'); end;"
        );
        execute_source(&source).expect("compiled numeric transform integer forms");
    }
}

#[test]
fn compiled_reordering_and_peaks_coordinates_preserve_all_integer_classes() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "v={constructor}([3 1 2]); P=perms(v); if ~isa(P,'{constructor}') || any(size(P)~=[6 3]); error('perms class'); end; A={constructor}([1 2 3;4 5 6]); B=permute(A,{constructor}([2 1])); if ~isa(B,'{constructor}') || any(size(B)~=[3 2]); error('permute class'); end; [X,Y,Z]=peaks({constructor}([0 1]),{constructor}([1 2])); if ~isa(X,'{constructor}') || ~isa(Y,'{constructor}') || X(2)~={constructor}(1) || Y(2)~={constructor}(2) || numel(Z)~=2; error('peaks coordinate class'); end;"
        );
        execute_source(&source).expect("compiled exact reordering semantics");
    }
}

#[test]
fn compiled_wide_uint64_reordering_and_coordinate_outputs_do_not_alias() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "base=bitshift(uint64(1),53); a=base+uint64(1); b=a+uint64(1); P=perms(uint64([a b])); if ~isa(P,'uint64') || P(1,1)~=b || P(1,2)~=a || P(2,1)~=a || P(2,2)~=b; error('wide perms'); end; A=uint64([a b]); B=permute(A,uint64([2 1])); if ~isa(B,'uint64') || B(1)~=a || B(2)~=b || any(size(B)~=[2 1]); error('wide permute'); end; [X,Y,Z]=peaks(uint64([a b]),uint64([1 2])); if X(1)~=a || X(2)~=b || Y(1)~=uint64(1) || Y(2)~=uint64(2) || numel(Z)~=2; error('wide peaks coordinates'); end;",
    )
    .expect("compiled wide exact transform semantics");
}
