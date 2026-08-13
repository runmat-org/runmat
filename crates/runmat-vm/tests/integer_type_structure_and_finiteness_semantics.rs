#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_type_equality_and_permutation_semantics_are_exact() {
    execute_source(
        "x=intmax('uint64'); if ~isa(x,'uint64') || ~isUnderlyingType(x,'uint64'); error('type predicates failed'); end; if ~isequal(x,intmax('uint64')) || isequal(x,uint64(0)); error('wide equality failed'); end; a=uint16(reshape(1:8,[2 2 2])); b=ipermute(permute(a,[2 1 3]),[2 1 3]); if ~isequal(a,b) || ~isa(b,'uint16'); error('ipermute failed'); end;",
    )
    .expect("compiled integer type, equality, and permutation semantics");
}

#[test]
fn compiled_integer_structure_and_finiteness_semantics_are_coherent() {
    execute_source(
        "a=int32([1 2;3 4]); if ~iscolumn(int32([1;2])) || iscolumn(a) || isempty(a); error('shape predicates failed'); end; if ~all(isfinite(a),'all') || any(isinf(a),'all'); error('finite predicates failed'); end; if isStringScalar(a) || iscell(a) || ischar(a) || isletter(a); error('universal predicates failed'); end;",
    )
    .expect("compiled integer structure and finiteness semantics");
}

#[test]
fn compiled_structure_extensions_are_gated_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error =
        execute_source("tf=isdiag(int32([1 0;0 1]));").expect_err("integer isdiag must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:IsdiagIntegerInputExtension")
    );
}

#[test]
fn compiled_integer_shape_missing_and_text_predicates_are_exact() {
    execute_source(
        "a=uint64([65 49 intmax('uint64')]); p=isstrprop(a,'alpha'); if ~isequal(p,logical([1 0 0])); error('isstrprop failed'); end; if ~ismatrix(a) || ~isrow(a) || isscalar(a) || ~isvector(a); error('shape predicates failed'); end; if any(isnan(a),'all') || any(ismissing(a),'all') || ~isnumeric(a) || ~isreal(a) || islogical(a) || issparse(a); error('numeric predicates failed'); end; if isstring(a) || istable(a) || istimetable(a) || isordinal(a) || isvarname(a); error('universal predicates failed'); end;",
    )
    .expect("compiled integer shape, missing, and text predicates");
}

#[test]
fn compiled_integer_structure_extensions_are_gated() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "tf=issymmetric(int32([1 0;0 1]));",
        "y=istril(int32([1 0;2 3]));",
        "y=istriu(int32([1 2;0 3]));",
    ] {
        let error = execute_source(source).expect_err("integer structure form must be gated");
        assert!(
            error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")),
            "{source}: {error}"
        );
    }
}
