#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_conversion_and_limit_semantics_are_exact() {
    execute_source(
        "a=int32(uint64(4294967295)); if ~isa(a,'int32') || a~=intmax('int32'); error('int32 saturation failed'); end; b=int2str(uint64(18446744073709551615)); if ~strcmp(b,'18446744073709551615'); error('int2str exactness failed'); end; c=intmax('like',uint64(1)); d=intmin('like',int64(1)); if ~isa(c,'uint64') || c~=uint64(18446744073709551615) || ~isa(d,'int64'); error('integer limits failed'); end;",
    )
    .expect("compiled integer conversion and limits");
}

#[test]
fn compiled_interpolation_extensions_are_runmat_only() {
    {
        let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        execute_source(
            "y=interp1(uint16([1 2 3]),uint16([10 20 40]),uint16([1 2])); if ~isequal(y,[10 20]); error('interp1 failed'); end; z=inv(uint16([4 1;2 3])); if abs(z(1,1)-0.3)>1e-12; error('inv failed'); end;",
        )
        .expect("compiled RunMat interpolation extensions");
    }
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("y=interp1(uint16([1 2]),uint16([3 4]),uint16(1));")
        .expect_err("integer interpolation must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Interp1IntegerSampleExtension")
    );
}
