#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_jsonencode_preserves_exact_uint64_decimal_text() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "encoded=jsonencode(intmax('uint64')); if ~strcmp(encoded,'18446744073709551615'); error('jsonencode exact uint64 failed'); end;",
    )
    .expect("compiled exact uint64 JSON serialization");
}

#[test]
fn compiled_join_distinguishes_documented_and_runmat_dimension_classes() {
    {
        let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        execute_source(
            "joined=join([\"a\" \"b\"],'-',2); if joined ~= \"a-b\"; error('documented join dimension failed'); end;",
        )
        .expect("compiled documented double join dimension");

        let error = execute_source("joined=join([\"a\" \"b\"],'-',uint8(2));")
            .expect_err("typed join dimension must reject in MATLAB-compatible mode");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:JoinTypedIntegerDimensionExtension")
        );
    }

    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "joined=join([\"a\" \"b\"],'-',uint8(2)); if joined ~= \"a-b\"; error('typed join dimension failed'); end;",
    )
    .expect("compiled RunMat typed join dimension");
}

#[test]
fn compiled_integer_statistical_extensions_execute_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "[h,p,ks,cv]=kstest(int16([-1;0;1])); if ~islogical(h) || p < 0 || p > 1 || ks < 0 || cv < 0; error('integer kstest failed'); end; B=lasso(int16([0;1;2;3]),int32([1;3;5;7]),'Lambda',0,'Standardize',false); if ~isa(B,'double') || abs(B(1)-2)>1e-8; error('integer lasso failed'); end; G=lassoglm(int16([0;1;2;3]),int32([1;3;5;7]),'normal','Lambda',0,'Standardize',false); if ~isa(G,'double') || abs(G(1)-2)>1e-3; error('integer lassoglm failed'); end;",
    )
    .expect("compiled RunMat integer statistical semantics");
}

#[test]
fn compiled_statistical_integer_boundaries_reject_lossy_uint64() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (source, identifier) in [
        (
            "wide=uint64(9007199254740992)+uint64(1); h=kstest([wide;wide+uint64(2)]);",
            "RunMat:kstest:InvalidArgument",
        ),
        (
            "wide=uint64(9007199254740992)+uint64(1); B=lasso([wide;wide+uint64(2)],uint8([1;2]));",
            "RunMat:lasso:InvalidArgument",
        ),
        (
            "wide=uint64(9007199254740992)+uint64(1); B=lassoglm([wide;wide+uint64(2)],uint8([1;2]),'normal');",
            "RunMat:lassoglm:InvalidArgument",
        ),
    ] {
        let error = execute_source(source).expect_err("lossy integer boundary must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}: {error}");
        assert!(
            error.message().contains("exactly representable"),
            "{source}: {error}"
        );
    }
}
