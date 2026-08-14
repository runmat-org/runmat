#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_page_and_decomposition_semantics_cover_all_classes() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a={constructor}([1 2;3 4]); zero={constructor}(0); z=pagemtimes(a,a); x=pinv(a,zero); [q,r,p]=qr(a,zero); k=rank(a,zero); c=rcond(a); t=pagetranspose(a); y=real(a); ci=complex(a,a); ct=pagetranspose(ci); cy=real(ci); if ~isa(z,'double') || ~isa(x,'double') || ~isa(q,'double') || ~isa(r,'double') || ~isa(p,'double') || ~isa(k,'double') || ~isa(c,'double'); error('floating boundary class mismatch'); end; if ~isa(t,'{constructor}') || t(1,2)~=a(2,1) || t(2,1)~=a(1,2) || ~isa(y,'{constructor}') || ~isequal(y,a) || ~isa(ct,'{constructor}') || real(ct(1,2))~=a(2,1) || imag(ct(1,2))~=a(2,1) || ~isa(cy,'{constructor}') || ~isequal(cy,a); error('exact page projection mismatch'); end;"
        );
        execute_source(&source).expect("compiled RunMat integer linear-algebra extensions");
    }
}

#[test]
fn matlab_mode_rejects_integer_page_and_decomposition_extensions() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "z=pagemtimes(uint16([1 0;0 2]),uint16([1 0;0 2]));",
            "RunMat:compatibility:PagemtimesIntegerInputExtension",
        ),
        (
            "x=pinv(uint16([1 0;0 2]));",
            "RunMat:compatibility:PinvIntegerInputExtension",
        ),
        (
            "[q,r,p]=qr(uint16([1 0;0 2]));",
            "RunMat:compatibility:QrIntegerInputExtension",
        ),
        (
            "k=rank(uint16([1 0;0 2]));",
            "RunMat:compatibility:RankIntegerInputExtension",
        ),
        (
            "c=rcond(uint16([1 0;0 2]));",
            "RunMat:compatibility:RcondIntegerInputExtension",
        ),
        (
            "x=pinv([1 0;0 2],uint16(0));",
            "RunMat:compatibility:PinvIntegerToleranceExtension",
        ),
        (
            "[q,r,p]=qr([1 0;0 2],uint16(0));",
            "RunMat:compatibility:QrIntegerOptionExtension",
        ),
        (
            "k=rank([1 0;0 2],uint16(0));",
            "RunMat:compatibility:RankIntegerToleranceExtension",
        ),
        (
            "x=pinv(logical([1 0;0 1]));",
            "RunMat:compatibility:PinvLogicalInputExtension",
        ),
        (
            "x=pinv([1 0;0 1],false);",
            "RunMat:compatibility:PinvLogicalToleranceExtension",
        ),
        (
            "[q,r,p]=qr(logical([1 0;0 1]));",
            "RunMat:compatibility:QrLogicalInputExtension",
        ),
        (
            "[q,r,p]=qr([1 0;0 1],false);",
            "RunMat:compatibility:QrLogicalOptionExtension",
        ),
        (
            "k=rank(logical([1 0;0 1]));",
            "RunMat:compatibility:RankLogicalInputExtension",
        ),
        (
            "k=rank([1 0;0 1],false);",
            "RunMat:compatibility:RankLogicalToleranceExtension",
        ),
        (
            "c=rcond(logical([1 0;0 1]));",
            "RunMat:compatibility:RcondLogicalInputExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict compatibility gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_wide_integer_decomposition_boundaries_reject_without_rounding() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "w=uint64(9007199254740992)+uint64(1); z=pagemtimes(w,w);",
        "w=uint64(9007199254740992)+uint64(1); x=pinv(w);",
        "w=uint64(9007199254740992)+uint64(1); [q,r,p]=qr(w);",
        "w=uint64(9007199254740992)+uint64(1); k=rank(w);",
        "w=uint64(9007199254740992)+uint64(1); c=rcond(w);",
        "w=uint64(9007199254740992)+uint64(1); x=pinv(1,w);",
        "w=uint64(9007199254740992)+uint64(1); k=rank(1,w);",
    ] {
        let error = execute_source(source).expect_err("lossy binary64 boundary must reject");
        assert!(
            error.message().contains("exactly representable"),
            "{source}: {error}"
        );
    }
}

#[test]
fn compiled_pagetranspose_and_real_preserve_wide_integer_storage_exactly() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "b=uint64(9007199254740992); a=reshape([b+uint64(1) b+uint64(2) b+uint64(3) b+uint64(4)],[2 2]); t=pagetranspose(a); if ~isa(t,'uint64') || t(1,2)~=a(2,1) || t(2,1)~=a(1,2); error('pagetranspose exact integer mismatch'); end; r=real(a); if ~isa(r,'uint64') || ~isequal(r,a); error('real exact integer mismatch'); end; z=complex(a,uint64([1 2;3 4])); rz=real(z); if ~isa(rz,'uint64') || ~isequal(rz,a); error('real complex-integer projection mismatch'); end;",
    )
    .expect("compiled exact page transpose and real projection");
}

#[test]
fn realsqrt_rejects_integer_classes_in_both_language_modes() {
    for enabled in [false, true] {
        let _mode = runmat_runtime::compatibility::push_runmat_extensions_enabled(enabled);
        for constructor in [
            "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        ] {
            let source = format!("y=realsqrt({constructor}([0 1 4]));");
            let error = execute_source(&source).expect_err("realsqrt integer class must reject");
            assert_eq!(
                error.identifier(),
                Some("RunMat:realsqrt:InvalidInput"),
                "{constructor}"
            );
        }
    }
}
