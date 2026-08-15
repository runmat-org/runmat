#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_floating_boundary_extensions_are_gated_before_conversion() {
    for (source, identifier) in [
        (
            "y=sqrt(uint8(4));",
            "RunMat:compatibility:SqrtIntegerInputExtension",
        ),
        (
            "y=squareform(uint8([1 2 3]));",
            "RunMat:compatibility:SquareformIntegerInputExtension",
        ),
        (
            "sys=ss(uint8(1),1,1,1);",
            "RunMat:compatibility:SsIntegerMatrixInputExtension",
        ),
    ] {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let error = execute_source(source).expect_err("typed extension must be gated");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_squareform_and_sscanf_preserve_native_integer_storage() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("m=squareform(uint64([9007199254740993 2 3])); if ~isa(m,'uint64') || m(2,1) ~= uint64(9007199254740993); error('squareform exact class'); end; v=squareform(m); if ~isa(v,'uint64') || v(1) ~= uint64(9007199254740993); error('squareform round trip'); end; s=sscanf(\"-9223372036854775808 9223372036854775807\",\"%ld\"); if ~isa(s,'int64') || s(1) ~= intmin('int64') || s(2) ~= intmax('int64'); error('sscanf signed long'); end; u=sscanf(\"18446744073709551615 ff\",\"%lu %lx\"); if ~isa(u,'uint64') || u(1) ~= intmax('uint64') || u(2) ~= uint64(255); error('sscanf unsigned long'); end;")
        .expect("compiled exact structural and scan semantics");
}

#[test]
fn compiled_documented_structural_forms_keep_integer_values_out_of_double() {
    execute_source("s=speye(uint16(3)); if ~isequal(size(s),[3 3]); error('speye integer size'); end; a=reshape(uint64([9007199254740993 2]),[1 1 2]); b=squeeze(a); if ~isa(b,'uint64') || b(1) ~= uint64(9007199254740993); error('squeeze exact storage'); end; t=sprintf('%u',uint64(18446744073709551615)); if ~strcmp(t,'18446744073709551615'); error('sprintf exact formatting'); end;")
        .expect("compiled documented integer structural forms");
}
