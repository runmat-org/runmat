#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_documented_multidimensional_fft_and_table_integer_forms_execute() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "a = uint64([9007199254740993 7; 5 9]); h = head(a, uint8(1)); if ~isa(h,'uint64') || h(1) ~= uint64(9007199254740993) || h(2) ~= uint64(7); error('head integer contract failed'); end; if height(a) ~= 2; error('height contract failed'); end; y2 = ifft2(uint16([1 0; 0 0])); if ~isa(y2,'double') || ~isequal(size(y2), [2 2]); error('ifft2 integer contract failed'); end; yn = ifftn(uint8(reshape(1:8, [2 2 2]))); if ~isa(yn,'double') || ~isequal(size(yn), [2 2 2]); error('ifftn integer contract failed'); end; s = ifftshift(uint64([9007199254740993 7 9])); if ~isa(s,'uint64') || s(1) ~= uint64(7); error('ifftshift integer contract failed'); end;",
    )
    .expect("compiled documented multidimensional FFT and table integer forms");
}

#[test]
fn compiled_multidimensional_fft_extensions_have_stable_strict_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "y = ifft2(uint64([1 0; 0 0]));",
            "RunMat:compatibility:Ifft2WideIntegerDataExtension",
        ),
        (
            "y = ifftn(uint64(1));",
            "RunMat:compatibility:IfftnWideIntegerDataExtension",
        ),
        (
            "y = ifftshift(uint8([1 2]), [1 2]);",
            "RunMat:compatibility:IfftshiftMultiDimensionExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_head_preserves_nd_integer_pages() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "a = reshape(uint64(9007199254740992) + uint64(0:11), [2 2 3]); h = head(a, uint8(1)); if ~isa(h,'uint64') || ~isequal(size(h), [1 2 3]) || h(1,1,1) ~= uint64(9007199254740992) || h(1,2,3) ~= uint64(9007199254741002); error('head N-D integer contract failed'); end;",
    )
    .expect("compiled N-D integer head");
}
