#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn documented_integer_complex_and_image_forms_execute_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "z=imag(uint64(9007199254740992)+uint64(1)); if ~isa(z,'uint64') || z~=uint64(0); error('imag integer class failed'); end; f=imfilter(uint8([1 2;3 4]),1); if ~isa(f,'uint8') || ~isequal(f,uint8([1 2;3 4])); error('imfilter integer failed'); end; [c,b]=imhist(uint8([0 255 255]),2); if ~isa(c,'double') || ~isa(b,'double') || ~isequal(c,[1;2]); error('imhist integer failed'); end; g=rgb2gray(uint8(reshape([255 0 0],[1 1 3]))); if ~isa(g,'uint8') || g~=uint8(76); error('rgb2gray integer failed'); end; l=rgb2lab(uint8(reshape([255 255 255],[1 1 3]))); if ~isa(l,'double'); error('rgb2lab integer boundary failed'); end; r=lab2rgb([70 5 10],'OutputType','uint8'); if ~isa(r,'uint8') || ~isequal(size(r),[1 3]); error('lab2rgb OutputType failed'); end;",
    )
    .expect("documented integer complex and image forms");
}

#[test]
fn imhist_typed_bin_count_is_mode_gated() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "[c,b]=imhist(uint8([0 255]),uint16(2));",
        "[c,b]=imhist(uint8([0 255]),int16(-1));",
    ] {
        let error = execute_source(source).expect_err("typed integer bin count must be gated");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ImhistTypedIntegerBinCountExtension")
        );
    }
}

#[test]
fn impulse_integer_roles_are_mode_gated() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("sys=tf(uint8(1),[1 1]); y=impulse(sys,uint8(1));")
        .expect_err("typed integer impulse roles must reject");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:ImpulseIntegerExtension")
    );
}

#[test]
fn lab2rgb_integer_input_rejects_without_floating_coercion() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "y=lab2rgb(uint8(reshape([70 5 10],[1 1 3])));",
        "y=lab2rgb(int64(reshape([70 5 10],[1 1 3])));",
    ] {
        let error = execute_source(source).expect_err("integer LAB input must reject");
        assert_eq!(error.identifier(), Some("RunMat:lab2rgb:InvalidInput"));
    }
}
