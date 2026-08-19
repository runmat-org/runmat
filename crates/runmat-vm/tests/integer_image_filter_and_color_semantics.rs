#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_image_integer_classes_shapes_and_values_follow_public_forms() {
    execute_source(
        "a = uint8(reshape([1 2 3 4],[2 2])); h = [1 1;1 1]; b = imfilter(a,h); if ~isa(b,'uint8') || ~isequal(size(b),[2 2]); error('imfilter class/shape failed'); end; [n,x] = imhist(a); if ~isa(n,'double') || ~isa(x,'double') || size(n,1) ~= 256; error('imhist outputs failed'); end; map = [1 0 0;0 0 1]; rgb = ind2rgb(uint8([0 1]),map); if ~isa(rgb,'double') || ~isequal(size(rgb),[1 2 3]); error('ind2rgb failed'); end; gray = rgb2gray(uint8(reshape([255 0 0],[1 1 3]))); if ~isa(gray,'uint8') || ~isequal(size(gray),[1 1]); error('rgb2gray failed'); end; hsv = rgb2hsv(uint8(reshape([255 0 0],[1 1 3]))); if ~isa(hsv,'double') || ~isequal(size(hsv),[1 1 3]); error('rgb2hsv failed'); end; lab = rgb2lab(uint16(reshape([65535 65535 65535],[1 3]))); if ~isa(lab,'double') || ~isequal(size(lab),[1 3]); error('rgb2lab failed'); end; z = imag(uint64([1 9007199254740993])); if ~isa(z,'uint64') || any(z ~= uint64(0)); error('imag integer projection failed'); end;",
    )
    .expect("compiled image/filter/color integer semantics");
}

#[test]
fn compiled_image_integer_rejections_and_extensions_are_stable() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let valid = execute_source("x = imfilter(uint8([1 2]),[1 1],'valid');")
        .expect_err("valid output extension must be gated");
    assert_eq!(
        valid.identifier(),
        Some("RunMat:compatibility:ImfilterValidExtension")
    );
    execute_source(
        "ok = false; try; x = ind2rgb(uint8(0),uint8([255 0 0])); catch; ok = true; end; if ~ok; error('integer colormap must reject'); end; ok = false; try; y = rgb2hsv(int16(reshape([1 2 3],[1 1 3]))); catch; ok = true; end; if ~ok; error('unsupported color class must reject'); end;",
    )
    .expect("compiled image integer rejection semantics");
}
