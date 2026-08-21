#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_histogram_and_image_integer_forms_preserve_contracts() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    execute_source(
        "u = uint16([0 128 65535]); d = im2double(u); if ~isa(d,'double') || d(1) ~= 0 || d(3) ~= 1; error('im2double contract failed'); end; b = im2uint8(uint16([127 128 383 384])); if ~isa(b,'uint8') || ~isequal(b,uint8([0 1 1 2])); error('im2uint8 contract failed'); end; w = im2uint16(uint8([0 128 255])); if ~isa(w,'uint16') || ~isequal(w,uint16([0 32896 65535])); error('im2uint16 contract failed'); end; [n,xe,ye,bx,by] = histcounts2(uint64([9007199254740993 9007199254740994]), uint8([1 2]), uint64([9007199254740993 9007199254740994 9007199254740995]), uint8([1 2 3])); if ~isequal(size(n),[2 2]) || bx(1) ~= 1 || bx(2) ~= 2 || by(1) ~= 1 || by(2) ~= 2; error('histcounts2 contract failed'); end; h = image(uint8(reshape([255 0 0 255 0 0],[1 2 3]))); if ~isa(get(h,'CData'),'uint8') || ~strcmp(get(h,'CDataMapping'),'direct'); error('image CData contract failed'); end; hs = imagesc(uint16([1 2;3 4]), [0 5]); if ~isa(get(hs,'CData'),'uint16') || ~strcmp(get(hs,'CDataMapping'),'scaled'); error('imagesc CData contract failed'); end;",
    )
    .expect("compiled histogram and image integer contracts");
}

#[test]
fn compiled_histogram_and_image_extensions_have_stable_identifiers() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "[n,c] = hist(uint8([1 2 3]));",
            "RunMat:compatibility:HistIntegerDataExtension",
        ),
        (
            "n = histc(uint64([1 2]),uint64([1 2]));",
            "RunMat:compatibility:HistcWideIntegerExtension",
        ),
        (
            "h = image(ones(1,1,4));",
            "RunMat:compatibility:ImageFourChannelCDataExtension",
        ),
        (
            "h = imagesc(ones(1,1,4));",
            "RunMat:compatibility:ImagescFourChannelCDataExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("extension should reject in MATLAB mode");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_histogram_image_cohort_covers_successful_integer_and_logical_forms() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "[hn,hc] = hist(uint8([1 2 2 3]), [1 2 3]); [hm,cm] = hist(uint8([1 2;2 3]),2);",
        "[cn,cb] = histc(uint32([1 2 2 3]),uint32([1 2 3]));",
        "hh = histogram(uint16([0 1 1 2]),[0 1 2 3],'Normalization','probability'); if ~isa(get(hh,'Data'),'uint16'); error('histogram Data class failed'); end;",
        "h2 = histogram2(uint16([0 0 1 1]),uint16([0 1 0 1]),[0 1 2],[0 1 2]); if ~isa(get(h2,'Data'),'uint16'); error('histogram2 Data class failed'); end;",
        "hh = histogram(uint8(1)); if ~isa(get(hh,'Data'),'uint8'); error('scalar histogram Data class failed'); end;",
        "h2 = histogram2(uint16(1),uint16(2)); if ~isa(get(h2,'Data'),'uint16'); error('scalar histogram2 Data class failed'); end;",
        "hi = image(uint32(3)); if ~isa(get(hi,'CData'),'uint32'); error('scalar image CData class failed'); end;",
        "hs = imagesc(int16(4)); if ~isa(get(hs,'CData'),'int16'); error('scalar imagesc CData class failed'); end;",
        "rgb = hsv2rgb(logical(reshape([0 1 1],[1 1 3]))); if ~isa(rgb,'double'); error('hsv2rgb class failed'); end;",
        "i8 = im2uint8([1 2 256],'indexed'); i16 = im2uint16([1 2 65536],'indexed'); id = im2double(uint8([0 1 255]),'indexed'); if ~isa(i8,'uint8') || ~isa(i16,'uint16') || ~isa(id,'double'); error('indexed image conversion class failed'); end;",
    ] {
        execute_source(source).unwrap_or_else(|error| panic!("compiled form failed: {source}: {error}"));
    }

    let error = execute_source("rgb = hsv2rgb(uint8(zeros(1,1,3)));")
        .expect_err("hsv2rgb rejects integer HSV data");
    assert_eq!(error.identifier(), Some("RunMat:hsv2rgb:InvalidInput"));
}
