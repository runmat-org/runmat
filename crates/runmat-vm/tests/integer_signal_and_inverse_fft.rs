#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_documented_integer_signal_and_inverse_fft_forms_execute() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "w = hann(uint16(4), 'single'); if ~isa(w,'single') || ~isequal(size(w), [4 1]); error('hann integer length contract failed'); end; y = ifft(uint16([1 0])); if ~isa(y,'double') || ~isequal(size(y), [1 2]); error('ifft integer data contract failed'); end;",
    )
    .expect("compiled documented signal and inverse FFT integer forms");
}

#[test]
fn compiled_signal_and_inverse_fft_extensions_have_stable_strict_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "w = hann(true);",
            "RunMat:compatibility:HannLogicalLengthExtension",
        ),
        (
            "y = heaviside(uint8(1));",
            "RunMat:compatibility:HeavisideIntegerInputExtension",
        ),
        (
            "z = hilbert(uint8([1 0]));",
            "RunMat:compatibility:HilbertIntegerDataExtension",
        ),
        (
            "r = hypot(uint8(3), 4);",
            "RunMat:compatibility:HypotIntegerInputExtension",
        ),
        (
            "y = ifft(uint64([1 0]));",
            "RunMat:compatibility:IfftWideIntegerDataExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_runmat_signal_extensions_remain_available() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "if heaviside(uint8(0)) ~= 0.5; error('heaviside extension failed'); end; if hypot(uint8(3), uint8(4)) ~= 5; error('hypot extension failed'); end; z = hilbert(uint8([1 0 1 0])); if ~isequal(size(z), [1 4]); error('hilbert extension shape failed'); end;",
    )
    .expect("compiled RunMat signal extensions");
}
