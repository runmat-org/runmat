#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_numeric_extensions_have_stable_strict_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "g = gradient(uint16([1 4 9]));",
            "RunMat:compatibility:GradientIntegerDataExtension",
        ),
        (
            "rgb = gray2rgb([0 1]);",
            "RunMat:compatibility:Gray2rgbCallExtension",
        ),
        (
            "f = griddedInterpolant(uint16([1 4 9]));",
            "RunMat:compatibility:GriddedInterpolantIntegerValuesExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_grp2idx_preserves_wide_integer_group_labels() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "base = uint64(9007199254740992); s = [base + uint64(1); base; base + uint64(1)]; [g,gN,gL] = grp2idx(s); if ~isa(gL,'uint64') || gL(1) ~= base || gL(2) ~= base + uint64(1) || g(1) ~= 2 || g(2) ~= 1 || g(3) ~= 2; error('grp2idx exact integer contract failed'); end; if ~iscell(gN); error('grp2idx names must be cellstr'); end;",
    )
    .expect("compiled exact grp2idx contract");
}

#[test]
fn compiled_gt_and_hamming_expose_documented_integer_forms() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "base = uint64(9007199254740992); tf = gt(base + uint64(1), base); if ~tf; error('wide gt failed'); end; w = hamming(uint16(4), 'periodic', 'single'); if ~isa(w,'single') || ~isequal(size(w), [4 1]); error('hamming class or shape failed'); end;",
    )
    .expect("compiled integer comparison and window contract");
}
