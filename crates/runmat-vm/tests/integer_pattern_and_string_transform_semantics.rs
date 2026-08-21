#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_pattern_counts_and_padding_preserve_exact_integer_controls() {
    execute_source(
        "letters = lettersPattern(uint64(2)); if ~matches(\"Ab\",letters) || matches(\"A1\",letters); error('lettersPattern failed'); end; wildcard = wildcardPattern(uint16(0)); if ~matches(\"\",wildcard) || matches(\"x\",wildcard); error('wildcardPattern failed'); end; padded = pad(\"x\",uint32(3)); if padded ~= \"x  \"; error('pad failed'); end;",
    )
    .expect("compiled documented pattern counts and padding");
}

#[test]
fn compiled_string_control_extensions_are_available_only_in_runmat_mode() {
    {
        let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        execute_source(
            "if ~startsWith(\"RunMat\",\"run\",\"IgnoreCase\",uint8(1)); error('startsWith failed'); end; k = strfind(\"mission\",\"s\",\"ForceCellOutput\",uint16(1)); if ~iscell(k) || ~isequal(k{1},[3 4]); error('strfind failed'); end; names = [\"Mary Butler\";\"Diana Lee\";\"James King\"]; pieces = split(names,\" \",uint32(1)); if ~isequal(size(pieces),[2 3]) || pieces(1,2) ~= \"Diana\" || pieces(2,3) ~= \"King\"; error('split failed'); end;",
        )
        .expect("compiled RunMat string-control extensions");
    }

    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let starts_with_error =
        execute_source("tf = startsWith(\"RunMat\",\"run\",\"IgnoreCase\",uint8(1));")
            .expect_err("typed startsWith extension must be gated");
    assert_eq!(
        starts_with_error.identifier(),
        Some("RunMat:compatibility:StartsWithNumericIgnoreCaseExtension")
    );
    let strfind_error =
        execute_source("k = strfind(\"mission\",\"s\",\"ForceCellOutput\",uint8(1));")
            .expect_err("typed strfind extension must be gated");
    assert_eq!(
        strfind_error.identifier(),
        Some("RunMat:compatibility:StrfindTypedForceCellOutputExtension")
    );
    let split_error = execute_source("s = split(\"Mary Butler\",\" \",uint8(1));")
        .expect_err("typed split dimension extension must be gated");
    assert_eq!(
        split_error.identifier(),
        Some("RunMat:compatibility:SplitTypedDimensionExtension")
    );
}

#[test]
fn compiled_matlab_mode_keeps_documented_logical_and_double_forms() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "if ~startsWith(\"RunMat\",\"run\",\"IgnoreCase\",true); error('logical startsWith failed'); end; k = strfind(\"mission\",\"s\",\"ForceCellOutput\",1); if ~iscell(k) || ~isequal(k{1},[3 4]); error('double strfind failed'); end; parts = split(\"Mary Butler\",\" \",1); if ~isequal(size(parts),[2 1]); error('double split failed'); end;",
    )
    .expect("compiled documented MATLAB-compatible forms");
}
