#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_text_only_apis_reject_every_integer_class() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        for call in [
            format!("normalizeWords({constructor}(1));"),
            format!("readWordEmbedding({constructor}(1));"),
            format!("regexp({constructor}(1),'x');"),
            format!("regexpi({constructor}(1),'x');"),
            format!("replace({constructor}(1),'a','b');"),
            format!("rethrow({constructor}(1));"),
            format!("splitlines({constructor}(1));"),
            format!("strip({constructor}(1));"),
            format!("str2double({constructor}(1));"),
            format!("str2func({constructor}(1));"),
            format!("str2num({constructor}(1));"),
            format!("strcat({constructor}(1),'x');"),
            format!("strjoin({constructor}(1),',');"),
            format!("strjust({constructor}(1));"),
            format!("strlength({constructor}(1));"),
            format!("strrep({constructor}(1),'a','b');"),
            format!("strsplit({constructor}(1));"),
            format!("strtok({constructor}(1));"),
            format!("strtrim({constructor}(1));"),
        ] {
            execute_source(&call).expect_err("text-only API must reject typed integer input");
        }
    }
}

#[test]
fn compiled_string_conversion_and_text_comparison_cover_all_integer_classes() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        execute_source(&format!(
            "x={constructor}(3); s=string(x); a=strings({constructor}(2),{constructor}(3)); p=strncmp('abc','abd',{constructor}(2)); q=strncmpi('ABC','abd',{constructor}(2)); if ~strcmp(s,'3') || ~isequal(size(a),[2 3]) || ~p || ~q; error('integer string semantics mismatch'); end;"
        ))
        .expect("documented integer string forms");
    }
}

#[test]
fn compiled_string_conversion_preserves_wide_integer_text_and_zero_clamps_counts() {
    execute_source(
        "x=uint64(9007199254740992)+uint64(1); if ~strcmp(string(x),'9007199254740993'); error('wide integer rounded'); end; a=strings(int8(-2),uint8(4)); if ~isequal(size(a),[0 4]); error('negative size did not clamp'); end; if ~strncmp('x','y',int8(-1)) || ~strncmpi('x','y',int16(-1)); error('negative prefix count did not clamp'); end;",
    )
    .expect("wide conversion and negative structural controls");
}

#[test]
fn compiled_text_comparisons_return_false_for_unsupported_integer_data() {
    execute_source(
        "x=uint64(9007199254740992)+uint64(1); if strcmp(x,'9007199254740993') || strcmpi(x,'9007199254740993') || strncmp(x,'9',uint8(1)) || strncmpi(x,'9',uint8(1)); error('unsupported numeric comparison did not return false'); end;",
    )
    .expect("unsupported numeric comparison semantics");
}

#[test]
fn compiled_string_extensions_are_gated_without_narrowing_documented_integer_forms() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source(
        "s=string(uint64(3)); a=strings(uint8(2),uint16(3)); if ~strcmp(s,'3') || ~isequal(size(a),[2 3]); error('documented integer forms narrowed'); end;",
    )
    .expect("documented forms in compatibility mode");

    for (source, identifier) in [
        (
            "string('value %d',uint64(3));",
            "RunMat:compatibility:StringFormatSpecExtension",
        ),
        (
            "string('abc','UTF-8');",
            "RunMat:compatibility:StringEncodingExtension",
        ),
        (
            "strings(2,'missing');",
            "RunMat:compatibility:StringsFillModeExtension",
        ),
        (
            "strings('like',uint64([1 2]));",
            "RunMat:compatibility:StringsLikePrototypeExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("RunMat extension must be gated");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_text_only_apis_retain_ordinary_text_behavior() {
    execute_source(
        "a=normalizeWords('running'); b=regexp('abc','b'); c=regexpi('AbC','a'); d=replace('abc','b','x'); e=splitlines(sprintf('a\\nb')); f=strip('  ok  '); if ~strcmp(a,'run') || b~=2 || c~=1 || ~strcmp(d,'axc') || ~strcmp(f,'ok'); error('text behavior mismatch'); end;",
    )
    .expect("compiled text behavior");
}
