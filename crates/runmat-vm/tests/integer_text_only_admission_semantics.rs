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
        ] {
            execute_source(&call).expect_err("text-only API must reject typed integer input");
        }
    }
}

#[test]
fn compiled_text_only_apis_retain_ordinary_text_behavior() {
    execute_source(
        "a=normalizeWords('running'); b=regexp('abc','b'); c=regexpi('AbC','a'); d=replace('abc','b','x'); e=splitlines(sprintf('a\\nb')); f=strip('  ok  '); if ~strcmp(a,'run') || b~=2 || c~=1 || ~strcmp(d,'axc') || ~strcmp(f,'ok'); error('text behavior mismatch'); end;",
    )
    .expect("compiled text behavior");
}
