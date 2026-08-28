#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::{compile_source, execute_source};

fn execute_source_with_catalog(
    source: &str,
) -> Result<Vec<runmat_value::Value>, Box<runmat_runtime::RuntimeError>> {
    let bytecode = compile_source(source).map_err(Box::new)?;
    let source_id = bytecode.source_id.unwrap_or(runmat_hir::SourceId(0));
    let _catalog = runmat_runtime::source_context::replace_source_catalog(vec![(
        source_id,
        "/tmp/integer_index_text_and_control_semantics.m".to_string(),
        source.to_string(),
    )]);
    test_helpers::interpret(&bytecode).map_err(Box::new)
}

#[test]
fn compiled_index_shape_and_text_integer_semantics_are_exact() {
    execute_source_with_catalog(
        "[r,c] = ind2sub(uint64([2 3]),uint64([5 7])); if ~isa(r,'double') || ~isa(c,'double') || ~isequal(r,[1 1]) || ~isequal(c,[3 4]); error('ind2sub failed'); end; a = horzcat(uint64(9007199254740993),uint64(9007199254740994)); if ~isa(a,'uint64') || a(1) ~= uint64(9007199254740993); error('horzcat failed'); end; s = insertAfter(\"abcd\",uint16(2),\"X\"); if s ~= \"abXcd\"; error('insertAfter failed'); end; t = insertBefore(\"abcd\",uint64(3),\"Y\"); if t ~= \"abYcd\"; error('insertBefore failed'); end;",
    )
    .expect("compiled index, shape, and text integer semantics");
}

#[test]
fn compiled_runmat_extensions_and_strict_gates_remain_distinct() {
    {
        let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
        execute_source(
            "w = ind2word(wordEncoding([\"one\" \"two\"]),uint16([1 2])); if ~isequal(w,[\"one\" \"two\"]); error('ind2word failed'); end;",
        )
        .expect("compiled RunMat integer extension");
    }
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("w = ind2word(wordEncoding([\"one\" \"two\"]),uint16(1));")
        .expect_err("typed ind2word extension must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Ind2wordTypedIntegerExtension")
    );
}

#[test]
fn compiled_inputname_bounds_and_comma_list_propagation_match_the_public_contract() {
    let values = execute_source_with_catalog(
        "alpha = 10; beta = 20; object = struct('field', 30); [a,b,c] = probe(alpha,object.field,beta); function [a,b,c] = probe(x,y,z); a=inputname(1); b=inputname(2); c=inputname(3); end;",
    )
    .expect("compiled inputname comma-list propagation");
    assert!(values.iter().any(|value| {
        matches!(value, runmat_value::Value::CharArray(chars) if chars.data.iter().collect::<String>() == "alpha")
    }));
    assert!(
        values
            .iter()
            .filter(|value| matches!(value, runmat_value::Value::CharArray(chars) if chars.data.is_empty()))
            .count()
            >= 2,
        "indexed and subsequent arguments must be unnamed: {values:?}"
    );

    let error = execute_source_with_catalog(
        "alpha = 10; out = exceeds(alpha); function out = exceeds(x); out=inputname(2); end;",
    )
    .expect_err("inputname beyond nargin must fail");
    assert_eq!(
        error.identifier(),
        Some("RunMat:InputnameArgumentExceedsInputs")
    );
}
