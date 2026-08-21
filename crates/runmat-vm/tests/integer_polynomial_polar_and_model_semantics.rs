#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_polynomial_coordinate_and_curve_extensions_accept_every_integer_class() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "[x,y]=pol2cart({constructor}([0 1]),{constructor}([1 1])); if numel(x)~=2 || numel(y)~=2; error('pol2cart integer input'); end; d=polyder({constructor}([1 2 3])); if numel(d)~=2; error('polyder integer input'); end; p=polyfit({constructor}([0 1 2]),{constructor}([1 3 5]),{constructor}(1)); if numel(p)~=2; error('polyfit integer input'); end; v=polyval({constructor}([1 2]),{constructor}([0 1])); if numel(v)~=2; error('polyval integer input'); end; q=pow2({constructor}([0 1])); if numel(q)~=2 || q(1)~=1 || q(2)~=2; error('pow2 integer input'); end; pp=pchip([0 1],[0 1]); pq=ppval(pp,{constructor}([0 1])); if numel(pq)~=2; error('ppval integer query'); end; [cx,cy]=perfcurve({constructor}([0 1]),{constructor}([0 1]),{constructor}(1)); if isempty(cx) || isempty(cy); error('perfcurve integer input'); end;"
        );
        execute_source(&source).unwrap_or_else(|error| panic!("{constructor}: {error:?}"));
    }
}

#[test]
fn compiled_perfcurve_keeps_adjacent_wide_uint64_labels_distinct() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "base=bitshift(uint64(1),53); labels=uint64([base base+uint64(1)]); [x,y]=perfcurve(labels,[0 1],base+uint64(1)); if isempty(x) || isempty(y); error('wide perfcurve labels'); end;",
    )
    .expect("compiled exact perfcurve labels");
}

#[test]
fn compiled_strict_mode_rejects_runmat_only_polynomial_input() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("y=pow2(uint8([0 1]));")
        .expect_err("strict mode must reject integer pow2 extension");
    assert!(error.to_string().contains("compatibility"), "{error:?}");
}
