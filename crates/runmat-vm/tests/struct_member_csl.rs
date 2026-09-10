#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::Value;
use test_helpers::execute_source;

fn has_number(values: &[Value], expected: f64) -> bool {
    values.iter().any(|value| value == &Value::Num(expected))
}

fn structure_array(values: &[Value]) -> &runmat_value::StructArray {
    values
        .iter()
        .find_map(|value| match value {
            Value::StructArray(array) => Some(array),
            _ => None,
        })
        .expect("expected a structure array")
}

#[test]
fn plain_member_use_requires_one_value() {
    let error = execute_source("s = struct('value', {1, 2}); x = s.value;")
        .expect_err("plain use of a nonscalar comma-separated list must fail");
    assert!(error.to_string().contains("requires one value"));
}

#[test]
fn bracketed_member_read_selects_prefix_and_reports_shortage() {
    let values = execute_source(
        "s = struct('value', {1, 2, 3}); [first, second] = s.value; out1 = first; out2 = second;",
    )
    .unwrap();
    assert!(has_number(&values, 1.0));
    assert!(has_number(&values, 2.0));

    let error = execute_source("s = struct('value', {1, 2}); [first, second, third] = s.value;")
        .expect_err("requesting more member values than exist must fail");
    assert!(error.to_string().contains("were requested"));
}

#[test]
fn static_and_dynamic_member_arguments_expand_in_order() {
    execute_source(
        "s = struct('value', {1, 2, 3}); a = horzcat(s.value); name = 'value'; b = horzcat(s.(name)); assert(isequal(a, [1,2,3])); assert(isequal(b, [1,2,3]));",
    )
    .unwrap();
}

#[test]
fn chained_member_use_requires_a_singleton_sequence() {
    let error = execute_source(
        "inner = struct('leaf', {1, 2}); s = struct('value', {inner, inner}); x = s.value.leaf;",
    )
    .expect_err("chained access through a nonscalar comma-separated list must fail");
    assert!(error.to_string().contains("requires one value"));
}

#[test]
fn bracketed_member_destination_distributes_exact_outputs() {
    let values = execute_source(
        "s = struct('value', {1, 2}); [s.value] = deal(7, 8); name = 'value'; [s.(name)] = deal(9, 10); out = s;",
    )
    .unwrap();
    let array = structure_array(&values);
    assert_eq!(array.field_values("value").unwrap()[0], Value::Num(9.0));
    assert_eq!(array.field_values("value").unwrap()[1], Value::Num(10.0));
}

#[test]
fn mixed_fixed_and_member_destinations_preserve_source_order() {
    let values = execute_source(
        "s = struct('value', {1, 2}); [first, s.value] = deal(9, 7, 8); out1 = first; out2 = s;",
    )
    .unwrap();
    assert!(has_number(&values, 9.0));
    let array = structure_array(&values);
    assert_eq!(
        array.field_values("value").unwrap(),
        &[Value::Num(7.0), Value::Num(8.0)]
    );
}

#[test]
fn multiple_member_destinations_on_one_root_accumulate_left_to_right() {
    let values = execute_source(
        "s = struct('a', {1, 2}, 'b', {3, 4}); [s.a, s.b] = deal(10, 20, 30, 40); out = s;",
    )
    .unwrap();
    let array = structure_array(&values);
    assert_eq!(
        array.field_values("a").unwrap(),
        &[Value::Num(10.0), Value::Num(20.0)]
    );
    assert_eq!(
        array.field_values("b").unwrap(),
        &[Value::Num(30.0), Value::Num(40.0)]
    );
}

#[test]
fn mixed_indexed_member_destination_uses_prepared_selectors() {
    let values = execute_source(
        "s = struct('value', {1, 2, 3}); indices = [1, 3]; [first, s(indices).value] = deal(5, 7, 9); out1 = first; out2 = s;",
    )
    .unwrap();
    assert!(has_number(&values, 5.0));
    let array = structure_array(&values);
    assert_eq!(
        array.field_values("value").unwrap(),
        &[Value::Num(7.0), Value::Num(2.0), Value::Num(9.0)]
    );
}

#[test]
fn mixed_cell_contents_destination_uses_canonical_brace_cardinality() {
    execute_source(
        "c = {1, 2}; [first, c{:}] = deal(5, 7, 9); assert(first == 5); assert(isequal(c, {7, 9}));",
    )
    .unwrap();
}

#[test]
fn member_destination_cardinality_is_exact() {
    let error = execute_source("s = struct('value', {1, 2}); [s.value] = 7;")
        .expect_err("one value cannot fill a two-element sequence destination");
    assert!(
        error
            .to_string()
            .contains("exactly one value per destination"),
        "{error}"
    );
}

#[test]
fn indexed_member_destination_updates_its_root() {
    let values =
        execute_source("s = struct('value', {1, 2, 3}); [s([1, 3]).value] = deal(7, 9); out = s;")
            .unwrap();
    let array = structure_array(&values);
    assert_eq!(
        array.field_values("value").unwrap(),
        &[Value::Num(7.0), Value::Num(2.0), Value::Num(9.0)]
    );
}

#[test]
fn sequence_destination_is_evaluated_once_before_rhs() {
    execute_source(
        "global trace; trace = 0; function idx = select_target(); global trace; trace = trace * 10 + 1; idx = [1, 2]; end; function name = select_member(); global trace; trace = trace * 10 + 1; name = 'value'; end; function [first, second] = make_values(); global trace; trace = trace * 10 + 2; first = 7; second = 8; end; s = struct('value', {1, 2}); [s(select_target()).value] = make_values(); assert(trace == 12); trace = 0; [s.(select_member())] = make_values(); assert(trace == 12); assert(isequal(horzcat(s.value), [7, 8]));",
    )
    .unwrap();
}

#[test]
fn failed_sequence_producer_and_consumer_do_not_leak_register_state() {
    execute_source(
        "s = struct('value', {1, 2}); short = struct('value', 3); try; [s.value] = short.value; catch; end; [s.value] = deal(4, 5); try; [s.value] = 6; catch; end; [s.value] = deal(7, 8); assert(isequal(horzcat(s.value), [7, 8]));",
    )
    .unwrap();
}

#[test]
fn aggregate_elements_expand_sequences_with_realized_row_validation() {
    execute_source(
        "s = struct('value', {1, 2}); a = [s.value]; b = {s.value}; assert(isequal(a, [1, 2])); assert(isequal(b, {1, 2})); c = [1, 2; [3, 4]]; assert(isequal(c, [1, 2; 3, 4])); d = {s.value; 3, 4}; assert(isequal(d, {1, 2; 3, 4}));",
    )
    .unwrap();

    let error = execute_source("s = struct('value', {1, 2}); x = {s.value; 3};")
        .expect_err("realized cell rows with different widths must fail");
    assert!(error.to_string().contains("different widths"), "{error}");
}

#[test]
fn dynamic_and_brace_sequences_expand_inside_aggregates() {
    execute_source(
        "s = struct('value', {1, 2}); name = 'value'; c = {s.(name)}; source = {3, 4}; d = {source{:}}; assert(isequal(c, {1, 2})); assert(isequal(d, {3, 4}));",
    )
    .unwrap();
}

#[test]
fn nested_member_and_brace_destination_paths_write_back_to_the_root() {
    execute_source(
        "inner = struct('value', {1, 2}); outer = struct('inner', inner); [outer.inner.value] = deal(7, 8); assert(isequal(horzcat(outer.inner.value), [7, 8])); c = {inner}; [c{1}.value] = deal(9, 10); assert(isequal(horzcat(c{1}.value), [9, 10]));",
    )
    .unwrap();
}

#[test]
fn nested_dynamic_destination_path_resolves_each_name_once() {
    execute_source(
        "global trace; trace = 0; function name = next_name(value); global trace; trace = trace * 10 + value; if value == 1; name = 'inner'; else; name = 'value'; end; end; inner = struct('value', {1, 2}); outer = struct('inner', inner); [outer.(next_name(1)).(next_name(2))] = deal(7, 8); assert(trace == 12); assert(isequal(horzcat(outer.inner.value), [7, 8]));",
    )
    .unwrap();
}

#[test]
fn indexed_sequence_targets_preserve_repetition_logic_and_nd_order() {
    execute_source(
        "s = struct('value', {1, 2, 3}); [s([3, 1, 3]).value] = deal(7, 8, 9); assert(isequal(horzcat(s.value), [8, 2, 9])); mask = logical([1, 0, 1]); [s(mask).value] = deal(4, 5); assert(isequal(horzcat(s.value), [4, 2, 5])); t = reshape(struct('value', {1, 2, 3, 4}), [2, 1, 2]); [t(:,1,2).value] = deal(11, 12); assert(isequal(horzcat(t.value), [1, 2, 11, 12]));",
    )
    .unwrap();
}

#[test]
fn indexed_sequence_targets_resolve_end_and_reject_undefined_colon_growth() {
    execute_source(
        "s = struct('value', {1, 2, 3}); [s([1, end]).value] = deal(7, 9); assert(isequal(horzcat(s.value), [7, 2, 9]));",
    )
    .unwrap();

    let error = execute_source("clear s; [s(:).value] = deal(1, 2);")
        .expect_err("colon cannot define the shape of an undefined aggregate destination");
    let diagnostic = error.to_string().to_ascii_lowercase();
    assert!(
        diagnostic.contains("undefined") || diagnostic.contains("colon"),
        "unexpected undefined-colon diagnostic: {error}"
    );
}

#[test]
fn contextual_end_is_shared_by_member_and_brace_selectors() {
    execute_source(
        "s = reshape(struct('value', {1, 2, 3, 4}), [2, 2]); [s([1, end], 2).value] = deal(7, 9); assert(isequal(horzcat(s.value), [1, 2, 7, 9])); c = {10, 20, 30}; out = [c{[1, end]}]; assert(isequal(out, [10, 30])); [c{[1, end]}] = deal(40, 50); assert(isequal(c, {40, 20, 50}));",
    )
    .unwrap();
}

#[test]
fn contextual_end_consumes_effectful_object_prefix_once() {
    execute_source(
        "global prefix_calls; prefix_calls = 0; classdef PrefixOnce methods function out = subsref(obj, S); global prefix_calls; prefix_calls = prefix_calls + 1; out = {1, 2, 3}; end end end; o = PrefixOnce(); out = o.child{end}; assert(out == 3); assert(prefix_calls == 1);",
    )
    .unwrap();
}

#[test]
fn contextual_selector_effects_run_once_before_the_rhs() {
    execute_source(
        "global trace; trace = 0; function out = choose(last); global trace; trace = trace * 10 + 1; out = [1, last]; end; function [a,b] = values(); global trace; trace = trace * 10 + 2; a = 7; b = 9; end; s = struct('value', {1, 2, 3}); [s(choose(end)).value] = values(); assert(trace == 12); assert(isequal(horzcat(s.value), [7, 2, 9]));",
    )
    .unwrap();
}

#[test]
fn duplicate_same_root_sequence_targets_commit_in_source_order() {
    execute_source(
        "s = struct('value', {1, 2}); [s.value, s.value] = deal(3, 4, 5, 6); assert(isequal(horzcat(s.value), [5, 6]));",
    )
    .unwrap();
}
