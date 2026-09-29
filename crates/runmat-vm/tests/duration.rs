#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_logical_duration_components_and_matrix_form_execute() {
    execute_source(
        r#"
        components = duration(logical([1 0]), 0, 0);
        assert(isduration(components));
        assert(isequal(hours(components), [1 0]));

        matrix = duration(logical([1 0 0; 0 1 0]));
        assert(isduration(matrix));
        assert(isequal(size(hours(matrix)), [2 1]));
        assert(abs(hours(matrix(1)) - 1) < 1e-12);
        assert(abs(minutes(matrix(2)) - 1) < 1e-12);
        "#,
    )
    .expect("compiled logical duration construction");
}

#[test]
fn compiled_duration_scalar_expansion_and_nd_shape_survive_unit_extraction() {
    execute_source(
        r#"
        input = reshape([1 2 3 4], 2, 1, 2);
        elapsed = duration(input, 15, 0);
        extracted = hours(elapsed);
        assert(isequal(size(elapsed), [2 1 2]));
        assert(numel(elapsed) == 4);
        assert(ndims(elapsed) == 3);
        assert(isequal(size(extracted), [2 1 2]));
        assert(all(abs(extracted(:) - (input(:) + 0.25)) < 1e-12));
        "#,
    )
    .expect("compiled N-D duration construction and extraction");
}

#[test]
fn compiled_object_backed_array_shape_queries_follow_payloads() {
    execute_source(
        r#"
        row = duration([1 2],15,0);
        assert(isequal(size(row), [1 2])); assert(isrow(row));
        assert(isvector(row)); assert(~iscolumn(row));
        assert(~isscalar(row)); assert(length(row) == 2);
        empty = hours(zeros(0,3));
        assert(isequal(size(empty), [0 3])); assert(isempty(empty)); assert(numel(empty) == 0);
        dates = datetime([2024 2025],4,9);
        assert(isequal(size(dates), [1 2])); assert(numel(dates) == 2);
        data = dlarray(reshape(1:6,2,3));
        assert(isequal(size(data), [2 3])); assert(numel(data) == 6);
        assert(ismatrix(data)); assert(~isscalar(data));
    "#,
    )
    .expect("compiled shape metadata reads array payloads, not scalar object wrappers");
}

#[test]
fn compiled_duration_format_mutation_member_read_and_indexed_extraction_work() {
    execute_source(
        r#"
        elapsed = hours([2 3]);
        elapsed.Format = 'hh:mm';
        stored_format = elapsed.Format;
        selected = elapsed(2);
        assert(strcmp(stored_format, 'hh:mm'));
        assert(strcmp(selected.Format, 'hh:mm'));
        selected.Format = 's';
        assert(strcmp(selected.Format, 's'));
        assert(strcmp(elapsed.Format, 'hh:mm'));
        assert(isduration(selected));
        assert(hours(selected) == 3);
        assert(isequal(size(hours(elapsed)), [1 2]));
        "#,
    )
    .expect("compiled duration member and indexing protocol");
}

#[test]
fn compiled_duration_comparison_uses_fixed_days_and_shifts_datetime() {
    execute_source(
        r#"
        elapsed = days([1 2]);
        flags = elapsed > 1;
        assert(isequal(flags, logical([0 1])));
        start = datetime(2024, 4, 9, 8, 0, 0);
        shifted = elapsed(2) + start;
        assert(day(shifted) == 11);
        assert(hour(shifted) == 8);
        "#,
    )
    .expect("compiled duration comparison and datetime shift");
}

#[test]
fn compiled_duration_short_form_reports_compatibility_identifier() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("elapsed = duration(1);")
        .expect_err("one-component form is a RunMat-only extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DurationShortComponentFormExtension")
    );
}

#[test]
fn compiled_duration_rejections_report_canonical_identifiers() {
    for (source, identifier) in [
        (
            "elapsed = duration('bad');",
            "RunMat:duration:InvalidArgument",
        ),
        (
            "elapsed = duration(1,2,3,'Format','bogus');",
            "RunMat:duration:InvalidFormat",
        ),
        (
            "elapsed = duration([1 2],[1 2 3],0);",
            "RunMat:duration:ShapeMismatch",
        ),
    ] {
        let error = execute_source(source).expect_err("invalid duration source");
        assert_eq!(error.identifier(), Some(identifier), "{source}: {error}");
    }
}
