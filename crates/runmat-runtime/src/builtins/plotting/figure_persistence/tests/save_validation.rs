use super::*;

#[test]
fn savefig_rejects_non_fig_extension() {
    let _guard = setup();
    let err = block_on(savefig_builtin(vec![Value::String("plot.png".into())]))
        .expect_err("savefig requires fig");
    assert!(err.message().contains("expected a .fig filename"));
}

#[test]
fn savefig_without_arguments_selects_the_documented_default() {
    let _guard = setup();
    let request = parse_savefig_args(crate::call::FunctionArgumentView::from_slice(&[]), SAVEFIG)
        .expect("default savefig request");
    assert_eq!(request.output_path, PathBuf::from("Untitled.fig"));
    assert_eq!(request.profile, FigureArchiveProfile::Default);
    let FigureSaveSelection::Figures(figures) = request.selection else {
        panic!("savefig default must select figures");
    };
    assert_eq!(figures, vec![current_figure_handle()]);
    assert!(
        parse_savefig_args(crate::call::FunctionArgumentView::from_slice(&[]), HGSAVE).is_err()
    );
}

#[test]
fn savefig_accepts_only_figure_handles_while_hgsave_preserves_hierarchy_roots() {
    let _guard = setup();
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");
    let axes = crate::builtins::plotting::subplot::subplot_builtin(
        Value::Num(1.0),
        Value::Num(1.0),
        Value::Num(1.0),
    )
    .expect("axes");
    let path = Value::String("selection.fig".into());

    let savefig_error = parse_savefig_args(
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(axes), path.clone()]),
        SAVEFIG,
    )
    .expect_err("savefig requires Figure objects");
    assert!(savefig_error.message().contains("Figure handle"));

    let hgsave = parse_savefig_args(
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(axes), path]),
        HGSAVE,
    )
    .expect("hgsave accepts an axes hierarchy root");
    let FigureSaveSelection::Hierarchy(roots) = hgsave.selection else {
        panic!("hgsave must preserve typed hierarchy roots");
    };
    assert!(matches!(roots.as_slice(), [PlotHandle::Axes(_, 0)]));
    assert_eq!(
        parent_figure_from_value(&Value::Num(axes), HGSAVE)
            .unwrap()
            .as_u32(),
        figure as u32
    );
}

#[test]
fn hgsave_rejects_ancestor_descendant_arrays_before_file_admission() {
    let _guard = setup();
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");
    let axes = crate::builtins::plotting::subplot::subplot_builtin(
        Value::Num(1.0),
        Value::Num(1.0),
        Value::Num(1.0),
    )
    .expect("axes");
    let roots =
        Value::Tensor(Tensor::new(vec![figure, axes], vec![1, 2]).expect("hierarchy root array"));

    let error = parse_savefig_args(
        crate::call::FunctionArgumentView::from_slice(&[
            roots,
            Value::String("selection.fig".into()),
        ]),
        HGSAVE,
    )
    .expect_err("ancestor and descendant roots are incompatible");
    assert!(error.message().contains("ancestor and descendant"));
}

#[test]
fn archive_version_tokens_select_explicit_profiles() {
    for (token, expected) in [
        ("-v6", FigureArchiveProfile::V6),
        ("-v7", FigureArchiveProfile::V7),
        ("-v7.3", FigureArchiveProfile::V73),
        ("compact", FigureArchiveProfile::Compact),
    ] {
        let request = parse_savefig_args(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::String("profile.fig".into()),
                Value::String(token.into()),
            ]),
            SAVEFIG,
        )
        .expect("supported archive profile");
        assert_eq!(request.profile, expected, "{token}");
    }
    let error = parse_savefig_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::String("profile.fig".into()),
            Value::String("-v99".into()),
        ]),
        SAVEFIG,
    )
    .expect_err("unknown profile");
    assert!(error.message().contains("unsupported FIG version"));
}
