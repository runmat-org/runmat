use super::*;

#[test]
fn saveas_rejects_unsupported_extensions_and_explicit_format_wins() {
    let _guard = setup();
    let fig = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");
    let jpg_err = block_on(saveas_builtin(vec![
        Value::Num(fig),
        Value::String("plot.jpg".into()),
    ]))
    .expect_err("jpg is unsupported");
    assert!(jpg_err.message().contains("not available yet"));

    let request = parse_saveas_args(crate::call::FunctionArgumentView::from_slice(&[
        Value::Num(fig),
        Value::String("plot.fig".into()),
        Value::String("png".into()),
    ]))
    .expect("explicit format takes precedence");
    assert_eq!(request.format, SaveFormat::Png);
    assert_eq!(request.output_path, PathBuf::from("plot.fig"));
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn saveas_png_uses_shared_renderer_capability_before_file_effects() {
    let _guard = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    let filesystem = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/figures",
    ));
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("figure");

    let error = block_on(saveas_builtin(vec![
        Value::Num(figure),
        Value::from("/figures/not-created.png"),
    ]))
    .expect_err("PNG export requires the shared renderer capability");
    assert_eq!(error.identifier(), ERROR_RENDER.identifier);
    let mut source = std::error::Error::source(&error);
    let mut found = false;
    while let Some(error) = source {
        found |= error
            .downcast_ref::<crate::builtins::plotting::engine::PngExportUnavailable>()
            .is_some();
        source = error.source();
    }
    assert!(found, "typed renderer capability cause must be retained");
    assert!(filesystem
        .read_project_path("/figures/not-created.png")
        .is_err());
}

#[test]
fn openfig_reuse_returns_the_session_owned_open_copy() {
    let _guard = setup();
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("source figure");
    let path = temp_path("reuse");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![
        Value::Num(figure),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect("save reusable figure");
    reset_plot_state();

    let first = block_on(openfig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("open new copy");
    let reused = block_on(openfig_builtin(vec![
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("reuse".into()),
    ]))
    .expect("reuse open copy");
    assert_eq!(reused, first);
    assert_eq!(figure_handles().len(), 1);
    let _ = std::fs::remove_file(path);
}

#[test]
fn openfig_reuse_does_not_cross_runtime_sessions() {
    let _guard = setup();
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("source figure");
    let path = temp_path("reuse_sessions");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![
        Value::Num(figure),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect("save reusable figure");
    reset_plot_state();
    let first_runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let second_runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let filename = Value::String(path.to_string_lossy().into_owned().into());

    let first = block_on(execute_openfig(
        Some(first_runtime.clone()),
        Some(1),
        crate::builtins::common::test_support::builtin_arguments(vec![filename.clone()]),
    ))
    .expect("first session opens copy");
    let first_reused = block_on(execute_openfig(
        Some(first_runtime),
        Some(1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            filename.clone(),
            Value::String("reuse".into()),
        ]),
    ))
    .expect("first session reuses copy");
    let second = block_on(execute_openfig(
        Some(second_runtime),
        Some(1),
        crate::builtins::common::test_support::builtin_arguments(vec![
            filename,
            Value::String("reuse".into()),
        ]),
    ))
    .expect("second session opens independent copy");
    assert_eq!(first_reused, first);
    assert_ne!(second, first);
    assert_eq!(figure_handles().len(), 2);
    let _ = std::fs::remove_file(path);
}

#[test]
fn openfig_reuse_reimports_a_complete_bundle_after_one_copy_closes() {
    let _guard = setup();
    let first = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("first source figure");
    let second = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("second source figure");
    let path = temp_path("reuse_partial_close");
    let _ = std::fs::remove_file(&path);
    block_on(savefig_builtin(vec![
        Value::Tensor(tensor(&[first, second])),
        Value::String(path.to_string_lossy().into_owned().into()),
    ]))
    .expect("save figure bundle");
    reset_plot_state();

    let opened = block_on(openfig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect("open complete bundle");
    let Value::Tensor(opened) = opened else {
        panic!("expected a handle row");
    };
    let handles = opened
        .as_f64_slice()
        .expect("graphics handles use authoritative double storage")
        .to_vec();
    assert_eq!(handles.len(), 2);
    let survivor = FigureHandle::from(handles[1] as u32);
    close_figure(Some(FigureHandle::from(handles[0] as u32))).expect("close one imported copy");

    let reopened = block_on(openfig_builtin(vec![
        Value::String(path.to_string_lossy().into_owned().into()),
        Value::String("reuse".into()),
    ]))
    .expect("partial reuse entry must reimport the whole bundle");
    let Value::Tensor(reopened) = reopened else {
        panic!("expected a replacement handle row");
    };
    let reopened = reopened
        .as_f64_slice()
        .expect("graphics handles use authoritative double storage");
    assert_eq!(reopened.len(), 2);
    assert!(reopened
        .iter()
        .all(|handle| *handle != survivor.as_u32() as f64));
    assert_eq!(figure_handles().len(), 3);
    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "plot-core")]
#[test]
fn openfig_validates_the_complete_archive_before_importing_any_figure() {
    let _guard = setup();
    let figure = figure_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    )
    .expect("source figure");
    let source = clone_figure(FigureHandle::from(figure as u32)).expect("source snapshot");
    let valid = block_on(export_figure_snapshot(&source, SAVEFIG)).expect("valid scene payload");
    let archive = block_on(encode_figure_bundle(
        None,
        &[valid, b"not-a-scene".to_vec()],
        FigureArchiveProfile::Default,
        SAVEFIG,
    ))
    .expect("mixed archive");
    let path = temp_path("transactional_import");
    std::fs::write(&path, archive).expect("write mixed archive");
    reset_plot_state();
    let before = figure_handles();

    block_on(openfig_builtin(vec![Value::String(
        path.to_string_lossy().into_owned().into(),
    )]))
    .expect_err("invalid second scene rejects whole archive");
    assert_eq!(figure_handles(), before);
    let _ = std::fs::remove_file(path);
}

#[test]
fn save_rejects_excessive_selection_before_resolving_or_cloning_handles() {
    let missing = FigureHandle::from(u32::MAX);
    let selected = vec![missing; MAX_FIGURE_BUNDLE_COUNT + 1];
    let path = FigureFsPath::capture(Path::new("never-created.fig")).expect("capture path");
    let error = block_on(save_figures(
        None,
        &path,
        &selected,
        FigureArchiveProfile::Default,
        SAVEFIG,
    ))
    .expect_err("selection bound");
    assert!(error.message().contains("too many figures"));
    assert!(!error.message().contains("does not exist"));
}
