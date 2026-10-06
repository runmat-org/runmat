use super::*;

#[test]
fn saveas_rejects_unsupported_or_mismatched_extensions() {
    let _guard = setup();
    let fig = figure_builtin(vec![]).expect("figure");
    let jpg_err = block_on(saveas_builtin(vec![
        Value::Num(fig),
        Value::String("plot.jpg".into()),
    ]))
    .expect_err("jpg is unsupported");
    assert!(jpg_err.message().contains("not available yet"));

    let mismatch_err = block_on(saveas_builtin(vec![
        Value::Num(fig),
        Value::String("plot.fig".into()),
        Value::String("png".into()),
    ]))
    .expect_err("extension/format mismatch");
    assert!(mismatch_err.message().contains("does not match"));
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
    let figure = figure_builtin(vec![]).expect("figure");

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
fn openfig_rejects_reuse_mode_until_tracking_exists() {
    let _guard = setup();
    let err = block_on(openfig_builtin(vec![
        Value::String("missing.fig".into()),
        Value::String("reuse".into()),
    ]))
    .expect_err("reuse is not implemented");
    assert!(err.message().contains("reuse mode"));
}

#[test]
fn decode_figure_bundle_rejects_hostile_counts_before_allocating() {
    let mut bytes = FIG_MAGIC.to_vec();
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    let err = decode_figure_bundle(&bytes, OPENFIG).expect_err("hostile count");
    assert!(err.message().contains("too many figures"));
}

#[test]
fn decode_figure_bundle_rejects_oversized_payload_length() {
    let mut bytes = FIG_MAGIC.to_vec();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(
        &(ReplayLimits::default().max_scene_payload_bytes as u64 + 1).to_le_bytes(),
    );
    let err = decode_figure_bundle(&bytes, OPENFIG).expect_err("oversized payload");
    assert!(err.message().contains("exceeds maximum scene payload size"));
}

#[test]
fn hgload_applies_property_overrides_and_returns_old_values() {
    let _guard = setup();
    let fig = figure_builtin(vec![
        Value::String("Name".into()),
        Value::String("Before".into()),
    ])
    .expect("figure");
    let path = temp_path("hgload");
    let _ = std::fs::remove_file(&path);
    block_on(hgsave_builtin(vec![
        Value::Num(fig),
        Value::String(path.to_string_lossy().into_owned()),
    ]))
    .expect("hgsave");
    clear_figure(None).expect("clear");
    let mut overrides = StructValue::new();
    overrides.insert("Name", Value::String("After".into()));
    overrides.insert("NotAProperty", Value::Num(1.0));
    let _outputs = crate::output_count::push_output_count(Some(2));
    let result = block_on(hgload_builtin(vec![
        Value::String(path.to_string_lossy().into_owned()),
        Value::Struct(overrides),
    ]))
    .expect("hgload");
    let Value::OutputList(outputs) = result else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 2);
    let Value::Cell(cell) = &outputs[1] else {
        panic!("expected old values cell");
    };
    let Value::Struct(old) = &cell.data[0] else {
        panic!("expected old values struct");
    };
    assert_eq!(
        old.fields.get("Name"),
        Some(&Value::String("Before".into()))
    );
    assert!(!old.fields.contains_key("NotAProperty"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn savefig_rejects_non_fig_extension() {
    let _guard = setup();
    let err = block_on(savefig_builtin(vec![Value::String("plot.png".into())]))
        .expect_err("savefig requires fig");
    assert!(err.message().contains("expected a .fig filename"));
}
