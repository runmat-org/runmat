#[test]
fn admitted_capture_free_nested_graph_uses_the_selected_source_path() {
    let source = crate::ExecutableSource::new(
        "fixture",
        "outer.m",
        "function h = outer()\nfunction y = inner(x)\ny = x + 1;\nend\nh = @inner;\nend\n",
    );
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let unit = futures::executor::block_on(session.compile_executable_unit(source, None))
        .expect("compiled nested unit");
    let nested = unit
        .functions()
        .functions
        .values()
        .find(|body| body.source_role == runmat_types::FunctionSourceRole::Nested)
        .expect("compiled nested body");
    assert_eq!(
        unit.functions().resolve_name(&nested.display_name),
        None,
        "nested bodies must remain unavailable to textual lookup"
    );
    assert_ne!(
        nested.function.0,
        usize::try_from(nested.portable_function.0).expect("portable fixture ordinal"),
        "fixture must distinguish compiler and portable coordinates"
    );
    let nested_origin = nested.binding_origin.expect("nested binding origin");
    let nested_envelope = unit
        .portable_envelope_for_function(nested.portable_function)
        .expect("origin-selected nested product");
    let imported_nested = crate::ExecutableUnit::from_portable_envelope(&nested_envelope)
        .expect("import origin-selected nested product");
    let imported_local = imported_nested
        .local_function_for_origin(nested_origin)
        .expect("nested origin lookup")
        .expect("imported nested body");
    assert_eq!(
        imported_nested
            .functions()
            .get(imported_local)
            .expect("imported nested registry body")
            .portable_function,
        nested.portable_function
    );
    assert!(
        imported_nested
            .function_analysis(nested.portable_function)
            .is_some(),
        "portable facts must remain keyed by the nested original identity"
    );
    assert_eq!(
        imported_nested
            .functions()
            .resolve_name(&nested.display_name),
        None,
        "cold import must not publish nested textual lookup"
    );
    let bytes = unit
        .portable_envelope()
        .expect("V2 envelope")
        .canonical_bytes()
        .expect("canonical V2 bytes");
    let admitted = runmat_vm::admit_executable_unit(&bytes).expect("admitted nested unit");
    let admitted_fingerprints = admitted
        .retained_callable_origin_fingerprints()
        .expect("admitted canonical definitions");
    let selected_bytes = std::sync::Arc::clone(admitted.canonical_v3());
    let mut records = unit.source_map().runtime_records();
    for record in &mut records {
        record.fullpath_name = Some(std::sync::Arc::from("/selected/outer.m"));
    }
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let bound = admitted
        .bind_runtime_sources_with_records(runtime, records)
        .expect("selected source issuance");
    assert_eq!(
        bound
            .registry()
            .selected_runtime_origin_fingerprints()
            .expect("exact issued definitions normalize to their portable origins"),
        admitted_fingerprints,
    );
    assert!(std::sync::Arc::ptr_eq(
        bound.canonical_v3().expect("selected V3 remains owned"),
        &selected_bytes,
    ));
    assert_eq!(bound.canonical_v3().unwrap().as_ref(), bytes.as_slice());
    let body = bound
        .registry()
        .functions
        .values()
        .find(|body| body.source_role == runmat_types::FunctionSourceRole::Nested)
        .expect("compiled nested body");
    let retained_source = body
        .selected_source_owner
        .as_ref()
        .expect("admitted definition retains its exact issued source table");
    assert!(retained_source.is_live());
    assert_eq!(
        retained_source.portable_map(),
        bound.selected_source_map().unwrap()
    );
    assert_eq!(
        retained_source.selected_sources().count(),
        bound.selected_source_map().unwrap().entries.len()
    );
    assert!(body.capture_slots.is_empty());
    assert!(body.inherited_storage_slots.is_empty());
    assert_ne!(body.source_file.as_deref(), Some("/selected/outer.m"));
    let origin = body.binding_origin.expect("admitted nested origin");
    let (selected_origin, selected_fingerprint) = bound
        .registry()
        .selected_runtime_body_fingerprint(body.function)
        .expect("one exact selected body fingerprint");
    assert_eq!(selected_origin, origin);
    assert_eq!(
        admitted_fingerprints.get(&origin),
        Some(&selected_fingerprint),
        "the selected body must match its admitted V3 definition"
    );
    let mut changed = bound.registry().clone();
    changed
        .functions
        .get_mut(&body.function)
        .expect("selected body")
        .display_name
        .push_str("_changed_after_admission");
    let (_, changed_fingerprint) = changed
        .selected_runtime_body_fingerprint(body.function)
        .expect("changed body still has a valid source owner");
    assert_ne!(
        changed_fingerprint, selected_fingerprint,
        "body changes must invalidate the exact selected-product comparison"
    );
    // Exercise the fingerprint projection independently of source issuance:
    // a second local body may carry the same portable origin, and the helper
    // must not collapse its FunctionId before exact lease/owner selection.
    let mut duplicate_origin = bound.registry().clone();
    let duplicate_id = runmat_hir::FunctionId(
        duplicate_origin
            .functions
            .keys()
            .map(|function| function.0)
            .max()
            .unwrap()
            + 1,
    );
    let mut same_origin = body.clone();
    same_origin.function = duplicate_id;
    duplicate_origin.functions.insert(duplicate_id, same_origin);
    let by_body = duplicate_origin
        .selected_runtime_body_fingerprints(&[body.function, duplicate_id])
        .expect("equal portable origins remain separately selected by body");
    assert_eq!(by_body.len(), 2);
    assert_eq!(by_body[&body.function].0, by_body[&duplicate_id].0);
    let selected = bound
        .registry()
        .execution_source_fact_for_origin_in(bound.runtime(), origin)
        .expect("selected nested source fact");
    assert_eq!(selected.source_file.as_deref(), Some("/selected/outer.m"));
    let closure = runmat_value::Value::Closure(runmat_value::Closure::new(
        selected.name,
        Some(runmat_value::BoundFunctionBinding::portable(
            body.function.0,
            origin,
        )),
        selected.role,
        selected.source_file,
        Vec::new(),
        None,
    ));
    let graph_codec = bound
        .graph_codec_admission()
        .expect("exact graph authority");
    let graph = graph_codec
        .snapshot(
            std::slice::from_ref(&closure),
            runmat_execution::value::ValueLimits::default(),
        )
        .expect("snapshot actual selected nested source");
    assert!(graph
        .nodes
        .iter()
        .all(|node| { !matches!(node, runmat_execution::value::GraphNode::Activation(_)) }));
    let rebuilt = graph_codec
        .reconstruct(&graph, runmat_execution::value::ValueLimits::default())
        .expect("reconstruct admitted nested closure");
    let runmat_value::Value::Closure(rebuilt) = &rebuilt.values()[0] else {
        panic!("nested closure")
    };
    assert_eq!(rebuilt.source_file.as_deref(), Some("/selected/outer.m"));
    assert!(rebuilt.lexical_environment.is_none());
}

#[test]
fn admitted_canonical_storage_nested_graph_needs_no_activation() {
    let source = crate::ExecutableSource::new(
        "fixture",
        "storage.m",
        "function h = outer()\nglobal shared\nfunction y = inner()\ny = shared;\nend\nh = @inner;\nend\n",
    );
    let mut session = crate::RunMatSession::with_options(false, false).expect("session");
    let unit = futures::executor::block_on(session.compile_executable_unit(source, None))
        .expect("compiled nested storage unit");
    let bytes = unit
        .portable_envelope_for(Some("outer"))
        .expect("V2 envelope")
        .canonical_bytes()
        .expect("canonical V2 bytes");
    let admitted = runmat_vm::admit_executable_unit(&bytes).expect("admitted storage unit");
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let bound = admitted
        .bind_runtime_sources_with_records(runtime, unit.source_map().runtime_records())
        .expect("selected source issuance");
    let body = bound
        .registry()
        .functions
        .values()
        .find(|body| body.source_role == runmat_types::FunctionSourceRole::Nested)
        .expect("compiled nested body");
    assert!(!body.inherited_storage_slots.is_empty());
    assert!(body.lexical_binding_slots.is_empty());
    let origin = body.binding_origin.expect("admitted nested origin");
    let selected = bound
        .registry()
        .execution_source_fact_for_origin_in(bound.runtime(), origin)
        .expect("selected nested source fact");
    let closure = runmat_value::Value::Closure(runmat_value::Closure::new(
        selected.name,
        Some(runmat_value::BoundFunctionBinding::portable(
            body.function.0,
            origin,
        )),
        selected.role,
        selected.source_file,
        Vec::new(),
        None,
    ));
    let codec = bound
        .graph_codec_admission()
        .expect("exact graph authority");
    let graph = codec
        .snapshot(
            std::slice::from_ref(&closure),
            runmat_execution::value::ValueLimits::default(),
        )
        .expect("storage-only nested snapshot");
    assert!(graph
        .nodes
        .iter()
        .all(|node| !matches!(node, runmat_execution::value::GraphNode::Activation(_))));
    let rebuilt = codec
        .reconstruct(&graph, runmat_execution::value::ValueLimits::default())
        .expect("storage-only nested reconstruction");
    let runmat_value::Value::Closure(rebuilt) = &rebuilt.values()[0] else {
        panic!("nested closure")
    };
    assert!(rebuilt.lexical_environment.is_none());
    assert!(rebuilt.captures.is_empty());
}
