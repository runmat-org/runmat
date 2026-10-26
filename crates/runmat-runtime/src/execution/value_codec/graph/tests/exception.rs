use super::*;
use std::sync::Arc;

use runmat_execution::{ProgramEnvironment, ProgramRevision};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn retired_local_diagnostic_does_not_admit_a_reused_numeric_source() {
    runmat_gc::gc_test_context(|| {
        let base = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let map = source_map();
        let old = crate::source_context::bind_portable_source_map_v2(base.clone(), &map)
            .expect("old issued source");
        let old_owner = crate::source_context::SelectedProgramSourceOwner::capture(&old, &map)
            .expect("old source owner");
        let old_token = old_owner
            .call_frame_source_owner_for(ProgramSourceId(0))
            .expect("old frame token");
        drop(old_owner);
        drop(old);

        let current = crate::source_context::bind_portable_source_map_v2(base, &map)
            .expect("new issued source");
        let current_owner =
            crate::source_context::SelectedProgramSourceOwner::capture(&current, &map)
                .expect("new source owner");
        let current_token = current_owner
            .call_frame_source_owner_for(ProgramSourceId(0))
            .expect("new frame token");
        let current_source = current
            .program_source_id(ProgramSourceId(0))
            .expect("new local source");
        let revision = ProgramRevision::new(
            Digest::sha256(b"graph-diagnostic-program"),
            Digest::sha256(b"graph-diagnostic-source"),
            ProgramEnvironment::new(
                runmat_execution::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
                runmat_execution::schema::PROGRAM_COMPILER_SCHEMA_V2,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                "matlab",
            )
            .expect("environment"),
        )
        .expect("revision");
        let selected = GraphSourceUnitFacts::capture_owner(current_owner, &revision, Arc::new(map))
            .expect("selected current source");
        let (issuer, portable_map) =
            GraphSourceIssuer::capture_selected_units(&[selected]).expect("current source issuer");

        let admitted_error = crate::build_runtime_error("selected failure")
            .with_call_frames(vec![crate::CallFrame {
                function: "new_callback".into(),
                source_id: Some(current_source.0),
                span: Some((0, 1)),
                source_owner: Some(current_token),
                diagnostic_source: None,
            }])
            .build();
        let admitted_failure =
            crate::execution::encode_runtime_failure_with_source_issuer_in_context(
                &admitted_error,
                &current,
                &issuer,
                &portable_map,
            )
            .expect("exact new source admits its selected ordinal");
        assert_eq!(
            admitted_failure.call_frames[0].program_source_id,
            Some(ProgramSourceId(0))
        );
        assert!(admitted_failure.call_frames[0]
            .detached_diagnostic
            .is_none());
        let admitted_local =
            crate::execution::decode_runtime_failure_with_source_issuer_in_context(
                &admitted_failure,
                &current,
                &issuer,
                &portable_map,
            )
            .expect("selected ordinal localizes under the issuing session");
        assert_eq!(
            admitted_local.context.call_frames[0].source_id,
            Some(current_source.0)
        );
        assert!(admitted_local.context.call_frames[0].source_owner.is_some());
        let foreign = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        assert!(
            crate::execution::decode_runtime_failure_with_source_issuer_in_context(
                &admitted_failure,
                &foreign,
                &issuer,
                &portable_map,
            )
            .is_err()
        );

        let error = crate::build_runtime_error("retained old failure")
            .with_call_frames(vec![crate::CallFrame {
                function: "old_callback".into(),
                // Deliberately the new table's numeric slot. The old token
                // and captured text must not turn it into admitted new code.
                source_id: Some(current_source.0),
                span: Some((4, 4)),
                source_owner: Some(old_token.clone()),
                diagnostic_source: Some(crate::CallFrameDiagnosticSource::new(
                    Arc::from("old.m"),
                    Arc::from("old\nboom"),
                )),
            }])
            .build();
        let failure = crate::execution::encode_runtime_failure_with_source_issuer_in_context(
            &error,
            &current,
            &issuer,
            &portable_map,
        )
        .expect("old frame is presentation-only, not rebound to new code");
        assert!(failure.call_frames[0].program_source_id.is_none());
        assert_eq!(
            failure.call_frames[0]
                .detached_diagnostic
                .as_ref()
                .map(|source| (source.display_name.as_str(), source.line, source.column)),
            Some(("old.m", Some(2), Some(1)))
        );
        let localized = crate::execution::decode_runtime_failure_with_source_issuer_in_context(
            &failure,
            &current,
            &issuer,
            &portable_map,
        )
        .expect("detached presentation survives exact receipt decoding");
        assert_eq!(localized.context.call_frames[0].source_id, None);
        assert!(localized.context.call_frames[0].source_owner.is_none());
        assert!(localized.context.call_frames[0]
            .diagnostic_source
            .as_ref()
            .is_some_and(crate::CallFrameDiagnosticSource::is_detached));

        let unknown_location = crate::build_runtime_error("retained old failure")
            .with_call_frames(vec![crate::CallFrame {
                function: "old_callback".into(),
                source_id: Some(current_source.0),
                span: None,
                source_owner: Some(old_token.clone()),
                diagnostic_source: Some(crate::CallFrameDiagnosticSource::new(
                    Arc::from("old.m"),
                    Arc::from("old\nboom"),
                )),
            }])
            .build();
        let unknown_failure =
            crate::execution::encode_runtime_failure_with_source_issuer_in_context(
                &unknown_location,
                &current,
                &issuer,
                &portable_map,
            )
            .expect("spanless diagnostic can detach without invented coordinates");
        let unknown = unknown_failure.call_frames[0]
            .detached_diagnostic
            .as_ref()
            .expect("structured presentation");
        assert_eq!((unknown.line, unknown.column), (None, None));
        let unknown_localized =
            crate::execution::decode_runtime_failure_with_source_issuer_in_context(
                &unknown_failure,
                &current,
                &issuer,
                &portable_map,
            )
            .expect("function-only presentation survives decoding");
        assert_eq!(unknown_localized.context.call_stack, ["old_callback"]);
        assert_eq!(
            crate::execution::render_source_call_stack_in_context(&unknown_localized, &current),
            ["old_callback"]
        );
        let value = Value::MException(crate::runtime_error::exception_from_error(error));
        let values = [value];
        let prepared = prepare_payload_graph(&values, &issuer, ValueLimits::default(), None)
            .expect("rooted snapshot");
        let frame_sources = prepared.exception_frame_sources();
        assert_eq!(frame_sources.len(), 1);
        assert_eq!(frame_sources[0].0, runmat_types::SourceId(current_source.0));
        assert!(frame_sources[0]
            .1
            .as_ref()
            .is_some_and(|owner| owner.same_issuance(&old_token)));
        assert_eq!(frame_sources[0].2, GraphFrameSourceRequirement::MayDetach);
        let graph = prepared
            .finish(
                &issuer,
                &portable_map,
                &HashMap::new(),
                &HashMap::new(),
                &HashMap::new(),
            )
            .expect("local diagnostic remains portable after code retirement");
        let GraphNode::Leaf { value } = &graph.nodes[graph.roots[0].0 as usize] else {
            panic!("exception leaf")
        };
        let runmat_execution::value::ValuePayload::Inline(inline) = value else {
            panic!("inline exception")
        };
        let runmat_execution::value::InlineValue::Exception(exception) = inline.as_ref() else {
            panic!("exception payload")
        };
        let frame = &exception
            .runtime
            .as_ref()
            .expect("runtime metadata")
            .call_frames[0];
        assert!(matches!(
            &frame.source,
            runmat_execution::value::ExceptionFrameSource::Detached(_)
        ));
        assert_eq!(frame.program_source_id(), None);
        let rebuilt = decode_payload_graph_with_source_issuer_in_context(
            &graph,
            &current,
            &portable_map,
            &issuer,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            ValueLimits::default(),
        )
        .expect("presentation-only diagnostic roundtrip");
        let Value::MException(rebuilt) = &rebuilt.values()[0] else {
            panic!("rebuilt exception")
        };
        let frame = &rebuilt
            .original()
            .expect("original error")
            .context
            .call_frames[0];
        assert_eq!(frame.source_id, None);
        assert!(frame.source_owner.is_none());
        assert!(frame
            .diagnostic_source
            .as_ref()
            .is_some_and(crate::CallFrameDiagnosticSource::is_detached));
        assert_eq!(
            rebuilt
                .stack
                .iter()
                .map(|frame| frame.as_str())
                .collect::<Vec<_>>(),
            ["old_callback"]
        );
        assert_eq!(
            rebuilt
                .original()
                .expect("original error")
                .context
                .call_stack,
            ["old_callback @ old.m:2:1"]
        );

        let spanless =
            Value::MException(crate::runtime_error::exception_from_error(unknown_location));
        let spanless_graph =
            prepare_payload_graph(&[spanless], &issuer, ValueLimits::default(), None)
                .expect("rooted spanless diagnostic")
                .finish(
                    &issuer,
                    &portable_map,
                    &HashMap::new(),
                    &HashMap::new(),
                    &HashMap::new(),
                )
                .expect("spanless retired diagnostic remains portable");
        let GraphNode::Leaf { value } = &spanless_graph.nodes[spanless_graph.roots[0].0 as usize]
        else {
            panic!("spanless exception leaf")
        };
        let runmat_execution::value::ValuePayload::Inline(inline) = value else {
            panic!("spanless inline exception")
        };
        let runmat_execution::value::InlineValue::Exception(exception) = inline.as_ref() else {
            panic!("spanless exception payload")
        };
        let frame = &exception
            .runtime
            .as_ref()
            .expect("runtime metadata")
            .call_frames[0];
        let runmat_execution::value::ExceptionFrameSource::Detached(diagnostic) = &frame.source
        else {
            panic!("spanless frame retains presentation without code authority");
        };
        assert_eq!(diagnostic.display_name, "old.m");
        assert_eq!((diagnostic.line, diagnostic.column), (None, None));
        assert_eq!(frame.program_source_id(), None);
        let spanless_rebuilt = decode_payload_graph_with_source_issuer_in_context(
            &spanless_graph,
            &current,
            &portable_map,
            &issuer,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            ValueLimits::default(),
        )
        .expect("unknown-location frame reconstructs without admitting reused source");
        let Value::MException(spanless_rebuilt) = &spanless_rebuilt.values()[0] else {
            panic!("spanless reconstructed exception")
        };
        assert_eq!(
            spanless_rebuilt
                .stack
                .iter()
                .map(|frame| frame.as_str())
                .collect::<Vec<_>>(),
            ["old_callback"]
        );
        assert!(spanless_rebuilt
            .original()
            .expect("original error")
            .context
            .call_frames[0]
            .source_owner
            .is_none());
        assert_eq!(
            spanless_rebuilt
                .original()
                .expect("original error")
                .context
                .call_frames[0]
                .source_id,
            None
        );
    });
}
