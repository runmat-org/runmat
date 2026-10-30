//! One wire/bit/continuation witness against the existing canonical serde
//! representation; this fixture grants no selected code/source authority.
use super::*;
use runmat_execution::value::*;
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::rc::Rc;

#[test]
fn selected_json_preserves_large_node_bits_frames_and_original_cancellation() {
    futures::executor::block_on(async {
        let cancellation = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            cancellation.token(),
        );
        let mut budget = runtime.cpu_cooperation();
        let mut cooperation = Cooperation::new(&runtime, &mut budget).unwrap();
        let dense_bits = [(-0.0f64).to_bits(), 0x7ff8_0000_0000_0137];
        let bytes: Vec<u8> = dense_bits.into_iter().flat_map(u64::to_le_bytes).collect();
        let mut graph = PayloadGraph {
            roots: vec![GraphNodeRef(0)],
            nodes: vec![
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::U64(u64::MAX))),
                },
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
                        element_type: ElementType::F64,
                        shape: vec![1, 2],
                        little_endian_data: bytes.clone(),
                    }))),
                },
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::Sparse(SparseValue {
                        element_type: ElementType::F64,
                        rows: 3,
                        columns: 1,
                        column_offsets: vec![0, 2],
                        row_indices: vec![0, 2],
                        little_endian_data: bytes,
                    }))),
                },
                GraphNode::Cell {
                    shape: vec![1, 12000],
                    values: vec![GraphNodeRef(0); 12000],
                },
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::String(
                        "λ\\\"\n😀".repeat(1400).into(),
                    ))),
                },
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::Exception(ExceptionValue {
                        identifier: "fixture:id".into(),
                        message: "retained".into(),
                        stack: vec!["frame".into()],
                        causes: vec![ExceptionValue {
                            identifier: "cause".into(),
                            message: "cause text".into(),
                            stack: Vec::new(),
                            causes: Vec::new(),
                            runtime: None,
                        }],
                        runtime: Some(ExceptionRuntimeMetadata {
                            span: Some((1, 4)),
                            builtin: None,
                            task_id: Some("actual-task".into()),
                            call_frames: vec![ExceptionCallFrame {
                                function: "original".into(),
                                source: ExceptionFrameSource::Detached(DetachedDiagnosticFrame {
                                    display_name: "fixture.m".into(),
                                    line: Some(2),
                                    column: Some(3),
                                }),
                                span: Some((1, 4)),
                            }],
                            call_frames_elided: 0,
                            call_stack: vec!["original".into()],
                            phase: Some("execute".into()),
                            gpu_gather_retry: ExceptionGpuGatherRetry::Never,
                            failure_origin: ExceptionFailureOrigin::LanguageEvaluation,
                        }),
                    }))),
                },
            ],
        };
        graph.nodes.extend([
            GraphNode::Leaf {
                value: ValuePayload::Inline(Box::new(InlineValue::String(StringElement(None)))),
            },
            GraphNode::Leaf {
                value: ValuePayload::Inline(Box::new(InlineValue::StringArray {
                    shape: vec![1, 4],
                    values: vec![
                        "<missing>".into(),
                        "<MISSING>".into(),
                        "".into(),
                        StringElement(None),
                    ],
                })),
            },
        ]);
        let expected = serde_json::to_vec(&graph).unwrap();
        let mut writer = Writer::new();
        writer.graph(&graph, &mut cooperation).await.unwrap();
        let encoded = writer.finish();
        assert_eq!(encoded, expected);
        let mut reader = Reader::new(encoded);
        let restored = reader
            .graph(ValueLimits::default(), &mut cooperation)
            .await
            .unwrap();
        reader.finish().unwrap();
        assert_eq!(restored, graph);
        // Root references and separate arrays share one element/byte budget.
        let aggregate = PayloadGraph {
            roots: vec![GraphNodeRef(0)],
            nodes: vec![
                GraphNode::Cell {
                    shape: vec![1, 2],
                    values: vec![GraphNodeRef(1), GraphNodeRef(2)],
                },
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::U8(7))),
                },
                GraphNode::Leaf {
                    value: ValuePayload::Inline(Box::new(InlineValue::U8(9))),
                },
            ],
        };
        let bytes = serde_json::to_vec(&aggregate).unwrap();
        let exact = ValueLimits {
            max_nodes: 3,
            max_elements: 3,
            max_inline_bytes: 30,
            ..ValueLimits::default()
        };
        Reader::new(bytes.clone())
            .graph(exact, &mut cooperation)
            .await
            .unwrap();
        for limits in [
            ValueLimits {
                max_elements: 2,
                ..exact
            },
            ValueLimits {
                max_inline_bytes: 29,
                ..exact
            },
        ] {
            let error = Reader::new(bytes.clone())
                .graph(limits, &mut cooperation)
                .await
                .unwrap_err();
            assert!(std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<runmat_execution::ContractError>())
                .is_some());
        }
        let exception = PayloadGraph {
            roots: vec![GraphNodeRef(0)],
            nodes: vec![GraphNode::Leaf {
                value: ValuePayload::Inline(Box::new(InlineValue::Exception(ExceptionValue {
                    identifier: String::new(),
                    message: String::new(),
                    stack: vec![String::new(), String::new()],
                    causes: vec![ExceptionValue {
                        identifier: String::new(),
                        message: String::new(),
                        stack: Vec::new(),
                        causes: Vec::new(),
                        runtime: None,
                    }],
                    runtime: None,
                }))),
            }],
        };
        let bytes = serde_json::to_vec(&exception).unwrap();
        for limits in [
            ValueLimits {
                max_elements: 2,
                ..ValueLimits::default()
            },
            ValueLimits {
                max_nodes: 1,
                ..ValueLimits::default()
            },
        ] {
            assert!(Reader::new(bytes.clone())
                .graph(limits, &mut cooperation)
                .await
                .is_err());
        }
        cancellation.cancel(CancellationReason::User);
        let mut reader = Reader::new(expected);
        let error = reader
            .graph(ValueLimits::default(), &mut cooperation)
            .await
            .unwrap_err();
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
    });
}
