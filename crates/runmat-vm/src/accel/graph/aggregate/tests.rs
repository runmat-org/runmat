use super::*;

fn aggregate() -> Instr {
    Instr::CreateMatrixFromSequences {
        rows: 1,
        row_lengths: vec![2],
        elements: vec![AggregateElementSpec::Single; 2],
    }
}

#[test]
fn numeric_aggregate_preserves_operand_and_store_tracking_without_fabricating_a_fold() {
    let instructions = [
        Instr::LoadVar(0),
        Instr::LoadVar(1),
        aggregate(),
        Instr::StoreVar(2),
    ];
    let types = [
        Type::Tensor {
            shape: Some(vec![Some(3), Some(1)]),
        },
        Type::Tensor {
            shape: Some(vec![Some(3), Some(1)]),
        },
        Type::Unknown,
    ];
    let (graph, stores) = build_accel_graph_with_stores(&instructions, &types);
    let value = graph.value(stores[0].value_id).unwrap();
    assert_eq!(
        value.ty,
        Type::Tensor {
            shape: Some(vec![Some(3), Some(2)])
        }
    );
    assert!(value.constant.is_none());
    let node = graph
        .nodes
        .iter()
        .find(|node| node.span.start == 2)
        .unwrap();
    assert_eq!(node.inputs.len(), 2);
    assert_eq!(node.category, AccelOpCategory::Other);
    assert!(!node.tags.contains(&AccelGraphTag::FoldedLiteralArray));

    for constructor in [aggregate(), Instr::CreateMatrix(1, 2)] {
        let instructions = [
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            constructor,
            Instr::StoreVar(2),
        ];
        let (graph, _) =
            build_accel_graph_with_stores(&instructions, &[Type::Num, Type::Num, Type::Unknown]);
        let node = graph
            .nodes
            .iter()
            .find(|node| node.span.start == 2)
            .unwrap();
        assert_eq!(node.inputs.len(), 2);
        for (expected, input) in node.inputs.iter().enumerate() {
            assert!(
                matches!(&graph.value(*input).unwrap().origin,
                ValueOrigin::Variable { kind: VarKind::Global, index } if *index == expected),
                "aggregate graph inputs preserve source variable order"
            );
        }
    }
}

#[test]
fn primitive_fold_survives_but_integer_values_and_overloaded_objects_do_not_fold() {
    let instructions = [
        Instr::LoadConst(3.0),
        Instr::LoadConst(4.0),
        aggregate(),
        Instr::StoreVar(0),
    ];
    let (graph, stores) = build_accel_graph_with_stores(&instructions, &[Type::Unknown]);
    let Some(Value::Tensor(tensor)) = graph.value(stores[0].value_id).unwrap().constant.as_ref()
    else {
        panic!("proved f64 constants keep their fold");
    };
    assert_eq!(tensor.materialize_f64(), vec![3.0, 4.0]);

    for constructor in [aggregate(), Instr::CreateMatrix(1, 2)] {
        let instructions = [
            Instr::LoadInt(runmat_value::IntValue::U64(9007199254740993)),
            Instr::LoadInt(runmat_value::IntValue::U64(u64::MAX)),
            constructor,
            Instr::StoreVar(0),
        ];
        let types = [Type::Unknown];
        let mut builder = GraphBuilder::new(&instructions, &types);
        for (pc, instruction) in instructions.iter().enumerate() {
            builder.process_instr(pc, instruction);
        }
        let id = builder.var_values[&VarKey::Global(0)];
        assert!(
            builder.values[id as usize].constant.is_none(),
            "integer payloads never round through f64"
        );
        assert!(
            builder.stack.is_empty(),
            "nonfoldable construction still consumes every exact operand"
        );
    }

    let instructions = [Instr::LoadVar(0), Instr::LoadConst(1.0), aggregate()];
    let types = [Type::Object {
        class_name: None,
        shape: None,
    }];
    let mut builder = GraphBuilder::new(&instructions, &types);
    for (pc, instruction) in instructions.iter().enumerate() {
        builder.process_instr(pc, instruction);
    }
    assert!(builder.stack.is_empty());
    assert!(
        builder.var_values.is_empty(),
        "overloaded concat invalidates stale bindings"
    );
}
