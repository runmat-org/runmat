use super::*;
use runmat_hir::{FunctionId, SourceId};
use runmat_vm::{Bytecode, FunctionBytecode, FunctionRegistry, Instr};
use std::collections::{HashMap, HashSet};

fn definition(id: usize, name: &str, callee: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        display_name: name.into(),
        source_id: Some(SourceId(5)),
        instructions: vec![Instr::CreateBoundFunctionHandle(
            FunctionId(callee),
            "callee".into(),
        )],
        ..Default::default()
    }
}

fn group(base: usize) -> FunctionRegistry {
    FunctionRegistry::new(HashMap::from([
        (FunctionId(base), definition(base, "first", base + 1)),
        (FunctionId(base + 1), definition(base + 1, "second", base)),
    ]))
}

#[test]
fn complete_mutual_group_reuses_normalized_identities() {
    let incoming = group(1);
    let published = group(100);
    let remap =
        unchanged_unit_remap(&Bytecode::default(), &incoming, &published, &HashSet::new()).unwrap();
    assert_eq!(remap.get(&FunctionId(1)), Some(&FunctionId(100)));
    assert_eq!(remap.get(&FunctionId(2)), Some(&FunctionId(101)));
}

#[test]
fn changed_body_source_scope_capture_layout_and_alias_do_not_reuse() {
    let published = group(100);
    for mutation in 0..6 {
        let mut incoming = group(1);
        let function = incoming.functions.get_mut(&FunctionId(1)).unwrap();
        match mutation {
            0 => function.instructions.push(Instr::LoadConst(9.0)),
            1 => function.source_id = Some(SourceId(6)),
            2 => function.private_owner_scope = "other_scope".into(),
            3 => function.capture_slots.push(1),
            4 => {
                incoming.names.insert("new_alias".into(), FunctionId(1));
            }
            5 => function.varargin_slot = Some(0),
            _ => unreachable!(),
        }
        assert!(
            unchanged_unit_remap(&Bytecode::default(), &incoming, &published, &HashSet::new())
                .is_none()
        );
    }
}
