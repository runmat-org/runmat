use super::*;
use crate::bytecode::Bytecode;
use runmat_value::Value;

fn store_transcript() -> Vec<Instr> {
    vec![
        Instr::LoadConst(7.0),
        Instr::StoreVar(0),
        Instr::LoadConst(9.0),
        Instr::StoreLocal(1),
        Instr::LoadConst(11.0),
        Instr::StoreVar(2),
        Instr::Return,
    ]
}

#[test]
fn final_only_kinds_reject_internal_stores_but_not_the_following_final_store() {
    let instructions = store_transcript();
    for kind in [
        FusionKind::Reduction,
        FusionKind::MatmulEpilogue,
        FusionKind::CenteredGram,
        FusionKind::ImageNormalize,
        FusionKind::PowerStepNormalize,
        FusionKind::ExplainedVariance,
    ] {
        assert!(!super::super::fusion_span_has_vm_barrier(
            &instructions,
            &InstrSpan { start: 0, end: 4 },
        ));
        assert!(final_only_span_requires_ordinary_stores(
            &kind,
            &instructions,
            &InstrSpan { start: 0, end: 4 },
        ));
        assert!(!final_only_span_requires_ordinary_stores(
            &kind,
            &instructions,
            &InstrSpan { start: 4, end: 4 },
        ));
        assert!(final_only_span_requires_ordinary_stores(
            &kind,
            &instructions,
            &InstrSpan { start: 4, end: 99 },
        ));
    }
    assert!(!final_only_span_requires_ordinary_stores(
        &FusionKind::ElementwiseChain,
        &instructions,
        &InstrSpan { start: 0, end: 4 },
    ));
}

#[test]
fn ordinary_vm_path_preserves_both_stores_when_final_only_span_is_deferred() {
    let instructions = store_transcript();
    assert!(final_only_span_requires_ordinary_stores(
        &FusionKind::ImageNormalize,
        &instructions,
        &InstrSpan { start: 0, end: 4 },
    ));
    let bytecode = Bytecode::with_instructions(instructions, 3);
    let vars = futures::executor::block_on(crate::interpret(&bytecode)).expect("ordinary stores");
    assert_eq!(*vars.first().unwrap(), Value::Num(7.0));
    assert_eq!(*vars.get(1).unwrap(), Value::Num(9.0));
    assert_eq!(*vars.get(2).unwrap(), Value::Num(11.0));
}
