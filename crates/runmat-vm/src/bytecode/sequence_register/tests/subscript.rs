use super::super::super::*;

#[test]
fn validates_nested_contextual_index_lifecycle() {
    let valid = [
        Instr::LoadVar(0),
        Instr::BeginContextualIndexSelectors { component_count: 1 },
        Instr::LoadVar(1),
        Instr::BeginContextualIndexSelectors { component_count: 2 },
        Instr::LoadContextualIndexEnd { component: 1 },
        Instr::FinishContextualIndexSelectors { component_count: 2 },
        Instr::LoadContextualIndexEnd { component: 0 },
        Instr::FinishContextualIndexSelectors { component_count: 1 },
        Instr::Return,
    ];
    assert!(validate_sequence_register_flow(&valid).is_ok());
    assert!(validate_sequence_register_flow(&[
        Instr::LoadContextualIndexEnd { component: 0 },
        Instr::Return,
    ])
    .unwrap_err()
    .contains("no live index context"));
    assert!(validate_sequence_register_flow(&[
        Instr::BeginContextualIndexSelectors { component_count: 1 },
        Instr::LoadContextualIndexEnd { component: 1 },
        Instr::FinishContextualIndexSelectors { component_count: 1 },
        Instr::Return,
    ])
    .unwrap_err()
    .contains("exceeds selector count"));
    assert!(validate_sequence_register_flow(&[
        Instr::BeginContextualIndexSelectors { component_count: 2 },
        Instr::FinishContextualIndexSelectors { component_count: 1 },
        Instr::Return,
    ])
    .unwrap_err()
    .contains("finishes with"));
}

#[test]
fn validates_subscript_end_receiver_lifecycle_and_path_shape() {
    let valid = [
        Instr::BeginSubscriptEndReceiver { prefix: Vec::new() },
        Instr::LoadSubscriptEnd {
            component: 0,
            component_count: 2,
        },
        Instr::LoadSubscriptEnd {
            component: 1,
            component_count: 2,
        },
        Instr::FinishSubscriptEndReceiver {
            selector_count: 2,
            prefix_operand_count: 0,
        },
        Instr::Return,
    ];
    assert!(validate_sequence_register_flow(&valid).is_ok());
    assert!(validate_sequence_register_flow(&[
        Instr::LoadSubscriptEnd {
            component: 0,
            component_count: 1,
        },
        Instr::Return,
    ])
    .unwrap_err()
    .contains("no prepared receiver"));
    assert!(validate_sequence_register_flow(&[
        Instr::ReadSubscriptPath {
            steps: Vec::new(),
            selection: runmat_types::SequenceUse::RequireSingle,
            context: runmat_types::ObjectIndexingContext::Expression,
            to_sequence_register: false,
        },
        Instr::Return,
    ])
    .unwrap_err()
    .contains("path at instruction 0 is empty"));
}
