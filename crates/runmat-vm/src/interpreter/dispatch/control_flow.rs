use crate::ops::control_flow::ControlFlowAction;

pub enum DispatchDecision {
    ContinueLoop,
    FallThrough,
    Return,
}

#[inline]
pub fn apply_control_flow_action(action: ControlFlowAction, pc: &mut usize) -> DispatchDecision {
    match action {
        ControlFlowAction::Jump(target) => {
            *pc = target;
            DispatchDecision::ContinueLoop
        }
        ControlFlowAction::Next => DispatchDecision::FallThrough,
        ControlFlowAction::Return => DispatchDecision::Return,
    }
}
