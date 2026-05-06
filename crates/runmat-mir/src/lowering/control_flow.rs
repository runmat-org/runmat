use crate::BasicBlockId;

#[derive(Debug, Default)]
#[allow(dead_code)]
pub(crate) struct ControlFlowBuilder {
    next_block: usize,
}

impl ControlFlowBuilder {
    #[allow(dead_code)]
    pub(crate) fn fresh_block(&mut self) -> BasicBlockId {
        let id = BasicBlockId(self.next_block);
        self.next_block += 1;
        id
    }
}
