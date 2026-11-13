use super::Budget;
use crate::ContractError;

impl Budget<'_> {
    pub(super) fn node(&mut self, depth: u16) -> Result<(), ContractError> {
        self.usage.depth(depth)?;
        self.usage.nodes(1)
    }
    pub(super) fn add_bytes(&mut self, count: u64) -> Result<(), ContractError> {
        self.usage.bytes(count)
    }
    pub(super) fn add_elements(&mut self, count: u64) -> Result<(), ContractError> {
        self.usage.elements(count)
    }
    pub(super) fn add_text(&mut self, value: &str) -> Result<(), ContractError> {
        self.usage.text(value.len() as u64)
    }
}
