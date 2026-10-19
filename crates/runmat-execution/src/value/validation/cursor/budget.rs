use super::Budget;
use crate::ContractError;

impl Budget<'_> {
    pub(super) fn node(&mut self, depth: u16) -> Result<(), ContractError> {
        if depth > self.limits.max_depth {
            return Err(ContractError::Limit {
                field: "value depth",
                limit: self.limits.max_depth.into(),
            });
        }
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > self.limits.max_nodes {
            Err(ContractError::Limit {
                field: "value nodes",
                limit: self.limits.max_nodes,
            })
        } else {
            Ok(())
        }
    }

    pub(super) fn add_bytes(&mut self, count: u64) -> Result<(), ContractError> {
        self.bytes = self.bytes.saturating_add(count);
        if self.bytes > self.limits.max_inline_bytes {
            Err(ContractError::Limit {
                field: "inline value bytes",
                limit: self.limits.max_inline_bytes,
            })
        } else {
            Ok(())
        }
    }

    pub(super) fn add_elements(&mut self, count: u64) -> Result<(), ContractError> {
        self.elements = self.elements.saturating_add(count);
        if self.elements > self.limits.max_elements {
            Err(ContractError::Limit {
                field: "value elements",
                limit: self.limits.max_elements,
            })
        } else {
            Ok(())
        }
    }

    pub(super) fn add_text(&mut self, value: &str) -> Result<(), ContractError> {
        self.text = self.text.saturating_add(value.len() as u64);
        if self.text > self.limits.max_text_bytes {
            Err(ContractError::Limit {
                field: "value text bytes",
                limit: self.limits.max_text_bytes,
            })
        } else {
            Ok(())
        }
    }
}
