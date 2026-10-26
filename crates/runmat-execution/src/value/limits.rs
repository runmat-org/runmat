//! Aggregate semantic counters shared by validation and incremental decoding.
use super::ValueLimits;
use crate::ContractError;

pub struct ValueLimitBudget {
    limits: ValueLimits,
    nodes: u64,
    bytes: u64,
    elements: u64,
    text: u64,
}
impl ValueLimitBudget {
    pub fn new(limits: ValueLimits) -> Self {
        Self {
            limits,
            nodes: 0,
            bytes: 0,
            elements: 0,
            text: 0,
        }
    }
    fn add(
        value: &mut u64,
        count: u64,
        limit: u64,
        field: &'static str,
    ) -> Result<(), ContractError> {
        *value = value
            .checked_add(count)
            .filter(|value| *value <= limit)
            .ok_or(ContractError::Limit { field, limit })?;
        Ok(())
    }
    pub fn nodes(&mut self, count: u64) -> Result<(), ContractError> {
        Self::add(&mut self.nodes, count, self.limits.max_nodes, "value nodes")
    }
    pub fn depth(&self, depth: u16) -> Result<(), ContractError> {
        if depth > self.limits.max_depth {
            Err(ContractError::Limit {
                field: "value depth",
                limit: self.limits.max_depth.into(),
            })
        } else {
            Ok(())
        }
    }
    pub fn bytes(&mut self, count: u64) -> Result<(), ContractError> {
        Self::add(
            &mut self.bytes,
            count,
            self.limits.max_inline_bytes,
            "inline value bytes",
        )
    }
    pub fn elements(&mut self, count: u64) -> Result<(), ContractError> {
        Self::add(
            &mut self.elements,
            count,
            self.limits.max_elements,
            "value elements",
        )
    }
    pub fn text(&mut self, count: u64) -> Result<(), ContractError> {
        Self::add(
            &mut self.text,
            count,
            self.limits.max_text_bytes,
            "value text bytes",
        )
    }
    pub fn remaining(&self) -> ValueLimits {
        ValueLimits {
            max_nodes: self.limits.max_nodes - self.nodes,
            max_inline_bytes: self.limits.max_inline_bytes - self.bytes,
            max_elements: self.limits.max_elements - self.elements,
            max_text_bytes: self.limits.max_text_bytes - self.text,
            ..self.limits
        }
    }
}
