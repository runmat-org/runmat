//! Typed factor controls and a prepared, effect-free category layout.
use super::*;
use crate::builtins::common::grouping_keys::KeyAtom;

mod controls;
mod prepare;
pub(super) use controls::{
    parse_directions, parse_full_factors, parse_gap, parse_separator, validate_direction_count,
    validate_gap_count, validate_separator_count, FactorDirection, FactorGap, FactorSeparator,
};
#[cfg(test)]
pub(super) use prepare::prepare_layout;
pub(super) use prepare::prepare_layout_cooperative;

#[derive(Clone, Debug)]
pub(super) struct FactorSlot {
    pub(super) key: Vec<KeyAtom>,
    pub(super) label: String,
    pub(super) position: f64,
    pub(super) box_index: Option<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct BoxLayout {
    pub(super) boxes: Vec<BoxStats>,
    pub(super) slots: Vec<FactorSlot>,
    pub(super) separators: Vec<f64>,
}
