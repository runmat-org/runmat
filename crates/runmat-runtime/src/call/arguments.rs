use serde::{Deserialize, Serialize};

/// Describes one source-level call argument after lowering.
///
/// The descriptor is shared by bytecode and native execution. It captures only
/// language-level expansion semantics and deliberately contains no operand-stack
/// or instruction-decoding state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentSpec {
    pub is_expand: bool,
    pub num_indices: usize,
    pub expand_all: bool,
}
