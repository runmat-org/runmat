//! Checked, portable arithmetic for a captured pure scalar producer.
//!
//! This is a semantic receipt, not a provider submission or a claim that a
//! device can execute the listed operations. Physical layout and effects are
//! admitted by the existing fusion and provider owners before execution.
use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, ensure, Result};
use runmat_execution::Digest;
use runmat_types::{NumericClass, ScalarOperation};
use serde::{Deserialize, Serialize};

pub const SCALAR_PROGRAM_REVISION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarLiteral {
    F32Bits(u32),
    F64Bits(u64),
}

impl ScalarLiteral {
    pub const fn class(self) -> NumericClass {
        match self {
            Self::F32Bits(_) => NumericClass::Single,
            Self::F64Bits(_) => NumericClass::Double,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarNode {
    Input {
        id: u32,
        class: NumericClass,
    },
    Literal {
        id: u32,
        value: ScalarLiteral,
    },
    Apply {
        id: u32,
        source_index: u32,
        class: NumericClass,
        operation: ScalarOperation,
        operands: Vec<u32>,
    },
}

impl ScalarNode {
    pub const fn id(&self) -> u32 {
        match self {
            Self::Input { id, .. } | Self::Literal { id, .. } | Self::Apply { id, .. } => *id,
        }
    }

    pub const fn class(&self) -> NumericClass {
        match self {
            Self::Input { class, .. } | Self::Apply { class, .. } => *class,
            Self::Literal { value, .. } => value.class(),
        }
    }
}

/// Serializable construction form. Deserialization never bypasses validation:
/// callers must convert a wire form through `ScalarProgram::from_wire`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarProgramWire {
    pub source_identity: Digest,
    pub semantic_revision: u32,
    pub nodes: Vec<ScalarNode>,
    pub root: u32,
}

#[derive(Clone)]
pub struct ScalarProgram {
    wire: ScalarProgramWire,
    identity: Digest,
}

impl ScalarProgram {
    pub fn new(
        source_identity: Digest,
        semantic_revision: u32,
        nodes: Vec<ScalarNode>,
        root: u32,
    ) -> Result<Self> {
        Self::from_wire(ScalarProgramWire {
            source_identity,
            semantic_revision,
            nodes,
            root,
        })
    }

    pub fn from_wire(wire: ScalarProgramWire) -> Result<Self> {
        ensure!(
            wire.semantic_revision == SCALAR_PROGRAM_REVISION,
            "unsupported scalar semantic revision"
        );
        ensure!(!wire.nodes.is_empty(), "empty scalar program");
        let mut previous = HashMap::new();
        let mut last_source_index = None;
        for node in &wire.nodes {
            let class = node.class();
            ensure!(
                matches!(class, NumericClass::Single | NumericClass::Double),
                "nonfloating scalar class"
            );
            ensure!(
                !previous.contains_key(&node.id()),
                "duplicate scalar value definition"
            );
            if let ScalarNode::Apply {
                source_index,
                operation,
                operands,
                ..
            } = node
            {
                ensure!(
                    last_source_index.is_none_or(|last| *source_index > last),
                    "unordered scalar source operations"
                );
                last_source_index = Some(*source_index);
                let arity = operation
                    .arity()
                    .ok_or_else(|| anyhow!("operation has no pure scalar lowering"))?;
                ensure!(operands.len() == arity, "scalar operation arity mismatch");
                for operand in operands {
                    let input_class = previous
                        .get(operand)
                        .ok_or_else(|| anyhow!("forward or missing scalar operand"))?;
                    if !matches!(operation, ScalarOperation::Cast(_)) {
                        ensure!(*input_class == class, "scalar operation class mismatch");
                    }
                }
                if let ScalarOperation::Cast(target) = operation {
                    ensure!(*target == class, "scalar cast result class mismatch");
                }
            }
            previous.insert(node.id(), class);
        }
        ensure!(
            previous.contains_key(&wire.root),
            "missing scalar program root"
        );
        let mut required = HashSet::from([wire.root]);
        for node in wire.nodes.iter().rev() {
            if required.contains(&node.id()) {
                if let ScalarNode::Apply { operands, .. } = node {
                    required.extend(operands.iter().copied());
                }
            }
        }
        ensure!(
            required.len() == wire.nodes.len(),
            "unreachable scalar program node"
        );
        let identity = Digest::sha256(serde_json::to_vec(&("scalar-program-v1", &wire))?);
        Ok(Self { wire, identity })
    }

    pub fn wire(&self) -> &ScalarProgramWire {
        &self.wire
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }

    /// A decoded wire is only structurally valid. A future submission owner
    /// must rederive the expected program from its captured, admitted source
    /// and compare the *whole* identity, including operations and literal bits.
    pub fn require_rederived_identity(&self, expected: &Self) -> Result<()> {
        ensure!(
            self.identity == expected.identity,
            "scalar program differs from rederived source"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
