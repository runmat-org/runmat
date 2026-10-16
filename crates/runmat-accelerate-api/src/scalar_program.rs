//! Checked, portable arithmetic for a captured pure scalar producer.
//!
//! This is a semantic receipt, not a provider submission or a claim that a
//! device can execute the listed operations. Physical layout and effects are
//! admitted by the existing fusion and provider owners before execution.
use std::collections::HashSet;

use anyhow::{anyhow, ensure, Result};
use runmat_execution::Digest;
use runmat_types::{NumericClass, ScalarOperation};
use serde::{Deserialize, Serialize};

mod exact_product;
pub use exact_product::ExactBinary32Product;
mod implementation;
pub use implementation::{EmittedScalarImplementation, ScalarImplementation, ScalarTransport};

pub const SCALAR_PROGRAM_REVISION: u32 = 6;

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
        let mut definitions = Vec::new();
        definitions.try_reserve_exact(wire.nodes.len())?;
        definitions.extend(
            wire.nodes
                .iter()
                .enumerate()
                .map(|(index, node)| (node.id(), index)),
        );
        definitions.sort_unstable_by_key(|(id, _)| *id);
        ensure!(
            !definitions.windows(2).any(|pair| pair[0].0 == pair[1].0),
            "duplicate scalar value definition"
        );
        let mut last_source_index = None;
        for (node_index, node) in wire.nodes.iter().enumerate() {
            let class = node.class();
            ensure!(
                matches!(class, NumericClass::Single | NumericClass::Double),
                "nonfloating scalar class"
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
                let mut input_classes = Vec::new();
                input_classes.try_reserve_exact(operands.len())?;
                for operand in operands {
                    let position = definitions
                        .binary_search_by_key(operand, |(id, _)| *id)
                        .ok()
                        .map(|index| definitions[index].1)
                        .filter(|&index| index < node_index)
                        .ok_or_else(|| anyhow!("forward or missing scalar operand"))?;
                    let input_class = wire.nodes[position].class();
                    input_classes.push(input_class);
                    if matches!(operation, ScalarOperation::Operator(operator) if *operator != runmat_types::OperatorKind::ElementwisePower)
                    {
                        ensure!(input_class == class, "scalar operation class mismatch");
                    }
                }
                if *operation
                    == ScalarOperation::Operator(runmat_types::OperatorKind::ElementwisePower)
                {
                    let control = match operands.as_slice() {
                        [_, exponent] => wire.nodes.iter().find_map(|node| match node {
                            ScalarNode::Literal {
                                id,
                                value: ScalarLiteral::F64Bits(bits),
                            } if id == exponent => {
                                Some(crate::PowerScalarControl::DoubleBits(*bits))
                            }
                            _ => None,
                        }),
                        _ => None,
                    };
                    ensure!(class == NumericClass::Single && input_classes == [NumericClass::Single, NumericClass::Double] && control.and_then(crate::ExactSingleBasePower::for_control).is_some(), "power scalar program requires certified Single base and original Double literal control");
                }
                if let ScalarOperation::RealArithmetic(recipe) = operation {
                    ensure!(
                        matches!(input_classes.as_slice(), [left, right]
                            if runmat_types::RealFloatingRecipe::for_classes(recipe.operator(), *left, *right) == Some(*recipe))
                            && class == recipe.result_class(),
                        "represented arithmetic differs from operand or result classes"
                    );
                }
                if let ScalarOperation::UnaryMath(_) = operation {
                    ensure!(
                        input_classes == [class],
                        "unary math must preserve represented floating class"
                    );
                }
                if let ScalarOperation::Cast(target) = operation {
                    ensure!(*target == class, "scalar cast result class mismatch");
                }
            }
        }
        ensure!(
            definitions
                .binary_search_by_key(&wire.root, |(id, _)| *id)
                .is_ok(),
            "missing scalar program root"
        );
        let mut required = Vec::new();
        required.try_reserve_exact(wire.nodes.len())?;
        required.resize(wire.nodes.len(), false);
        let root_position = definitions
            .binary_search_by_key(&wire.root, |(id, _)| *id)
            .map(|index| definitions[index].1)
            .map_err(|_| anyhow!("missing scalar program root"))?;
        required[root_position] = true;
        for (index, node) in wire.nodes.iter().enumerate().rev() {
            if required[index] {
                if let ScalarNode::Apply { operands, .. } = node {
                    for operand in operands {
                        let position = definitions
                            .binary_search_by_key(operand, |(id, _)| *id)
                            .map(|position| definitions[position].1)
                            .map_err(|_| anyhow!("missing scalar program operand"))?;
                        required[position] = true;
                    }
                }
            }
        }
        ensure!(
            required.iter().all(|&needed| needed),
            "unreachable scalar program node"
        );
        let identity = Digest::sha256_serialized(&("scalar-program-v6", &wire))?;
        Ok(Self { wire, identity })
    }

    pub fn wire(&self) -> &ScalarProgramWire {
        &self.wire
    }

    /// Structural physical equivalent, not source/provenance authority. The
    /// whole program must still match the captured source before submission.
    pub fn exact_product(&self, output: u32) -> Option<(u32, u32, ExactBinary32Product)> {
        let ScalarNode::Apply {
            operation: ScalarOperation::RealArithmetic(recipe),
            operands,
            class,
            ..
        } = self.wire.nodes.iter().find(|node| node.id() == output)?
        else {
            return None;
        };
        if recipe.operator() != runmat_types::OperatorKind::ElementwiseMultiply
            || recipe.computation_class() != NumericClass::Double
            || *class != NumericClass::Single
        {
            return None;
        }
        for (base, control) in [(operands[0], operands[1]), (operands[1], operands[0])] {
            let base_class = self
                .wire
                .nodes
                .iter()
                .find(|node| node.id() == base)?
                .class();
            if base_class != NumericClass::Single {
                continue;
            }
            if let Some(ScalarNode::Literal {
                value: ScalarLiteral::F64Bits(bits),
                ..
            }) = self.wire.nodes.iter().find(|node| node.id() == control)
            {
                if let Some(exact) = ExactBinary32Product::for_double_bits(*bits) {
                    return Some((base, control, exact));
                }
            }
        }
        None
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }

    /// Literal controls whose every use is a checked exact implementation. This
    /// describes source metadata; it does not authenticate live allocations.
    pub fn metadata_control_ids(&self) -> HashSet<u32> {
        self.metadata_control_ids_sorted()
            .expect("scalar control geometry is representable")
            .into_iter()
            .collect()
    }

    /// The indexed producer uses a request-sized vector rather than creating
    /// an unbounded hash table while preparing a finite host request.
    pub fn metadata_control_ids_sorted(&self) -> Result<Vec<u32>> {
        let mut controls = Vec::new();
        controls.try_reserve_exact(
            self.wire
                .nodes
                .len()
                .checked_mul(2)
                .ok_or_else(|| anyhow!("scalar control count overflows"))?,
        )?;
        controls.extend(self.wire.nodes.iter().filter_map(|node| match node {
            ScalarNode::Apply {
                operation: ScalarOperation::Operator(runmat_types::OperatorKind::ElementwisePower),
                operands,
                ..
            } => operands.get(1).copied(),
            _ => None,
        }));
        controls.extend(
            self.wire
                .nodes
                .iter()
                .filter_map(|node| self.exact_product(node.id()).map(|(_, control, _)| control)),
        );
        controls.sort_unstable();
        controls.dedup();
        let excluded_capacity = self
            .wire
            .nodes
            .iter()
            .try_fold(0usize, |count, node| {
                count.checked_add(match node {
                    ScalarNode::Apply { operands, .. } => operands.len(),
                    _ => 0,
                })
            })
            .ok_or_else(|| anyhow!("scalar control operand count overflows"))?;
        let mut excluded = Vec::new();
        excluded.try_reserve_exact(
            excluded_capacity
                .checked_add(1)
                .ok_or_else(|| anyhow!("scalar control exclusion count overflows"))?,
        )?;
        excluded.push(self.wire.root);
        for node in &self.wire.nodes {
            if let ScalarNode::Apply {
                operation,
                operands,
                ..
            } = node
            {
                for (position, operand) in operands.iter().enumerate() {
                    let power = position == 1
                        && *operation
                            == ScalarOperation::Operator(
                                runmat_types::OperatorKind::ElementwisePower,
                            );
                    let product = self
                        .exact_product(node.id())
                        .is_some_and(|(_, control, _)| control == *operand);
                    if !power && !product {
                        excluded.push(*operand);
                    }
                }
            }
        }
        excluded.sort_unstable();
        excluded.dedup();
        controls.retain(|id| excluded.binary_search(id).is_err());
        Ok(controls)
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
