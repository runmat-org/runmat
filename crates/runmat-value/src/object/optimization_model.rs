//! Immutable optimization-model array DAGs on the canonical GC/native owner.
//! Public runtime producers validate class, dimensions, operators and numeric
//! carriers. These layouts create no source, callable or allocation authority.
mod equation;
mod roles;
pub use equation::EquationProblemData;
pub use roles::{OptimizationConstraintRelation, OptimizationObjectiveSense};

use crate::{ExactArrayStorage, SharedShape, Utf8Text, Value};
use runmat_gc_api::{GcHandle, Trace, Tracer};
use runmat_types::{BuiltinId, OperatorKind};

pub type ModelAxisLabels = ExactArrayStorage<Utf8Text>;
pub type ModelAxisNames = ExactArrayStorage<Option<ModelAxisLabels>>;

/// Move ordinary parser backing into shared metadata without cloning names.
/// This is explicitly unmanaged construction, not a finite admission receipt.
pub fn model_axis_names_unbounded(
    names: Vec<Option<Vec<Utf8Text>>>,
) -> Result<ModelAxisNames, crate::ExactHostNumericAllocationError> {
    let mut axes = crate::exact_capacity_vec(names.len())?;
    for names in names {
        axes.push(names.map(ExactArrayStorage::shared_unbounded));
    }
    Ok(ExactArrayStorage::shared_unbounded(axes))
}

#[derive(Debug, PartialEq)]
pub struct OptimizationExpressionNode {
    pub shape: SharedShape,
    pub index_names: ModelAxisNames,
    pub kind: OptimizationExpressionKind,
}
#[derive(Debug, PartialEq)]
pub enum OptimizationExpressionKind {
    /// Unassigned expression slots are not numerical zeros.
    Unassigned,
    /// Nullary accumulator, distinct from unassigned coordinates and scalar zero.
    EmptyInitializer,
    /// A shape-owned all-zero expression; no numerical payload is duplicated.
    Zero,
    Labels {
        source: GcHandle,
    },
    /// Actual native variable reference; its retained original definition is
    /// authoritative. There is no variable-name lookup in expression storage.
    Variable(Value),
    Constant(Value),
    Unary {
        operator: OperatorKind,
        operand: GcHandle,
    },
    Binary {
        operator: OperatorKind,
        left: GcHandle,
        right: GcHandle,
    },
    Index {
        source: GcHandle,
        indices: ExactArrayStorage<usize>,
    },
    Assign {
        source: GcHandle,
        indices: ExactArrayStorage<usize>,
        replacement: GcHandle,
        selection_shape: SharedShape,
    },
    Concatenate {
        dimension: usize,
        operands: Vec<GcHandle>,
    },
    /// A selected canonical builtin, not an untrusted function-name string.
    Builtin {
        function: BuiltinId,
        operands: Vec<GcHandle>,
        arguments: Vec<Value>,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationExpressionRef {
    pub node: GcHandle,
    pub shape: SharedShape,
}
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationConstraintRef {
    pub index_names: ModelAxisNames,
    pub left: GcHandle,
    pub right: GcHandle,
    pub relation: OptimizationConstraintRelation,
    pub shape: SharedShape,
}
#[derive(Debug, PartialEq)]
pub enum OptimizationObjectiveSenses {
    Uniform(OptimizationObjectiveSense),
    Named(Vec<(Utf8Text, OptimizationObjectiveSense)>),
}
#[derive(Debug, PartialEq)]
pub struct OptimizationProblemData {
    pub description: Value,
    pub sense: OptimizationObjectiveSenses,
    pub objective: Value,
    pub constraints: Value,
}

#[derive(Debug, PartialEq)]
pub enum OptimizationModelPayload {
    Node(OptimizationExpressionNode),
    Expression(OptimizationExpressionRef),
    Constraint(OptimizationConstraintRef),
}
impl OptimizationModelPayload {
    /// Direct edges for the existing iterative Value tracer; no recursive
    /// reference-counted child chain or second tracing traversal is needed.
    pub fn handle_at(&self, index: usize) -> Option<GcHandle> {
        use OptimizationExpressionKind as Kind;
        match self {
            Self::Expression(value) => (index == 0).then_some(value.node),
            Self::Constraint(value) => match index {
                0 => Some(value.left),
                1 => Some(value.right),
                _ => None,
            },
            Self::Node(node) => match &node.kind {
                Kind::Unary { operand, .. } => (index == 0).then_some(*operand),
                Kind::Binary { left, right, .. } => match index {
                    0 => Some(*left),
                    1 => Some(*right),
                    _ => None,
                },
                Kind::Labels { source } | Kind::Index { source, .. } => {
                    (index == 0).then_some(*source)
                }
                Kind::Assign {
                    source,
                    replacement,
                    ..
                } => match index {
                    0 => Some(*source),
                    1 => Some(*replacement),
                    _ => None,
                },
                Kind::Concatenate { operands, .. } | Kind::Builtin { operands, .. } => {
                    operands.get(index).copied()
                }
                Kind::Unassigned
                | Kind::EmptyInitializer
                | Kind::Zero
                | Kind::Variable(_)
                | Kind::Constant(_) => None,
            },
        }
    }
    pub fn value_at(&self, index: usize) -> Option<&Value> {
        match self {
            Self::Node(node) => match &node.kind {
                OptimizationExpressionKind::Variable(value)
                | OptimizationExpressionKind::Constant(value) => (index == 0).then_some(value),
                OptimizationExpressionKind::Builtin { arguments, .. } => arguments.get(index),
                _ => None,
            },
            Self::Expression(_) | Self::Constraint(_) => None,
        }
    }
}

pub(in crate::object) mod storage;
pub use storage::{
    ExistingOptimizationNativeInput, OptimizationMetadataQuote, OptimizationNativeAdmission,
    OptimizationNativeEdge, OptimizationNativeInput, OptimizationNativeSource,
    PreparedOptimizationNative,
};

impl OptimizationProblemData {
    pub fn value_at(&self, index: usize) -> Option<&Value> {
        match index {
            0 => Some(&self.description),
            1 => Some(&self.objective),
            2 => Some(&self.constraints),
            _ => None,
        }
    }
}
impl Trace for OptimizationProblemData {
    fn trace(&self, tracer: &mut dyn Tracer) {
        let mut index = 0;
        while let Some(value) = self.value_at(index) {
            value.trace(tracer);
            index += 1;
        }
    }
}
impl Trace for OptimizationModelPayload {
    fn trace(&self, tracer: &mut dyn Tracer) {
        let mut index = 0;
        while let Some(handle) = self.handle_at(index) {
            tracer.mark(handle);
            index += 1;
        }
        let mut index = 0;
        while let Some(value) = self.value_at(index) {
            value.trace(tracer);
            index += 1;
        }
    }
}
