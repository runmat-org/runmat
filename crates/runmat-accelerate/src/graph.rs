use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use runmat_builtins::Type;
use runmat_value::Value as BuiltinValue;

#[path = "graph/folded_unary.rs"]
mod folded_unary;
mod matrix_product;

pub type NodeId = u32;
pub type ValueId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelGraph {
    pub nodes: Vec<AccelNode>,
    pub values: Vec<ValueInfo>,
    pub var_bindings: HashMap<ValueId, VarBinding>,
    pub node_bindings: HashMap<NodeId, VarBinding>,
}

impl AccelGraph {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: NodeId) -> Option<&AccelNode> {
        self.nodes.get(id as usize)
    }

    pub fn value(&self, id: ValueId) -> Option<&ValueInfo> {
        self.values.get(id as usize)
    }

    pub fn var_binding(&self, id: ValueId) -> Option<&VarBinding> {
        self.var_bindings.get(&id)
    }

    pub fn node_binding(&self, id: NodeId) -> Option<&VarBinding> {
        self.node_bindings.get(&id)
    }

    /// A value fact on a live binding is not a frozen source literal. Fusion
    /// planning and reduction signatures share this provenance boundary.
    pub fn source_literal_constant(&self, id: ValueId) -> Option<&BuiltinValue> {
        if self.var_binding(id).is_some() {
            return None;
        }
        let info = self.value(id)?;
        if matches!(info.origin, ValueOrigin::Constant) {
            info.constant.as_ref()
        } else {
            None
        }
    }

    /// Prove the compiler's folded matrix constructor from its original
    /// row-major numeric literals. A constant fact on a general node is not
    /// enough to replace an executable source producer.
    pub fn exact_folded_literal_array(&self, id: ValueId) -> Option<&BuiltinValue> {
        let info = self.value(id)?;
        let ValueOrigin::NodeOutput { node, output: 0 } = info.origin else {
            return None;
        };
        let producer = self.node(node)?;
        if producer.id != node
            || producer.outputs.as_slice() != [id]
            || producer.span.start > producer.span.end
            || producer.category != AccelOpCategory::Other
            || producer.label != AccelNodeLabel::Primitive(PrimitiveOp::UPlus)
            || producer.tags.as_slice() != [AccelGraphTag::FoldedLiteralArray]
        {
            return None;
        }
        let Some(BuiltinValue::Tensor(tensor)) = info.constant.as_ref() else {
            return None;
        };
        let [rows, columns] = tensor.shape() else {
            return None;
        };
        if !matches!(&info.shape, ShapeInfo::Tensor(dims)
            if dims.as_slice() == [Some(*rows), Some(*columns)])
            || !matches!(&info.ty, Type::Tensor { shape: Some(dims) }
                if dims.as_slice() == [Some(*rows), Some(*columns)])
            || producer.inputs.len() != rows.checked_mul(*columns)?
        {
            return None;
        }
        let elements = tensor.as_f64_slice()?;
        if elements.len() != producer.inputs.len() {
            return None;
        }
        for (position, input) in producer.inputs.iter().enumerate() {
            let Some(BuiltinValue::Num(source)) = self.source_literal_constant(*input) else {
                return None;
            };
            let row = position / *columns;
            let column = position % *columns;
            if elements[row + column * *rows].to_bits() != source.to_bits() {
                return None;
            }
        }
        info.constant.as_ref()
    }

    /// An immutable source array, not a constant fact on a live binding.
    pub fn source_literal_array(&self, id: ValueId) -> Option<&BuiltinValue> {
        let value = self.exact_folded_literal_array(id)?;
        let ValueOrigin::NodeOutput { node, .. } = self.value(id)?.origin else {
            return None;
        };
        if self.var_binding(id).is_some() || self.node_binding(node).is_some() {
            return None;
        }
        Some(value)
    }

    /// Exact vector control represented by a proved folded source array.
    pub fn folded_literal_numeric_vector(&self, id: ValueId) -> Option<&BuiltinValue> {
        let value = self.source_literal_array(id)?;
        let BuiltinValue::Tensor(tensor) = value else {
            return None;
        };
        let [rows, columns] = tensor.shape() else {
            return None;
        };
        (*rows == 1 || *columns == 1).then_some(value)
    }

    pub fn detect_fusion_groups(&self) -> Vec<crate::fusion::FusionGroup> {
        crate::fusion::detect_fusion_groups(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelNode {
    pub id: NodeId,
    pub label: AccelNodeLabel,
    pub category: AccelOpCategory,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub span: InstrSpan,
    pub tags: Vec<AccelGraphTag>,
}

impl AccelNode {
    /// Syntax/compiled-binding identity only, not a numerical execution grant.
    pub fn is_matrix_product(&self) -> bool {
        matrix_product::is_matrix_product(self)
    }
    pub fn is_elementwise(&self) -> bool {
        self.category == AccelOpCategory::Elementwise
    }

    pub fn is_reduction(&self) -> bool {
        self.category == AccelOpCategory::Reduction
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstrSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccelOpCategory {
    Elementwise,
    Reduction,
    MatMul,
    Transpose,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccelNodeLabel {
    Primitive(PrimitiveOp),
    Builtin {
        name: String,
        #[serde(default)]
        origin: FusionCallOrigin,
    },
    Unknown,
}

mod call_origin;
pub use call_origin::FusionCallOrigin;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrimitiveOp {
    Add,
    Sub,
    Mul,
    Pow,
    Neg,
    UPlus,
    ElemMul,
    ElemDiv,
    ElemPow,
    ElemLeftDiv,
    LessEqual,
    Less,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Transpose,
}

impl PrimitiveOp {
    /// Language identity, shared by inference and exact host-owner selection.
    pub const fn operator_kind(self) -> runmat_types::OperatorKind {
        use runmat_types::OperatorKind as O;
        match self {
            Self::Add => O::Add,
            Self::Sub => O::Subtract,
            Self::Mul => O::MatrixMultiply,
            Self::Pow => O::MatrixPower,
            Self::Neg => O::UnaryMinus,
            Self::UPlus => O::UnaryPlus,
            Self::ElemMul => O::ElementwiseMultiply,
            Self::ElemDiv => O::ElementwiseDivide,
            Self::ElemPow => O::ElementwisePower,
            Self::ElemLeftDiv => O::ElementwiseLeftDivide,
            Self::LessEqual => O::LessEqual,
            Self::Less => O::Less,
            Self::Greater => O::Greater,
            Self::GreaterEqual => O::GreaterEqual,
            Self::Equal => O::Equal,
            Self::NotEqual => O::NotEqual,
            Self::Transpose => O::Transpose,
        }
    }
}

impl fmt::Display for PrimitiveOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PrimitiveOp::Add => "Add",
            PrimitiveOp::Sub => "Sub",
            PrimitiveOp::Mul => "Mul",
            PrimitiveOp::Pow => "Pow",
            PrimitiveOp::Neg => "Neg",
            PrimitiveOp::UPlus => "UPlus",
            PrimitiveOp::ElemMul => "ElemMul",
            PrimitiveOp::ElemDiv => "ElemDiv",
            PrimitiveOp::ElemPow => "ElemPow",
            PrimitiveOp::ElemLeftDiv => "ElemLeftDiv",
            PrimitiveOp::LessEqual => "LessEqual",
            PrimitiveOp::Less => "Less",
            PrimitiveOp::Greater => "Greater",
            PrimitiveOp::GreaterEqual => "GreaterEqual",
            PrimitiveOp::Equal => "Equal",
            PrimitiveOp::NotEqual => "NotEqual",
            PrimitiveOp::Transpose => "Transpose",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccelGraphTag {
    Unary,
    Elementwise,
    Reduction,
    MatMul,
    Transpose,
    ArrayConstruct,
    /// A VM matrix instruction folded from exact numeric literal operands.
    /// Unlike ArrayConstruct, this is source-value provenance, not a shape hint.
    FoldedLiteralArray,
    /// The graph builder computed this unary output from an exact scalar
    /// constant. Consumers must still read the represented output value.
    UnaryScalarFold,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueInfo {
    pub id: ValueId,
    pub origin: ValueOrigin,
    pub ty: Type,
    pub shape: ShapeInfo,
    #[serde(skip)]
    pub constant: Option<BuiltinValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VarBinding {
    pub kind: VarKind,
    pub index: usize,
}

impl ValueInfo {
    pub fn update_type(&mut self, ty: &Type) {
        self.ty = match (&self.ty, ty) {
            (Type::Unknown, other) => other.clone(),
            (existing, other) => existing.unify(other),
        };
        self.shape = ShapeInfo::from_type(&self.ty);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValueOrigin {
    Variable { kind: VarKind, index: usize },
    NodeOutput { node: NodeId, output: usize },
    Constant,
    Unknown,
}

#[derive(Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialEq, Eq)]
pub enum VarKind {
    Global,
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShapeInfo {
    Unknown,
    Scalar,
    Tensor(Vec<Option<usize>>),
}

impl ShapeInfo {
    pub fn from_type(ty: &Type) -> Self {
        match ty {
            Type::Int | Type::Num | Type::Bool => ShapeInfo::Scalar,
            Type::Logical { shape } => match shape {
                Some(dims) => ShapeInfo::Tensor(dims.clone()),
                None => ShapeInfo::Tensor(Vec::new()),
            },
            Type::Tensor { shape } => match shape {
                Some(dims) => ShapeInfo::Tensor(dims.clone()),
                None => ShapeInfo::Tensor(Vec::new()),
            },
            _ => ShapeInfo::Unknown,
        }
    }

    pub fn unify(&self, other: &ShapeInfo) -> ShapeInfo {
        match (self, other) {
            (ShapeInfo::Unknown, _) | (_, ShapeInfo::Unknown) => ShapeInfo::Unknown,
            (ShapeInfo::Scalar, ShapeInfo::Scalar) => ShapeInfo::Scalar,
            (ShapeInfo::Scalar, ShapeInfo::Tensor(dims))
            | (ShapeInfo::Tensor(dims), ShapeInfo::Scalar) => ShapeInfo::Tensor(dims.clone()),
            (ShapeInfo::Tensor(a), ShapeInfo::Tensor(b)) => {
                ShapeInfo::Tensor(runmat_builtins::shape_rules::broadcast_shapes(a, b))
            }
        }
    }

    pub fn to_type(&self) -> Type {
        match self {
            ShapeInfo::Unknown => Type::Unknown,
            ShapeInfo::Scalar => Type::Num,
            ShapeInfo::Tensor(dims) => {
                if dims.is_empty() {
                    Type::Tensor { shape: None }
                } else {
                    Type::Tensor {
                        shape: Some(dims.clone()),
                    }
                }
            }
        }
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, ShapeInfo::Scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folded_axis_vector_requires_the_exact_constructor_proof() {
        let mut graph = AccelGraph {
            nodes: vec![AccelNode {
                id: 0,
                label: AccelNodeLabel::Primitive(PrimitiveOp::UPlus),
                category: AccelOpCategory::Other,
                inputs: vec![0, 1],
                outputs: vec![2],
                span: InstrSpan { start: 3, end: 3 },
                tags: vec![AccelGraphTag::FoldedLiteralArray],
            }],
            values: vec![
                ValueInfo {
                    id: 0,
                    origin: ValueOrigin::Constant,
                    ty: Type::Num,
                    shape: ShapeInfo::Scalar,
                    constant: Some(BuiltinValue::Num(2.0)),
                },
                ValueInfo {
                    id: 1,
                    origin: ValueOrigin::Constant,
                    ty: Type::Num,
                    shape: ShapeInfo::Scalar,
                    constant: Some(BuiltinValue::Num(3.0)),
                },
                ValueInfo {
                    id: 2,
                    origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                    ty: Type::Tensor {
                        shape: Some(vec![Some(1), Some(2)]),
                    },
                    shape: ShapeInfo::Tensor(vec![Some(1), Some(2)]),
                    constant: Some(BuiltinValue::Tensor(
                        runmat_value::Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap(),
                    )),
                },
            ],
            var_bindings: HashMap::new(),
            node_bindings: HashMap::new(),
        };
        assert!(graph.folded_literal_numeric_vector(2).is_some());
        graph.var_bindings.insert(
            0,
            VarBinding {
                kind: VarKind::Global,
                index: 9,
            },
        );
        assert!(
            graph.exact_folded_literal_array(2).is_none(),
            "a live child binding is not a source literal"
        );
        assert!(graph.source_literal_array(2).is_none());
        graph.var_bindings.remove(&0);
        assert!(graph.source_literal_array(2).is_some());
        graph.nodes[0].tags.push(AccelGraphTag::Elementwise);
        assert!(graph.folded_literal_numeric_vector(2).is_none());
        graph.nodes[0].tags.pop();
        graph.nodes[0].label = AccelNodeLabel::Primitive(PrimitiveOp::Add);
        assert!(graph.folded_literal_numeric_vector(2).is_none());
        graph.nodes[0].label = AccelNodeLabel::Primitive(PrimitiveOp::UPlus);
        graph.values[2].shape = ShapeInfo::Unknown;
        assert!(graph.folded_literal_numeric_vector(2).is_none());
        graph.values[2].shape = ShapeInfo::Tensor(vec![Some(1), Some(2)]);
        graph.var_bindings.insert(
            2,
            VarBinding {
                kind: VarKind::Global,
                index: 2,
            },
        );
        assert!(graph.exact_folded_literal_array(2).is_some());
        assert!(graph.source_literal_array(2).is_none());
        assert!(graph.folded_literal_numeric_vector(2).is_none());
    }

    #[test]
    fn bound_folded_unary_cast_retains_structural_proof_only() {
        let [variant] = runmat_builtins::SINGLE_CATALOG_ENTRY.bindings else {
            panic!("one canonical Single binding");
        };
        let binding = runmat_builtins::SINGLE_CATALOG_ENTRY.binding_identity(variant);
        let mut graph = AccelGraph {
            nodes: vec![AccelNode {
                id: 0,
                label: AccelNodeLabel::Builtin {
                    name: "single".into(),
                    origin: FusionCallOrigin::Named { binding },
                },
                category: AccelOpCategory::Elementwise,
                inputs: vec![0],
                outputs: vec![1],
                span: InstrSpan { start: 1, end: 1 },
                tags: vec![AccelGraphTag::Elementwise, AccelGraphTag::UnaryScalarFold],
            }],
            values: vec![
                ValueInfo {
                    id: 0,
                    origin: ValueOrigin::Constant,
                    ty: Type::Num,
                    shape: ShapeInfo::Scalar,
                    constant: Some(BuiltinValue::Num(-0.0)),
                },
                ValueInfo {
                    id: 1,
                    origin: ValueOrigin::NodeOutput { node: 0, output: 0 },
                    ty: Type::Tensor {
                        shape: Some(vec![Some(1), Some(1)]),
                    },
                    shape: ShapeInfo::Tensor(vec![Some(1), Some(1)]),
                    constant: Some(BuiltinValue::Tensor(
                        runmat_value::Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap(),
                    )),
                },
            ],
            var_bindings: HashMap::from([(
                1,
                VarBinding {
                    kind: VarKind::Global,
                    index: 1,
                },
            )]),
            node_bindings: HashMap::from([(
                0,
                VarBinding {
                    kind: VarKind::Global,
                    index: 1,
                },
            )]),
        };
        assert!(graph.exact_folded_unary_scalar(1).is_some());
        graph.var_bindings.insert(
            0,
            VarBinding {
                kind: VarKind::Global,
                index: 7,
            },
        );
        assert!(
            graph.exact_folded_unary_scalar(1).is_none(),
            "a live cast input is not a source literal"
        );
        graph.var_bindings.remove(&0);
        graph.values[1].constant = Some(BuiltinValue::Tensor(
            runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap(),
        ));
        assert!(
            graph.exact_folded_unary_scalar(1).is_none(),
            "signed zero is semantic"
        );
    }

    #[test]
    fn test_unify_dims_basic() {
        assert_eq!(
            runmat_builtins::shape_rules::broadcast_shapes(
                &[Some(4), Some(3)],
                &[Some(4), Some(3)]
            ),
            vec![Some(4), Some(3)]
        );
        assert_eq!(
            // Treat vectors as 2D shapes (column-major): 4x1 broadcast with 1x3 -> 4x3.
            runmat_builtins::shape_rules::broadcast_shapes(
                &[Some(4), Some(1)],
                &[Some(1), Some(3)]
            ),
            vec![Some(4), Some(3)]
        );
        assert_eq!(
            runmat_builtins::shape_rules::broadcast_shapes(&[None], &[Some(5)]),
            vec![Some(5)]
        );
        assert_eq!(
            // Treat vectors as 2D shapes (column-major): 2x3 broadcast with 2x1 -> 2x3.
            runmat_builtins::shape_rules::broadcast_shapes(
                &[Some(2), Some(3)],
                &[Some(2), Some(1)]
            ),
            vec![Some(2), Some(3)]
        );
    }

    #[test]
    fn test_shape_unify() {
        let a = ShapeInfo::Tensor(vec![Some(4), Some(3)]);
        let b = ShapeInfo::Scalar;
        assert!(matches!(a.unify(&b), ShapeInfo::Tensor(_)));
    }

    #[test]
    fn test_shape_unify_broadcasts() {
        let a = ShapeInfo::Tensor(vec![Some(1), Some(3)]);
        let b = ShapeInfo::Tensor(vec![Some(2), Some(1)]);
        assert_eq!(a.unify(&b), ShapeInfo::Tensor(vec![Some(2), Some(3)]));
    }
}
