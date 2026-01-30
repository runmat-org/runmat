use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use runmat_builtins::{Type, Value as BuiltinValue};

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
    Builtin { name: String },
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrimitiveOp {
    Add,
    Sub,
    Mul,
    Div,
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

impl fmt::Display for PrimitiveOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            PrimitiveOp::Add => "Add",
            PrimitiveOp::Sub => "Sub",
            PrimitiveOp::Mul => "Mul",
            PrimitiveOp::Div => "Div",
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
            (ShapeInfo::Tensor(a), ShapeInfo::Tensor(b)) => ShapeInfo::Tensor(unify_dims(a, b)),
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

fn unify_dims(a: &[Option<usize>], b: &[Option<usize>]) -> Vec<Option<usize>> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let da = a.get(i).cloned().unwrap_or(None);
        let db = b.get(i).cloned().unwrap_or(None);
        let dim = match (da, db) {
            (Some(x), Some(y)) if x == y => Some(x),
            (Some(1), Some(y)) => Some(y),
            (Some(x), Some(1)) => Some(x),
            (Some(x), Some(y)) if x != y => None,
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
            _ => None,
        };
        result.push(dim);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{unify_dims, ShapeInfo};

    #[test]
    fn test_unify_dims_basic() {
        assert_eq!(
            unify_dims(&[Some(4), Some(3)], &[Some(4), Some(3)]),
            vec![Some(4), Some(3)]
        );
        assert_eq!(
            unify_dims(&[Some(4)], &[Some(1), Some(3)]),
            vec![Some(4), Some(3)]
        );
        assert_eq!(unify_dims(&[None], &[Some(5)]), vec![Some(5)]);
        assert_eq!(
            unify_dims(&[Some(2), Some(3)], &[Some(2)]),
            vec![Some(2), Some(3)]
        );
    }

    #[test]
    fn test_shape_unify() {
        let a = ShapeInfo::Tensor(vec![Some(4), Some(3)]);
        let b = ShapeInfo::Scalar;
        assert!(matches!(a.unify(&b), ShapeInfo::Tensor(_)));
    }
}
