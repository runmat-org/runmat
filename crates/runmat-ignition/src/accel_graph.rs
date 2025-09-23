#![cfg(feature = "native-accel")]

use std::collections::HashMap;

use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, NodeId,
    PrimitiveOp, ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarKind,
};
use runmat_builtins::{builtin_functions, AccelTag, Type};

use crate::instr::Instr;

pub fn build_accel_graph(instructions: &[Instr], var_types: &[Type]) -> AccelGraph {
    GraphBuilder::new(instructions, var_types).build()
}

struct GraphBuilder<'a> {
    instructions: &'a [Instr],
    var_types: &'a [Type],
    nodes: Vec<AccelNode>,
    values: Vec<ValueInfo>,
    stack: Vec<ValueId>,
    var_values: HashMap<VarKey, ValueId>,
    local_types: HashMap<usize, Type>,
    builtin_cache: HashMap<String, BuiltinInfo>,
}

#[derive(Clone)]
struct BuiltinInfo {
    category: AccelOpCategory,
    tags: Vec<AccelGraphTag>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VarKey {
    Global(usize),
    Local(usize),
}

impl<'a> GraphBuilder<'a> {
    fn new(instructions: &'a [Instr], var_types: &'a [Type]) -> Self {
        let builtin_cache = builtin_functions()
            .iter()
            .map(|func| {
                let name = func.name.to_ascii_lowercase();
                let tags: Vec<AccelGraphTag> = func
                    .accel_tags
                    .iter()
                    .map(|tag| match tag {
                        AccelTag::Unary => AccelGraphTag::Unary,
                        AccelTag::Elementwise => AccelGraphTag::Elementwise,
                        AccelTag::Reduction => AccelGraphTag::Reduction,
                        AccelTag::MatMul => AccelGraphTag::MatMul,
                        AccelTag::Transpose => AccelGraphTag::Transpose,
                    })
                    .collect();
                let category = categorize_builtin(&tags);
                (name, BuiltinInfo { category, tags })
            })
            .collect();
        Self {
            instructions,
            var_types,
            nodes: Vec::new(),
            values: Vec::new(),
            stack: Vec::new(),
            var_values: HashMap::new(),
            local_types: HashMap::new(),
            builtin_cache,
        }
    }

    fn build(mut self) -> AccelGraph {
        for (pc, instr) in self.instructions.iter().enumerate() {
            self.process_instr(pc, instr);
        }
        AccelGraph {
            nodes: self.nodes,
            values: self.values,
        }
    }

    fn process_instr(&mut self, pc: usize, instr: &Instr) {
        match instr {
            Instr::LoadConst(_) => self.push_constant(Type::Num),
            Instr::LoadBool(_) => self.push_constant(Type::Bool),
            Instr::LoadString(_) | Instr::LoadCharRow(_) => self.push_constant(Type::String),
            Instr::LoadVar(idx) => self.handle_load_var(*idx),
            Instr::StoreVar(idx) => self.handle_store_var(*idx),
            Instr::LoadLocal(idx) => self.handle_load_local(*idx),
            Instr::StoreLocal(idx) => self.handle_store_local(*idx),
            Instr::Add => self.handle_binary_primitive(pc, PrimitiveOp::Add),
            Instr::Sub => self.handle_binary_primitive(pc, PrimitiveOp::Sub),
            Instr::Mul => self.handle_binary_primitive(pc, PrimitiveOp::Mul),
            Instr::Div => self.handle_binary_primitive(pc, PrimitiveOp::Div),
            Instr::Pow => self.handle_binary_primitive(pc, PrimitiveOp::Pow),
            Instr::ElemMul => self.handle_binary_primitive(pc, PrimitiveOp::ElemMul),
            Instr::ElemDiv => self.handle_binary_primitive(pc, PrimitiveOp::ElemDiv),
            Instr::ElemPow => self.handle_binary_primitive(pc, PrimitiveOp::ElemPow),
            Instr::ElemLeftDiv => self.handle_binary_primitive(pc, PrimitiveOp::ElemLeftDiv),
            Instr::LessEqual => self.handle_binary_primitive(pc, PrimitiveOp::LessEqual),
            Instr::Less => self.handle_binary_primitive(pc, PrimitiveOp::Less),
            Instr::Greater => self.handle_binary_primitive(pc, PrimitiveOp::Greater),
            Instr::GreaterEqual => self.handle_binary_primitive(pc, PrimitiveOp::GreaterEqual),
            Instr::Equal => self.handle_binary_primitive(pc, PrimitiveOp::Equal),
            Instr::NotEqual => self.handle_binary_primitive(pc, PrimitiveOp::NotEqual),
            Instr::Neg => self.handle_unary_primitive(pc, PrimitiveOp::Neg),
            Instr::UPlus => self.handle_unary_primitive(pc, PrimitiveOp::UPlus),
            Instr::Transpose => self.handle_transpose(pc),
            Instr::CallBuiltin(name, argc) => self.handle_call_builtin(pc, name, *argc),
            Instr::Pop => {
                let _ = self.pop_value();
            }
            Instr::Swap => {
                if self.stack.len() >= 2 {
                    let len = self.stack.len();
                    self.stack.swap(len - 1, len - 2);
                } else {
                    self.reset_stack();
                }
            }
            Instr::Jump(_) | Instr::JumpIfFalse(_) | Instr::Return | Instr::ReturnValue => {
                self.reset_stack();
            }
            _ => self.mark_unknown(),
        }
    }

    fn handle_load_var(&mut self, idx: usize) {
        let key = VarKey::Global(idx);
        if let Some(id) = self.var_values.get(&key).copied() {
            self.stack.push(id);
            return;
        }
        let ty = self.var_types.get(idx).cloned().unwrap_or(Type::Unknown);
        let id = self.new_value(
            ValueOrigin::Variable {
                kind: VarKind::Global,
                index: idx,
            },
            ty,
        );
        self.var_values.insert(key, id);
        self.stack.push(id);
    }

    fn handle_store_var(&mut self, idx: usize) {
        if let Some(value_id) = self.pop_value() {
            let key = VarKey::Global(idx);
            self.var_values.insert(key, value_id);
            if let Some(ty) = self.var_types.get(idx) {
                self.apply_type(value_id, ty);
            }
        } else {
            self.reset_stack();
        }
    }

    fn handle_load_local(&mut self, idx: usize) {
        let key = VarKey::Local(idx);
        if let Some(id) = self.var_values.get(&key).copied() {
            self.stack.push(id);
            return;
        }
        let ty = self.local_types.get(&idx).cloned().unwrap_or(Type::Unknown);
        let id = self.new_value(
            ValueOrigin::Variable {
                kind: VarKind::Local,
                index: idx,
            },
            ty,
        );
        self.var_values.insert(key, id);
        self.stack.push(id);
    }

    fn handle_store_local(&mut self, idx: usize) {
        if let Some(value_id) = self.pop_value() {
            let key = VarKey::Local(idx);
            self.var_values.insert(key, value_id);
            let ty = self
                .values
                .get(value_id as usize)
                .map(|info| info.ty.clone());
            if let Some(ty) = ty {
                self.local_types
                    .entry(idx)
                    .and_modify(|existing| *existing = existing.unify(&ty))
                    .or_insert(ty);
            }
        } else {
            self.reset_stack();
        }
    }

    fn handle_binary_primitive(&mut self, pc: usize, op: PrimitiveOp) {
        let rhs = match self.pop_value() {
            Some(id) => id,
            None => {
                self.reset_stack();
                return;
            }
        };
        let lhs = match self.pop_value() {
            Some(id) => id,
            None => {
                self.reset_stack();
                return;
            }
        };
        let inputs = vec![lhs, rhs];
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let shape = self.infer_elementwise_shape(&inputs);
        let out_type = if matches!(shape, ShapeInfo::Unknown) {
            Type::Unknown
        } else {
            shape.to_type()
        };
        let mut node = AccelNode {
            id: node_id,
            label: AccelNodeLabel::Primitive(op),
            category: primitive_category(op),
            inputs: inputs.clone(),
            outputs: Vec::new(),
            span,
            tags: primitive_tags(op),
        };
        let out_value = self.new_node_output(node_id, 0, out_type);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
    }

    fn handle_unary_primitive(&mut self, pc: usize, op: PrimitiveOp) {
        let input = match self.pop_value() {
            Some(id) => id,
            None => {
                self.reset_stack();
                return;
            }
        };
        let inputs = vec![input];
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let shape = self.infer_elementwise_shape(&inputs);
        let out_type = if matches!(shape, ShapeInfo::Unknown) {
            Type::Unknown
        } else {
            shape.to_type()
        };
        let mut node = AccelNode {
            id: node_id,
            label: AccelNodeLabel::Primitive(op),
            category: primitive_category(op),
            inputs,
            outputs: Vec::new(),
            span,
            tags: primitive_tags(op),
        };
        let out_value = self.new_node_output(node_id, 0, out_type);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
    }

    fn handle_transpose(&mut self, pc: usize) {
        let input = match self.pop_value() {
            Some(id) => id,
            None => {
                self.reset_stack();
                return;
            }
        };
        let inputs = vec![input];
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let mut node = AccelNode {
            id: node_id,
            label: AccelNodeLabel::Primitive(PrimitiveOp::Transpose),
            category: AccelOpCategory::Transpose,
            inputs,
            outputs: Vec::new(),
            span,
            tags: vec![AccelGraphTag::Transpose],
        };
        let out_value = self.new_node_output(node_id, 0, Type::Unknown);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
    }

    fn handle_call_builtin(&mut self, pc: usize, name: &str, argc: usize) {
        let mut inputs = Vec::with_capacity(argc);
        for _ in 0..argc {
            if let Some(id) = self.pop_value() {
                inputs.push(id);
            } else {
                self.reset_stack();
                return;
            }
        }
        inputs.reverse();
        let info = self
            .builtin_cache
            .get(&name.to_ascii_lowercase())
            .cloned()
            .unwrap_or(BuiltinInfo {
                category: AccelOpCategory::Other,
                tags: Vec::new(),
            });
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let mut node = AccelNode {
            id: node_id,
            label: AccelNodeLabel::Builtin {
                name: name.to_string(),
            },
            category: info.category.clone(),
            inputs: inputs.clone(),
            outputs: Vec::new(),
            span,
            tags: info.tags.clone(),
        };
        let out_type = match info.category {
            AccelOpCategory::Elementwise => self.infer_elementwise_shape(&inputs).to_type(),
            AccelOpCategory::Reduction => Type::Num,
            AccelOpCategory::MatMul => self.infer_matmul_type(&inputs),
            AccelOpCategory::Transpose => Type::Unknown,
            AccelOpCategory::Other => Type::Unknown,
        };
        let out_value = self.new_node_output(node_id, 0, out_type);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
    }

    fn push_constant(&mut self, ty: Type) {
        let id = self.new_value(ValueOrigin::Constant, ty);
        self.stack.push(id);
    }

    fn pop_value(&mut self) -> Option<ValueId> {
        self.stack.pop()
    }

    fn new_value(&mut self, origin: ValueOrigin, ty: Type) -> ValueId {
        let id = self.values.len() as ValueId;
        let shape = ShapeInfo::from_type(&ty);
        self.values.push(ValueInfo {
            id,
            origin,
            ty,
            shape,
        });
        id
    }

    fn new_node_output(&mut self, node: NodeId, output: usize, ty: Type) -> ValueId {
        self.new_value(ValueOrigin::NodeOutput { node, output }, ty)
    }

    fn apply_type(&mut self, value_id: ValueId, ty: &Type) {
        if let Some(info) = self.values.get_mut(value_id as usize) {
            info.update_type(ty);
        }
    }

    fn infer_elementwise_shape(&self, inputs: &[ValueId]) -> ShapeInfo {
        let mut shape = ShapeInfo::Scalar;
        for &input in inputs {
            let info = match self.values.get(input as usize) {
                Some(info) => info,
                None => return ShapeInfo::Unknown,
            };
            shape = shape.unify(&info.shape);
            if matches!(shape, ShapeInfo::Unknown) {
                break;
            }
        }
        shape
    }

    fn infer_matmul_type(&self, inputs: &[ValueId]) -> Type {
        if inputs.len() != 2 {
            return Type::Unknown;
        }
        let lhs = self.values.get(inputs[0] as usize);
        let rhs = self.values.get(inputs[1] as usize);
        match (lhs, rhs) {
            (Some(a), Some(b)) => match (&a.shape, &b.shape) {
                (ShapeInfo::Tensor(sa), ShapeInfo::Tensor(sb))
                    if sa.len() >= 2 && sb.len() >= 2 =>
                {
                    let m = sa.get(0).cloned().unwrap_or(None);
                    let k_left = sa.get(1).cloned().unwrap_or(None);
                    let k_right = sb.get(0).cloned().unwrap_or(None);
                    let n = sb.get(1).cloned().unwrap_or(None);
                    let dims = if let (Some(kl), Some(kr)) = (k_left, k_right) {
                        if kl != kr && kl != 1 && kr != 1 {
                            vec![m, None]
                        } else {
                            vec![m, n]
                        }
                    } else {
                        vec![m, n]
                    };
                    Type::Tensor { shape: Some(dims) }
                }
                _ => Type::Unknown,
            },
            _ => Type::Unknown,
        }
    }

    fn mark_unknown(&mut self) {
        self.reset_stack();
    }

    fn reset_stack(&mut self) {
        self.stack.clear();
    }
}

fn categorize_builtin(tags: &[AccelGraphTag]) -> AccelOpCategory {
    if tags.iter().any(|t| matches!(t, AccelGraphTag::MatMul)) {
        AccelOpCategory::MatMul
    } else if tags.iter().any(|t| matches!(t, AccelGraphTag::Reduction)) {
        AccelOpCategory::Reduction
    } else if tags
        .iter()
        .any(|t| matches!(t, AccelGraphTag::Elementwise | AccelGraphTag::Unary))
    {
        AccelOpCategory::Elementwise
    } else if tags.iter().any(|t| matches!(t, AccelGraphTag::Transpose)) {
        AccelOpCategory::Transpose
    } else {
        AccelOpCategory::Other
    }
}

fn primitive_category(op: PrimitiveOp) -> AccelOpCategory {
    match op {
        PrimitiveOp::Transpose => AccelOpCategory::Transpose,
        PrimitiveOp::Less
        | PrimitiveOp::LessEqual
        | PrimitiveOp::Greater
        | PrimitiveOp::GreaterEqual
        | PrimitiveOp::Equal
        | PrimitiveOp::NotEqual
        | PrimitiveOp::Add
        | PrimitiveOp::Sub
        | PrimitiveOp::Mul
        | PrimitiveOp::Div
        | PrimitiveOp::Pow
        | PrimitiveOp::Neg
        | PrimitiveOp::UPlus
        | PrimitiveOp::ElemMul
        | PrimitiveOp::ElemDiv
        | PrimitiveOp::ElemPow
        | PrimitiveOp::ElemLeftDiv => AccelOpCategory::Elementwise,
    }
}

fn primitive_tags(op: PrimitiveOp) -> Vec<AccelGraphTag> {
    match op {
        PrimitiveOp::Transpose => vec![AccelGraphTag::Transpose],
        PrimitiveOp::Neg | PrimitiveOp::UPlus => {
            vec![AccelGraphTag::Unary, AccelGraphTag::Elementwise]
        }
        _ => vec![AccelGraphTag::Elementwise],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instr::Instr;

    #[test]
    fn builds_simple_elementwise_graph() {
        let instrs = vec![
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::ElemMul,
            Instr::StoreVar(2),
        ];
        let mut var_types = vec![Type::tensor(), Type::tensor(), Type::tensor()];
        if let Type::Tensor { shape } = &mut var_types[0] {
            *shape = Some(vec![Some(4), Some(4)]);
        }
        if let Type::Tensor { shape } = &mut var_types[1] {
            *shape = Some(vec![Some(4), Some(4)]);
        }
        if let Type::Tensor { shape } = &mut var_types[2] {
            *shape = Some(vec![Some(4), Some(4)]);
        }
        let graph = build_accel_graph(&instrs, &var_types);
        assert_eq!(graph.nodes.len(), 1);
        let node = &graph.nodes[0];
        assert!(node.is_elementwise());
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
    }
}
