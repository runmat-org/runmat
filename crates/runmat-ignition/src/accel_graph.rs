use std::collections::HashMap;

use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, NodeId,
    PrimitiveOp, ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_builtins::{builtin_functions, AccelTag, Type, Value};

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
    var_bindings: HashMap<ValueId, VarBinding>,
    node_bindings: HashMap<NodeId, VarBinding>,
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
                        AccelTag::ArrayConstruct => AccelGraphTag::ArrayConstruct,
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
            var_bindings: HashMap::new(),
            node_bindings: HashMap::new(),
        }
    }

    fn build(mut self) -> AccelGraph {
        for (pc, instr) in self.instructions.iter().enumerate() {
            self.process_instr(pc, instr);
        }
        AccelGraph {
            nodes: self.nodes,
            values: self.values,
            var_bindings: self.var_bindings,
            node_bindings: self.node_bindings,
        }
    }

    fn process_instr(&mut self, pc: usize, instr: &Instr) {
        match instr {
            Instr::LoadConst(value) => self.push_constant(Type::Num, Some(Value::Num(*value))),
            Instr::LoadComplex(re, im) => {
                self.push_constant(Type::Num, Some(Value::Complex(*re, *im)))
            }
            Instr::LoadBool(value) => self.push_constant(Type::Bool, Some(Value::Bool(*value))),
            Instr::LoadString(s) => {
                self.push_constant(Type::String, Some(Value::String(s.clone())))
            }
            Instr::LoadCharRow(s) => {
                self.push_constant(Type::String, Some(Value::String(s.clone())))
            }
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
            Instr::Transpose | Instr::ConjugateTranspose => self.handle_transpose(pc),
            Instr::CallBuiltin(name, argc) => self.handle_call_builtin(pc, name, *argc),
            Instr::Pop => {
                let _ = self.pop_value();
            }
            Instr::CreateMatrix(rows, cols) => {
                let rows = *rows;
                let cols = *cols;
                let total = rows * cols;
                if self.stack.len() < total {
                    self.reset_stack();
                    return;
                }
                // Pop total elements (column-major expectation). We'll reconstruct column-major data.
                let mut elems: Vec<f64> = Vec::with_capacity(total);
                for _ in 0..total {
                    if let Some(id) = self.pop_value() {
                        let info = &self.values[id as usize];
                        let val = match &info.constant {
                            Some(Value::Num(n)) => *n,
                            Some(Value::Int(i)) => i.to_f64(),
                            _ => {
                                elems.clear();
                                break;
                            }
                        };
                        elems.push(val);
                    } else {
                        elems.clear();
                        break;
                    }
                }
                let node_id = self.nodes.len() as NodeId;
                let span = InstrSpan { start: pc, end: pc };
                let mut node = AccelNode {
                    id: node_id,
                    label: AccelNodeLabel::Primitive(PrimitiveOp::UPlus),
                    category: AccelOpCategory::Other,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    span,
                    tags: vec![],
                };
                let ty = Type::Tensor {
                    shape: Some(vec![Some(rows), Some(cols)]),
                };
                // Column-major data: elements were popped last-in-first-out; reverse to original push order for row vector cases
                let tensor_const = if elems.len() == total {
                    elems.reverse();
                    // For generality, map from row-major order in elems to column-major storage expected
                    // Assuming elems are in left-to-right, top-to-bottom order, convert to column-major
                    let mut data_cm = vec![0.0f64; total];
                    for r in 0..rows {
                        for c in 0..cols {
                            let idx_row_major = r * cols + c;
                            let idx_col_major = r + c * rows;
                            data_cm[idx_col_major] = elems[idx_row_major];
                        }
                    }
                    runmat_builtins::Tensor::new(data_cm, vec![rows, cols]).ok()
                } else {
                    None
                };
                let out_value = if let Some(t) = tensor_const {
                    self.new_value(
                        ValueOrigin::NodeOutput {
                            node: node_id,
                            output: 0,
                        },
                        ty.clone(),
                        Some(Value::Tensor(t)),
                    )
                } else {
                    self.new_node_output(node_id, 0, ty)
                };
                node.outputs.push(out_value);
                self.nodes.push(node);
                self.stack.push(out_value);
            }
            Instr::Swap => {
                if self.stack.len() >= 2 {
                    let len = self.stack.len();
                    self.stack.swap(len - 1, len - 2);
                } else {
                    self.reset_stack();
                }
            }
            // Control-flow breaks straight-line analysis. Invalidate both the operand stack and
            // our variable/value cache so we don't incorrectly treat loop induction variables
            // (e.g. `t` in `for t = 0:dt:T`) as compile-time constants inside the loop body.
            Instr::Jump(_) | Instr::JumpIfFalse(_) | Instr::Return | Instr::ReturnValue => {
                self.reset_control_flow_state();
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
            None,
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
            self.var_bindings.insert(
                value_id,
                VarBinding {
                    kind: VarKind::Global,
                    index: idx,
                },
            );
            self.record_node_binding(value_id, VarKind::Global, idx);
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
            None,
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
            self.var_bindings.insert(
                value_id,
                VarBinding {
                    kind: VarKind::Local,
                    index: idx,
                },
            );
            self.record_node_binding(value_id, VarKind::Local, idx);
        } else {
            self.reset_stack();
        }
    }

    fn record_node_binding(&mut self, value_id: ValueId, kind: VarKind, index: usize) {
        if let Some(info) = self.values.get(value_id as usize) {
            if let ValueOrigin::NodeOutput { node, .. } = info.origin {
                self.node_bindings.insert(node, VarBinding { kind, index });
            }
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
        let mut out_type = match info.category {
            AccelOpCategory::Elementwise => self.infer_elementwise_shape(&inputs).to_type(),
            AccelOpCategory::Reduction => Type::Num,
            AccelOpCategory::MatMul => self.infer_matmul_type(&inputs),
            AccelOpCategory::Transpose => Type::Unknown,
            AccelOpCategory::Other => {
                // Prefer tag-driven array construct inference over name-based
                if info
                    .tags
                    .iter()
                    .any(|t| matches!(t, AccelGraphTag::ArrayConstruct))
                {
                    self.infer_array_constructor_from_tags(&inputs)
                        .unwrap_or(Type::Unknown)
                } else {
                    Type::Unknown
                }
            }
        };
        // If still unknown for an ArrayConstruct-tagged builtin, try a constant size-vector inference
        if matches!(out_type, Type::Unknown)
            && info
                .tags
                .iter()
                .any(|t| matches!(t, AccelGraphTag::ArrayConstruct))
            && inputs.len() == 1
        {
            if let Some(info0) = self.values.get(inputs[0] as usize) {
                if let Some(Value::Tensor(t)) = &info0.constant {
                    let rows = t.rows();
                    let cols = t.cols();
                    if rows == 1 && cols > 0 {
                        let mut dims: Vec<Option<usize>> = Vec::with_capacity(cols);
                        for j in 0..cols {
                            dims.push(Some(t.data[j].round() as usize));
                        }
                        out_type = Type::Tensor { shape: Some(dims) };
                    } else if cols == 1 && rows > 0 {
                        let mut dims: Vec<Option<usize>> = Vec::with_capacity(rows);
                        for i in 0..rows {
                            dims.push(Some(t.data[i].round() as usize));
                        }
                        out_type = Type::Tensor { shape: Some(dims) };
                    }
                }
            }
        }
        let out_value = self.new_node_output(node_id, 0, out_type);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
        self.maybe_fold_builtin_constant(name, &inputs, out_value);
    }

    fn infer_array_constructor_from_tags(&self, inputs: &[ValueId]) -> Option<Type> {
        // Generic version of infer_array_constructor_type without any name checks
        // 1) 'like' prototype takes precedence
        let mut i = 0usize;
        while i + 1 < inputs.len() {
            let s = &self.values[inputs[i] as usize].constant;
            if let Some(Value::String(text)) = s {
                if text.eq_ignore_ascii_case("like") {
                    let proto = &self.values[inputs[i + 1] as usize];
                    if let Type::Tensor { shape: Some(dims) } = &proto.ty {
                        return Some(Type::Tensor {
                            shape: Some(dims.clone()),
                        });
                    }
                    break;
                }
            }
            i += 1;
        }

        // 2) Leading numeric dims
        let mut dims: Vec<Option<usize>> = Vec::new();
        for &vid in inputs {
            let Some(info) = self.values.get(vid as usize) else {
                break;
            };
            match &info.constant {
                Some(Value::Num(n)) => {
                    if n.is_finite() {
                        let r = n.round();
                        if (r - n).abs() <= f64::EPSILON && r >= 0.0 {
                            dims.push(Some(r as usize));
                            continue;
                        }
                    }
                    break;
                }
                Some(Value::Int(i)) => {
                    let v = i.to_i64();
                    if v >= 0 {
                        dims.push(Some(v as usize));
                        continue;
                    }
                    break;
                }
                Some(Value::String(_)) => break,
                _ => break,
            }
        }
        if !dims.is_empty() {
            if dims.len() == 1 {
                let n = dims[0].unwrap_or(0);
                return Some(Type::Tensor {
                    shape: Some(vec![Some(n), Some(n)]),
                });
            }
            return Some(Type::Tensor { shape: Some(dims) });
        }

        // 3) size-vector tensor argument: 1xN or Nx1 with concrete numeric dims
        for &vid in inputs {
            let Some(info) = self.values.get(vid as usize) else {
                continue;
            };
            match &info.ty {
                Type::Tensor { shape: Some(dims) } if dims.len() == 2 => {
                    // Expect a constant tensor encoded as Value::Tensor; pull from constant if available
                    if let Some(Value::Tensor(t)) = &info.constant {
                        let rows = t.rows();
                        let cols = t.cols();
                        if (rows == 1 || rows == 0) && cols > 0 {
                            // 1xN row vector
                            let mut out: Vec<Option<usize>> = Vec::with_capacity(cols);
                            for j in 0..cols {
                                let v = t.data[j].round() as i64;
                                if v >= 0 {
                                    out.push(Some(v as usize));
                                } else {
                                    out.push(None);
                                }
                            }
                            return Some(Type::Tensor { shape: Some(out) });
                        } else if (cols == 1 || cols == 0) && rows > 0 {
                            // Nx1 column vector
                            let mut out: Vec<Option<usize>> = Vec::with_capacity(rows);
                            for i in 0..rows {
                                let v = t.data[i].round() as i64;
                                if v >= 0 {
                                    out.push(Some(v as usize));
                                } else {
                                    out.push(None);
                                }
                            }
                            return Some(Type::Tensor { shape: Some(out) });
                        }
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn push_constant(&mut self, ty: Type, constant: Option<Value>) {
        let id = self.new_value(ValueOrigin::Constant, ty, constant);
        self.stack.push(id);
    }

    fn pop_value(&mut self) -> Option<ValueId> {
        self.stack.pop()
    }

    fn new_value(&mut self, origin: ValueOrigin, ty: Type, constant: Option<Value>) -> ValueId {
        let id = self.values.len() as ValueId;
        let shape = ShapeInfo::from_type(&ty);
        self.values.push(ValueInfo {
            id,
            origin,
            ty,
            shape,
            constant,
        });
        id
    }

    fn new_node_output(&mut self, node: NodeId, output: usize, ty: Type) -> ValueId {
        self.new_value(ValueOrigin::NodeOutput { node, output }, ty, None)
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
                    let m = sa.first().cloned().unwrap_or(None);
                    let k_left = sa.get(1).cloned().unwrap_or(None);
                    let k_right = sb.first().cloned().unwrap_or(None);
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
        // Unknown instructions can mutate state in ways we don't model. Be conservative.
        self.reset_control_flow_state();
    }

    fn reset_stack(&mut self) {
        self.stack.clear();
    }

    fn reset_control_flow_state(&mut self) {
        self.reset_stack();
        self.var_values.clear();
        self.local_types.clear();
    }

    fn maybe_fold_builtin_constant(&mut self, name: &str, inputs: &[ValueId], out_value: ValueId) {
        if !name.eq_ignore_ascii_case("single") {
            return;
        }
        if inputs.len() != 1 {
            return;
        }
        let Some(input_info) = self.values.get(inputs[0] as usize) else {
            return;
        };
        let Some(constant) = &input_info.constant else {
            return;
        };
        let folded = match constant {
            Value::Num(n) => Some(Value::Num((*n as f32) as f64)),
            Value::Int(i) => Some(Value::Num((i.to_f64() as f32) as f64)),
            Value::Bool(flag) => Some(Value::Num(if *flag { 1.0f32 } else { 0.0f32 } as f64)),
            _ => None,
        };
        if let Some(value) = folded {
            if let Some(out_info) = self.values.get_mut(out_value as usize) {
                out_info.constant = Some(value);
            }
        }
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
