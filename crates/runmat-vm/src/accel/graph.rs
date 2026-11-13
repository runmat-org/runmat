use std::collections::{HashMap, HashSet};

use runmat_accelerate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, InstrSpan, NodeId,
    PrimitiveOp, ShapeInfo, ValueId, ValueInfo, ValueOrigin, VarBinding, VarKind,
};
use runmat_builtins::{
    builtin_functions, shape_rules::element_count_if_known, AccelTag, ArrayCreationInferenceRule,
    ArrayInferenceRule, BuiltinInferenceRule, Type,
};
use runmat_value::Value;

use crate::instr::Instr;

#[path = "graph/aggregate.rs"]
mod aggregate;

pub fn build_accel_graph(instructions: &[Instr], var_types: &[Type]) -> AccelGraph {
    GraphBuilder::new(instructions, var_types).build()
}

/// Execution-order store transcript from the same bytecode walk that creates
/// graph value IDs. A value may be stored more than once, so the graph's
/// `ValueId -> VarBinding` map is not an ordered publication authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AccelStore {
    pub pc: usize,
    pub value_id: ValueId,
    pub binding: VarBinding,
}

/// A load occurrence from the same instruction walk as the graph value IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AccelLoad {
    pub pc: usize,
    pub value_id: ValueId,
    pub binding: VarBinding,
}

pub(crate) fn build_accel_graph_with_transcript(
    instructions: &[Instr],
    var_types: &[Type],
) -> (AccelGraph, Vec<AccelStore>, Vec<AccelLoad>) {
    GraphBuilder::new(instructions, var_types).build_with_transcript()
}

pub(crate) fn build_accel_graph_with_stores(
    instructions: &[Instr],
    var_types: &[Type],
) -> (AccelGraph, Vec<AccelStore>) {
    GraphBuilder::new(instructions, var_types).build_with_stores()
}

#[cfg(test)]
#[path = "graph/folded_scalar_tests.rs"]
mod folded_scalar_tests;

struct GraphBuilder<'a> {
    instructions: &'a [Instr],
    var_types: &'a [Type],
    nodes: Vec<AccelNode>,
    values: Vec<ValueInfo>,
    stack: Vec<ValueId>,
    var_values: HashMap<VarKey, ValueId>,
    creation_invalidated: HashSet<usize>,
    local_types: HashMap<usize, Type>,
    builtin_cache: HashMap<String, BuiltinInfo>,
    var_bindings: HashMap<ValueId, VarBinding>,
    node_bindings: HashMap<NodeId, VarBinding>,
    stores: Option<Vec<AccelStore>>,
    loads: Vec<AccelLoad>,
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
        let mut builtin_cache: HashMap<_, _> = builtin_functions()
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
        // Canonical placement and selected expression declarations supersede
        // legacy AccelTags. Inference categories do not authorize fusion.
        for entry in runmat_builtins::builtin_catalog_entries() {
            let mut info = BuiltinInfo {
                category: AccelOpCategory::Other,
                tags: Vec::new(),
            };
            // Constructor shape provenance is independent of executable
            // fusion admission. The canonical inference rule, not a legacy
            // acceleration tag, owns this structural analysis hint.
            if canonical_array_constructor_shape_tag(entry.contract.inference_rule) {
                info.tags.push(AccelGraphTag::ArrayConstruct);
            }
            match super::fusion_declaration::canonical_kind(entry, None) {
                Some(crate::bytecode::FusionInstructionKind::Elementwise) => {
                    info.category = AccelOpCategory::Elementwise;
                    info.tags.push(AccelGraphTag::Elementwise);
                }
                Some(crate::bytecode::FusionInstructionKind::Reduction) => {
                    info.category = AccelOpCategory::Reduction;
                    info.tags.push(AccelGraphTag::Reduction);
                }
                Some(crate::bytecode::FusionInstructionKind::Matmul) => {
                    info.category = AccelOpCategory::MatMul;
                    info.tags.push(AccelGraphTag::MatMul);
                }
                None => {}
            }
            builtin_cache.insert(entry.identity.name.to_ascii_lowercase(), info);
        }
        Self {
            instructions,
            var_types,
            nodes: Vec::new(),
            values: Vec::new(),
            stack: Vec::new(),
            var_values: HashMap::new(),
            creation_invalidated: HashSet::new(),
            local_types: HashMap::new(),
            builtin_cache,
            var_bindings: HashMap::new(),
            node_bindings: HashMap::new(),
            stores: None,
            loads: Vec::new(),
        }
    }

    fn build(mut self) -> AccelGraph {
        for (pc, instr) in self.instructions.iter().enumerate() {
            self.process_instr(pc, instr);
        }
        self.finish_graph()
    }

    fn build_with_stores(self) -> (AccelGraph, Vec<AccelStore>) {
        let (graph, stores, _) = self.build_with_transcript();
        (graph, stores)
    }

    fn build_with_transcript(mut self) -> (AccelGraph, Vec<AccelStore>, Vec<AccelLoad>) {
        self.stores = Some(Vec::new());
        for (pc, instr) in self.instructions.iter().enumerate() {
            self.process_instr(pc, instr);
        }
        let stores = self.stores.take().expect("store recording selected");
        let loads = std::mem::take(&mut self.loads);
        (self.finish_graph(), stores, loads)
    }

    fn finish_graph(self) -> AccelGraph {
        AccelGraph {
            nodes: self.nodes,
            values: self.values,
            var_bindings: self.var_bindings,
            node_bindings: self.node_bindings,
        }
    }

    fn process_instr(&mut self, pc: usize, instr: &Instr) {
        self.process_instruction(pc, instr);
        if self.stores.is_none() {
            return;
        }
        let binding = match instr {
            Instr::LoadVar(index) => VarBinding {
                kind: VarKind::Global,
                index: *index,
            },
            Instr::LoadLocal(index) => VarBinding {
                kind: VarKind::Local,
                index: *index,
            },
            _ => return,
        };
        if let Some(&value_id) = self.stack.last() {
            self.loads.push(AccelLoad {
                pc,
                value_id,
                binding,
            });
        }
    }

    fn process_instruction(&mut self, pc: usize, instr: &Instr) {
        match instr {
            Instr::LoadConst(value) => self.push_constant(Type::Num, Some(Value::Num(*value))),
            Instr::LoadInt(value) => self.push_constant(Type::Int, Some(Value::Int(value.clone()))),
            Instr::LoadComplex(re, im) => {
                self.push_constant(Type::Num, Some(Value::Complex(*re, *im)))
            }
            Instr::LoadBool(value) => self.push_constant(Type::Bool, Some(Value::Bool(*value))),
            Instr::LoadString(s) => {
                self.push_constant(Type::String, Some(Value::String(s.clone().into())))
            }
            Instr::LoadCharRow(s) => {
                self.push_constant(Type::String, Some(Value::String(s.clone().into())))
            }
            Instr::LoadVar(idx) => self.handle_load_var(*idx),
            Instr::LoadVarForIndexAssignment(idx)
            | Instr::LoadVarForCellAssignment(idx)
            | Instr::LoadVarForStructAssignment(idx) => {
                self.mark_unknown();
                self.creation_invalidated.insert(*idx);
            }
            Instr::StoreVar(idx) => self.handle_store_var(pc, *idx),
            Instr::LoadLocal(idx) => self.handle_load_local(*idx),
            Instr::StoreLocal(idx) => self.handle_store_local(pc, *idx),
            Instr::Add => self.handle_binary_primitive(pc, PrimitiveOp::Add),
            Instr::Sub => self.handle_binary_primitive(pc, PrimitiveOp::Sub),
            Instr::Mul => self.handle_binary_primitive(pc, PrimitiveOp::Mul),
            Instr::RightDiv => self.handle_right_divide(pc),
            Instr::LeftDiv => self.handle_left_divide(pc),
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
            Instr::CallBuiltinMulti(name, argc, outputs) => {
                self.handle_call_builtin(pc, name, *argc, *outputs)
            }
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
                let mut element_ids = Vec::with_capacity(total);
                let mut foldable = true;
                for _ in 0..total {
                    if let Some(id) = self.pop_value() {
                        element_ids.push(id);
                        let info = &self.values[id as usize];
                        let val = match &info.constant {
                            Some(Value::Num(n)) => *n,
                            // A Tensor constant has f64 storage.  Keep native
                            // integer literals out of this acceleration-only
                            // fold instead of silently rounding them.
                            Some(Value::Int(_)) => {
                                foldable = false;
                                continue;
                            }
                            _ => {
                                foldable = false;
                                continue;
                            }
                        };
                        elems.push(val);
                    } else {
                        elems.clear();
                        break;
                    }
                }
                element_ids.reverse();
                let node_id = self.nodes.len() as NodeId;
                let span = InstrSpan { start: pc, end: pc };
                let mut node = AccelNode {
                    id: node_id,
                    label: AccelNodeLabel::Primitive(PrimitiveOp::UPlus),
                    category: AccelOpCategory::Other,
                    inputs: element_ids,
                    outputs: Vec::new(),
                    span,
                    tags: vec![],
                };
                let ty = Type::Tensor {
                    shape: Some(vec![Some(rows), Some(cols)]),
                };
                // Column-major data: elements were popped last-in-first-out; reverse to original push order for row vector cases
                let tensor_const = if foldable && elems.len() == total {
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
                    runmat_value::Tensor::new(data_cm, vec![rows, cols]).ok()
                } else {
                    None
                };
                if tensor_const.is_some() {
                    node.tags.push(AccelGraphTag::FoldedLiteralArray);
                }
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
            Instr::CreateMatrixFromSequences {
                rows,
                row_lengths,
                elements,
            } => {
                self.handle_matrix_aggregate(pc, *rows, row_lengths, elements);
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
            Instr::Jump(_)
            | Instr::JumpIfFalse(_)
            | Instr::StochasticEvolution { .. }
            | Instr::Return
            | Instr::ReturnValue => {
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
        let ty = if self.creation_invalidated.contains(&idx) {
            Type::Unknown
        } else {
            self.var_types.get(idx).cloned().unwrap_or(Type::Unknown)
        };
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

    fn handle_store_var(&mut self, pc: usize, idx: usize) {
        if let Some(value_id) = self.pop_value() {
            let binding = VarBinding {
                kind: VarKind::Global,
                index: idx,
            };
            if let Some(stores) = &mut self.stores {
                stores.push(AccelStore {
                    pc,
                    value_id,
                    binding: binding.clone(),
                });
            }
            let key = VarKey::Global(idx);
            self.var_values.insert(key, value_id);
            if let Some(ty) = self
                .var_types
                .get(idx)
                .filter(|_| !self.creation_invalidated.contains(&idx))
            {
                self.apply_type(value_id, ty);
            }
            self.var_bindings.insert(value_id, binding);
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

    fn handle_store_local(&mut self, pc: usize, idx: usize) {
        if let Some(value_id) = self.pop_value() {
            let binding = VarBinding {
                kind: VarKind::Local,
                index: idx,
            };
            if let Some(stores) = &mut self.stores {
                stores.push(AccelStore {
                    pc,
                    value_id,
                    binding: binding.clone(),
                });
            }
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
            self.var_bindings.insert(value_id, binding);
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
        if op == PrimitiveOp::Mul && !self.value_is_scalar(lhs) && !self.value_is_scalar(rhs) {
            // Share the matrix declaration without erasing operator syntax:
            // a named mtimes override is not primitive numeric multiplication.
            let inputs = vec![lhs, rhs];
            let out_type = self.infer_matmul_type(&inputs);
            let name = runmat_types::OperatorOverloadMethod::MatrixMultiply
                .name()
                .display_name();
            self.push_builtin_node(
                pc,
                name,
                inputs,
                out_type,
                runmat_types::OperatorKind::MatrixMultiply,
            );
            return;
        }
        let inputs = vec![lhs, rhs];
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let out_type = match op {
            PrimitiveOp::Mul => {
                let lhs_type = self.values.get(lhs as usize).map(|v| &v.ty);
                let rhs_type = self.values.get(rhs as usize).map(|v| &v.ty);
                match (lhs_type, rhs_type) {
                    (Some(left), Some(right)) => {
                        runmat_builtins::shape_rules::matmul_output_type(left, right)
                    }
                    _ => Type::Unknown,
                }
            }
            _ => {
                let shape = self.infer_elementwise_shape(&inputs);
                if matches!(shape, ShapeInfo::Unknown) {
                    Type::Unknown
                } else {
                    shape.to_type()
                }
            }
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

    fn handle_right_divide(&mut self, pc: usize) {
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

        if self.value_is_scalar(rhs) {
            self.push_binary_primitive_node(pc, PrimitiveOp::ElemDiv, lhs, rhs);
            return;
        }

        let out_type = self.infer_right_divide_type(lhs, rhs);
        self.push_builtin_node(
            pc,
            "mrdivide",
            vec![lhs, rhs],
            out_type,
            runmat_types::OperatorKind::Mrdivide,
        );
    }

    fn handle_left_divide(&mut self, pc: usize) {
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

        if self.value_is_scalar(lhs) {
            self.push_binary_primitive_node(pc, PrimitiveOp::ElemLeftDiv, lhs, rhs);
            return;
        }

        let out_type = self.infer_left_divide_type(lhs, rhs);
        self.push_builtin_node(
            pc,
            "mldivide",
            vec![lhs, rhs],
            out_type,
            runmat_types::OperatorKind::Mldivide,
        );
    }

    fn push_binary_primitive_node(
        &mut self,
        pc: usize,
        op: PrimitiveOp,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let inputs = vec![lhs, rhs];
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let out_type = match op {
            PrimitiveOp::Mul => self.infer_matmul_type(&inputs),
            _ => {
                let shape = self.infer_elementwise_shape(&inputs);
                if matches!(shape, ShapeInfo::Unknown) {
                    Type::Unknown
                } else {
                    shape.to_type()
                }
            }
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

    fn push_builtin_node(
        &mut self,
        pc: usize,
        name: &str,
        inputs: Vec<ValueId>,
        out_type: Type,
        operator: runmat_types::OperatorKind,
    ) {
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
                name: name.into(),
                origin: runmat_accelerate::graph::FusionCallOrigin::Operator(operator),
            },
            category: info.category,
            inputs: inputs.clone(),
            outputs: Vec::new(),
            span,
            tags: info.tags,
        };
        let out_value = self.new_node_output(node_id, 0, out_type);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
    }

    fn infer_left_divide_type(&self, lhs: ValueId, rhs: ValueId) -> Type {
        let lhs_type = self.values.get(lhs as usize).map(|v| &v.ty);
        let rhs_type = self.values.get(rhs as usize).map(|v| &v.ty);
        match (lhs_type, rhs_type) {
            (Some(left), Some(right)) => {
                runmat_builtins::shape_rules::left_divide_output_type(left, right)
            }
            _ => Type::Unknown,
        }
    }

    fn infer_right_divide_type(&self, lhs: ValueId, rhs: ValueId) -> Type {
        let lhs_type = self.values.get(lhs as usize).map(|v| &v.ty);
        let rhs_type = self.values.get(rhs as usize).map(|v| &v.ty);
        match (lhs_type, rhs_type) {
            (Some(left), Some(right)) => {
                runmat_builtins::shape_rules::right_divide_output_type(left, right)
            }
            _ => Type::Unknown,
        }
    }

    fn value_is_scalar(&self, value_id: ValueId) -> bool {
        let Some(info) = self.values.get(value_id as usize) else {
            return false;
        };
        match &info.shape {
            ShapeInfo::Scalar => true,
            ShapeInfo::Tensor(dims) => element_count_if_known(dims.as_slice()) == Some(1),
            ShapeInfo::Unknown => false,
        }
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

    fn handle_call_builtin(&mut self, pc: usize, name: &str, argc: usize, outputs: usize) {
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
        let mut info = self
            .builtin_cache
            .get(&name.to_ascii_lowercase())
            .cloned()
            .unwrap_or(BuiltinInfo {
                category: AccelOpCategory::Other,
                tags: Vec::new(),
            });
        if outputs != 1 {
            info.category = AccelOpCategory::Other;
            info.tags.clear();
        } else if let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(name) {
            // Cached identity-level candidates do not admit a particular call's
            // controls. Structural constructor shape evidence is not an
            // executable fusion claim, so retain only that tag on rejection.
            if super::fusion_declaration::canonical_kind(entry, Some(argc)).is_none() {
                info.category = AccelOpCategory::Other;
                info.tags
                    .retain(|tag| matches!(tag, AccelGraphTag::ArrayConstruct));
            }
        }
        let category = self.call_category(name, &inputs, info.category.clone());
        let node_id = self.nodes.len() as NodeId;
        let span = InstrSpan { start: pc, end: pc };
        let mut node = AccelNode {
            id: node_id,
            label: AccelNodeLabel::Builtin {
                name: name.into(),
                origin: runmat_accelerate::graph::FusionCallOrigin::named(name),
            },
            category: category.clone(),
            inputs: inputs.clone(),
            outputs: Vec::new(),
            span,
            tags: info.tags.clone(),
        };
        // Array constructors derive geometry from admitted structural controls,
        // even when their canonical expression declaration is elementwise.
        // Broadcasting those controls would describe the inputs, not the result.
        let out_type = if info.tags.contains(&AccelGraphTag::ArrayConstruct) {
            self.infer_array_constructor_from_tags(&inputs)
                .unwrap_or(Type::Unknown)
        } else {
            match category {
                AccelOpCategory::Elementwise => self.infer_elementwise_shape(&inputs).to_type(),
                AccelOpCategory::Reduction => Type::Num,
                AccelOpCategory::MatMul => self.infer_matmul_type(&inputs),
                AccelOpCategory::Transpose | AccelOpCategory::Other => Type::Unknown,
            }
        };
        let out_value = self.new_node_output(node_id, 0, out_type);
        node.outputs.push(out_value);
        self.nodes.push(node);
        self.stack.push(out_value);
        if outputs == 1 {
            self.maybe_fold_builtin_constant(name, &inputs, out_value);
        }
    }

    fn call_category(
        &self,
        name: &str,
        inputs: &[ValueId],
        default_category: AccelOpCategory,
    ) -> AccelOpCategory {
        if matches!(default_category, AccelOpCategory::Reduction)
            && (name.eq_ignore_ascii_case("max") || name.eq_ignore_ascii_case("min"))
            && inputs.len() >= 2
            && self.value_is_known_nonempty_scalar_constant(inputs[1])
        {
            return AccelOpCategory::Elementwise;
        }
        default_category
    }

    fn value_is_empty_placeholder(&self, value_id: ValueId) -> bool {
        let Some(info) = self.values.get(value_id as usize) else {
            return false;
        };
        match info.constant.as_ref() {
            Some(Value::Tensor(t)) => t.is_empty(),
            Some(Value::LogicalArray(l)) => l.data.is_empty(),
            Some(Value::String(_)) => false,
            Some(Value::StringArray(sa)) => sa.data().is_empty(),
            Some(Value::CharArray(ca)) => ca.data().is_empty(),
            Some(Value::Cell(cell)) => cell.data().is_empty(),
            _ => false,
        }
    }

    fn value_is_known_nonempty_scalar_constant(&self, value_id: ValueId) -> bool {
        if self.value_is_empty_placeholder(value_id) {
            return false;
        }
        let Some(info) = self.values.get(value_id as usize) else {
            return false;
        };
        match info.constant.as_ref() {
            Some(Value::Num(value)) => value.is_finite(),
            Some(Value::Int(_)) | Some(Value::Bool(_)) => true,
            Some(Value::Tensor(t)) => t.len() == 1,
            Some(Value::LogicalArray(l)) => l.data.len() == 1,
            _ => false,
        }
    }

    fn infer_array_constructor_from_tags(&self, inputs: &[ValueId]) -> Option<Type> {
        // Generic version of infer_array_constructor_type without any name checks
        // 1) 'like' prototype takes precedence
        let mut i = 0usize;
        while i + 1 < inputs.len() {
            let s = &self.values[inputs[i] as usize].constant;
            if let Some(Value::String(text)) = s {
                if text
                    .as_str()
                    .is_some_and(|text| text.eq_ignore_ascii_case("like"))
                {
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
                Some(Value::Int(i)) => match i.try_to_usize() {
                    Some(v) => {
                        dims.push(Some(v));
                        continue;
                    }
                    None => break,
                },
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
            // Expect a constant tensor encoded as Value::Tensor; pull from constant if available
            if let Some(Value::Tensor(t)) = &info.constant {
                let rows = t.rows();
                let cols = t.cols();
                if (rows == 1 || rows == 0) && cols > 0 {
                    // 1xN row vector
                    let mut out: Vec<Option<usize>> = Vec::with_capacity(cols);
                    for j in 0..cols {
                        out.push(tensor_dimension_at(t, j));
                    }
                    return Some(Type::Tensor { shape: Some(out) });
                } else if (cols == 1 || cols == 0) && rows > 0 {
                    // Nx1 column vector
                    let mut out: Vec<Option<usize>> = Vec::with_capacity(rows);
                    for i in 0..rows {
                        out.push(tensor_dimension_at(t, i));
                    }
                    return Some(Type::Tensor { shape: Some(out) });
                }
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
        let rounded = match constant {
            Value::Num(n) => Some(*n as f32),
            // Integer conversion has class-specific exactness rules; leave it
            // to the ordinary runtime rather than a binary64 mirror.
            Value::Int(_) => None,
            Value::Bool(flag) => Some(if *flag { 1.0f32 } else { 0.0f32 }),
            _ => None,
        };
        let Some(rounded) = rounded else {
            return;
        };
        let Ok(tensor) = runmat_value::Tensor::from_f32(vec![rounded], vec![1, 1]) else {
            return;
        };
        let Some(ValueOrigin::NodeOutput { node, output }) =
            self.values.get(out_value as usize).map(|info| &info.origin)
        else {
            return;
        };
        let (node, output) = (*node, *output);
        if self
            .nodes
            .get(node as usize)
            .and_then(|entry| entry.outputs.get(output))
            != Some(&out_value)
        {
            return;
        }
        if let Some(out_info) = self.values.get_mut(out_value as usize) {
            out_info.ty = Type::Tensor {
                shape: Some(vec![Some(1), Some(1)]),
            };
            out_info.shape = ShapeInfo::Tensor(vec![Some(1), Some(1)]);
            out_info.constant = Some(Value::Tensor(tensor));
        }
        self.nodes[node as usize]
            .tags
            .push(AccelGraphTag::UnaryScalarFold);
    }
}

/// Read a tensor element used as structural metadata without consulting the
/// compatibility f64 mirror for typed integer tensors.
fn tensor_dimension_at(tensor: &runmat_value::Tensor, index: usize) -> Option<usize> {
    let scalar = tensor.numeric_value_at(index)?;
    if let Some(value) = scalar.into_int_value() {
        return value.try_to_usize();
    }
    let value = scalar.materialize_f64();
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    let rounded = value.round();
    ((rounded - value).abs() <= f64::EPSILON)
        .then_some(rounded)
        .and_then(|value| usize::try_from(value as u128).ok())
}

fn canonical_array_constructor_shape_tag(rule: BuiltinInferenceRule) -> bool {
    match rule {
        BuiltinInferenceRule::Identity(identity) => identity.has_structural_dimensions(),
        BuiltinInferenceRule::Array(ArrayInferenceRule::Creation(creation)) => match creation {
            // Zeros takes structural dimensions or a prototype. Full instead
            // materializes its input; treating that input as dimensions would
            // fabricate output geometry.
            ArrayCreationInferenceRule::Zeros => true,
            ArrayCreationInferenceRule::Full => false,
        },
        _ => false,
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
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn catalog_constructor_shape_is_analysis_only_across_nonfusion_calls() {
        let like = build_accel_graph(
            &[
                Instr::LoadConst(3.0),
                Instr::LoadConst(4.0),
                Instr::CallBuiltinMulti("rand".into(), 2, 1),
                Instr::StoreVar(0),
                Instr::LoadString("like".into()),
                Instr::LoadVar(0),
                Instr::CallBuiltinMulti("zeros".into(), 2, 1),
            ],
            &[Type::Unknown],
        );
        let like_node = like.nodes.last().expect("zeros graph node");
        assert_eq!(like_node.category, AccelOpCategory::Elementwise);
        assert!(like_node.tags.contains(&AccelGraphTag::ArrayConstruct));
        assert!(like_node.tags.contains(&AccelGraphTag::Elementwise));
        assert_eq!(
            like.value(like_node.outputs[0])
                .expect("zeros output")
                .shape,
            ShapeInfo::Tensor(vec![Some(3), Some(4)])
        );

        let size_vector = build_accel_graph(
            &[
                Instr::LoadConst(5.0),
                Instr::LoadConst(6.0),
                Instr::CreateMatrix(1, 2),
                Instr::StoreVar(0),
                Instr::LoadVar(0),
                Instr::CallBuiltinMulti("zeros".into(), 1, 1),
            ],
            &[Type::Unknown],
        );
        let vector_node = size_vector.nodes.last().expect("size-vector zeros node");
        assert_eq!(vector_node.category, AccelOpCategory::Elementwise);
        assert!(vector_node.tags.contains(&AccelGraphTag::ArrayConstruct));
        assert_eq!(
            size_vector
                .value(vector_node.outputs[0])
                .expect("size-vector output")
                .shape,
            ShapeInfo::Tensor(vec![Some(5), Some(6)])
        );
    }

    #[test]
    fn constructor_shape_does_not_fabricate_unknown_or_multioutput_geometry() {
        let unknown = build_accel_graph(
            &[
                Instr::LoadVar(0),
                Instr::CallBuiltinMulti("zeros".into(), 1, 1),
            ],
            &[Type::Unknown],
        );
        let unknown_node = unknown.nodes.last().expect("unknown zeros node");
        assert_eq!(unknown_node.category, AccelOpCategory::Elementwise);
        assert!(unknown_node.tags.contains(&AccelGraphTag::ArrayConstruct));
        assert_eq!(
            unknown.value(unknown_node.outputs[0]).unwrap().shape,
            ShapeInfo::Unknown
        );

        let multi = build_accel_graph(
            &[
                Instr::LoadConst(3.0),
                Instr::CallBuiltinMulti("zeros".into(), 1, 2),
            ],
            &[],
        );
        let multi_node = multi.nodes.last().expect("multi-output zeros node");
        assert_eq!(multi_node.category, AccelOpCategory::Other);
        assert!(multi_node.tags.is_empty());
        assert_eq!(
            multi.value(multi_node.outputs[0]).unwrap().shape,
            ShapeInfo::Unknown
        );

        let full = GraphBuilder::new(&[], &[]);
        let full_entry = runmat_builtins::builtin_catalog_entry_by_name("full").unwrap();
        assert!(!canonical_array_constructor_shape_tag(
            full_entry.contract.inference_rule
        ));
        assert!(!full
            .builtin_cache
            .get("full")
            .unwrap()
            .tags
            .contains(&AccelGraphTag::ArrayConstruct));
    }

    #[test]
    fn structural_tensor_dimensions_read_all_integer_classes_exactly() {
        macro_rules! assert_dimensions {
            ($storage:expr) => {{
                let tensor = Tensor::new_integer($storage, vec![1, 2]).expect("dimensions");
                assert_eq!(tensor_dimension_at(&tensor, 0), Some(2));
                assert_eq!(tensor_dimension_at(&tensor, 1), Some(3));
            }};
        }

        assert_dimensions!(IntegerStorage::I8(vec![2, 3]));
        assert_dimensions!(IntegerStorage::I16(vec![2, 3]));
        assert_dimensions!(IntegerStorage::I32(vec![2, 3]));
        assert_dimensions!(IntegerStorage::I64(vec![2, 3]));
        assert_dimensions!(IntegerStorage::U8(vec![2, 3]));
        assert_dimensions!(IntegerStorage::U16(vec![2, 3]));
        assert_dimensions!(IntegerStorage::U32(vec![2, 3]));
        assert_dimensions!(IntegerStorage::U64(vec![2, 3]));
    }

    #[test]
    fn graph_folding_declines_wide_integer_scalars() {
        for constant in [
            Value::Int(IntValue::I64(i64::MAX)),
            Value::Int(IntValue::U64(u64::MAX)),
        ] {
            let mut builder = GraphBuilder::new(&[], &[]);
            let input = builder.new_value(ValueOrigin::Constant, Type::Num, Some(constant));
            let output = builder.new_value(ValueOrigin::Constant, Type::Num, None);

            builder.maybe_fold_builtin_constant("single", &[input], output);
            assert!(builder.values[output as usize].constant.is_none());
        }
    }

    #[test]
    fn graph_builder_tags_only_actual_unary_scalar_folds() {
        for source in [0.0, 0.1] {
            let graph = build_accel_graph(
                &[
                    Instr::LoadConst(source),
                    Instr::CallBuiltinMulti("single".into(), 1, 1),
                ],
                &[],
            );
            let node = &graph.nodes[0];
            assert!(node.tags.contains(&AccelGraphTag::UnaryScalarFold));
            let output = graph.value(node.outputs[0]).expect("folded output");
            assert_eq!(output.shape, ShapeInfo::Tensor(vec![Some(1), Some(1)]));
            assert_eq!(
                output.ty,
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(1)])
                }
            );
            let Some(Value::Tensor(folded)) = &output.constant else {
                panic!("folded single must retain native F32 tensor storage");
            };
            assert_eq!(folded.shape(), [1, 1]);
            assert_eq!(
                folded.as_f32_slice().unwrap()[0].to_bits(),
                (source as f32).to_bits()
            );
            let ordinary = futures::executor::block_on(
                runmat_runtime::dispatcher::call_builtin_async("single", &[Value::Num(source)]),
            )
            .expect("ordinary single binding");
            assert_eq!(output.constant.as_ref(), Some(ordinary.value()));
        }

        for instructions in [
            vec![
                Instr::LoadInt(IntValue::U64(u64::MAX)),
                Instr::CallBuiltinMulti("single".into(), 1, 1),
            ],
            vec![
                Instr::LoadVar(0),
                Instr::CallBuiltinMulti("single".into(), 1, 1),
            ],
            vec![
                Instr::LoadConst(0.0),
                Instr::CallBuiltinMulti("single".into(), 1, 2),
            ],
        ] {
            let graph = build_accel_graph(&instructions, &[Type::Unknown]);
            let node = &graph.nodes[0];
            assert!(!node.tags.contains(&AccelGraphTag::UnaryScalarFold));
            assert!(graph.value(node.outputs[0]).unwrap().constant.is_none());
        }
    }

    #[test]
    fn store_transcript_retains_repeated_and_aliased_publications_in_pc_order() {
        let (graph, stores) = build_accel_graph_with_stores(
            &[
                Instr::LoadConst(7.0),
                Instr::StoreVar(0),
                Instr::LoadVar(0),
                Instr::StoreLocal(1),
                Instr::LoadConst(8.0),
                Instr::StoreVar(0),
            ],
            &[],
        );
        assert_eq!(stores.len(), 3);
        assert_eq!(
            stores.iter().map(|store| store.pc).collect::<Vec<_>>(),
            [1, 3, 5]
        );
        assert_eq!(stores[0].value_id, stores[1].value_id);
        assert_ne!(stores[1].value_id, stores[2].value_id);
        assert_eq!(stores[0].binding.kind, VarKind::Global);
        assert_eq!(stores[1].binding.kind, VarKind::Local);
        assert_eq!(stores[2].binding.kind, VarKind::Global);
        assert_eq!(stores[0].binding.index, 0);
        assert_eq!(stores[1].binding.index, 1);
        assert_eq!(stores[2].binding.index, 0);
        // The historical map remains a last-binding graph hint, not a store log.
        assert_eq!(
            graph.var_binding(stores[0].value_id),
            Some(&stores[1].binding)
        );
    }

    #[test]
    fn assignment_creation_loads_do_not_reuse_numeric_graph_cache() {
        for load in [
            Instr::LoadVarForIndexAssignment(0),
            Instr::LoadVarForCellAssignment(0),
            Instr::LoadVarForStructAssignment(0),
        ] {
            let (graph, stores) = build_accel_graph_with_stores(
                &[
                    Instr::LoadConst(7.0),
                    Instr::StoreVar(0),
                    load,
                    Instr::StoreVar(1),
                    Instr::LoadVar(0),
                    Instr::StoreVar(2),
                ],
                &[Type::Num, Type::Unknown, Type::Unknown],
            );
            assert_eq!(
                stores.len(),
                2,
                "a creation load cannot reuse the cached numeric value"
            );
            assert_eq!(stores[0].pc, 1);
            assert_eq!(stores[1].pc, 5);
            assert!(matches!(
                &graph.value(stores[1].value_id).unwrap().ty,
                Type::Unknown
            ));
        }
    }

    #[test]
    fn graph_shape_inference_rejects_negative_integer_dimension() {
        let mut builder = GraphBuilder::new(&[], &[]);
        let dimension = builder.new_value(
            ValueOrigin::Constant,
            Type::Num,
            Some(Value::Int(IntValue::I64(-1))),
        );

        assert_eq!(
            builder.infer_array_constructor_from_tags(&[dimension]),
            None
        );
    }

    #[test]
    fn accel_graph_mul_uses_matmul_shape() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::Mul,
            Instr::StoreVar(2),
        ];
        let var_types = vec![
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            },
            Type::Tensor {
                shape: Some(vec![Some(3), Some(4)]),
            },
            Type::Unknown,
        ];
        let graph = build_accel_graph(&instructions, &var_types);
        let mut out_type = None;
        for node in &graph.nodes {
            if node.category == AccelOpCategory::MatMul {
                let out_id = node.outputs.first().copied().expect("output");
                let value = graph.value(out_id).expect("value");
                out_type = Some(value.ty.clone());
            }
        }
        assert_eq!(
            out_type,
            Some(Type::Tensor {
                shape: Some(vec![Some(2), Some(4)])
            })
        );
    }

    #[test]
    fn operator_product_refines_only_proven_scalar_operands_to_elementwise() {
        let matrix = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        let scalar_tensor = Type::Tensor {
            shape: Some(vec![Some(1), Some(1)]),
        };
        for (left, right, category) in [
            (Type::Num, matrix.clone(), AccelOpCategory::Elementwise),
            (matrix.clone(), Type::Num, AccelOpCategory::Elementwise),
            (
                scalar_tensor.clone(),
                matrix.clone(),
                AccelOpCategory::Elementwise,
            ),
            (matrix, scalar_tensor, AccelOpCategory::Elementwise),
            (Type::Unknown, Type::Unknown, AccelOpCategory::MatMul),
        ] {
            let graph = build_accel_graph(
                &[Instr::LoadVar(0), Instr::LoadVar(1), Instr::Mul],
                &[left, right],
            );
            let node = graph.nodes.last().unwrap();
            assert_eq!(node.category, category);
            if category == AccelOpCategory::Elementwise {
                assert!(matches!(
                    node.label,
                    AccelNodeLabel::Primitive(PrimitiveOp::Mul)
                ));
            } else {
                assert!(node.tags.contains(&AccelGraphTag::MatMul));
            }
        }
    }

    #[test]
    fn accel_graph_right_divide_uses_mrdivide_shape() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::RightDiv,
            Instr::StoreVar(2),
        ];
        let var_types = vec![
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)]),
            },
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)]),
            },
            Type::Unknown,
        ];
        let graph = build_accel_graph(&instructions, &var_types);
        let mut out_type = None;
        for node in &graph.nodes {
            if matches!(
                &node.label,
                AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("mrdivide")
            ) {
                let out_id = node.outputs.first().copied().expect("output");
                let value = graph.value(out_id).expect("value");
                out_type = Some(value.ty.clone());
            }
        }
        assert_eq!(
            out_type,
            Some(Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            })
        );
    }

    #[test]
    fn accel_graph_trapz_is_not_classified_as_generic_reduction() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::CallBuiltinMulti("trapz".into(), 1, 1),
        ];
        let var_types = vec![Type::Tensor {
            shape: Some(vec![Some(1), Some(3)]),
        }];
        let graph = build_accel_graph(&instructions, &var_types);

        let node = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.label,
                    AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("trapz")
                )
            })
            .expect("trapz node");

        assert_eq!(node.category, AccelOpCategory::Other);
        assert!(!node
            .tags
            .iter()
            .any(|tag| matches!(tag, AccelGraphTag::Reduction)));
    }

    #[test]
    fn accel_graph_cumtrapz_is_not_classified_as_generic_reduction() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::CallBuiltinMulti("cumtrapz".into(), 1, 1),
        ];
        let var_types = vec![Type::Tensor {
            shape: Some(vec![Some(1), Some(3)]),
        }];
        let graph = build_accel_graph(&instructions, &var_types);

        let node = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.label,
                    AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("cumtrapz")
                )
            })
            .expect("cumtrapz node");

        assert_eq!(node.category, AccelOpCategory::Other);
        assert!(!node
            .tags
            .iter()
            .any(|tag| matches!(tag, AccelGraphTag::Reduction)));
    }

    #[test]
    fn canonical_graph_candidates_preserve_unary_controls_and_output_arity() {
        for name in ["single", "double", "sin", "cos", "tan", "atan"] {
            for (arguments, outputs, expected) in [
                (1, 1, AccelOpCategory::Elementwise),
                (3, 1, AccelOpCategory::Other),
                (1, 0, AccelOpCategory::Other),
                (1, 2, AccelOpCategory::Other),
            ] {
                let mut instructions = (0..arguments).map(Instr::LoadVar).collect::<Vec<_>>();
                instructions.push(Instr::CallBuiltinMulti(name.into(), arguments, outputs));
                let graph = build_accel_graph(&instructions, &vec![Type::Unknown; arguments]);
                let node = graph.nodes.iter().find(|node| {
                    matches!(&node.label, AccelNodeLabel::Builtin { name: actual, .. } if actual == name)
                }).expect("canonical call node");
                assert_eq!(node.category, expected, "{name}/{arguments}/{outputs}");
                if expected == AccelOpCategory::Other {
                    assert!(
                        node.tags.is_empty(),
                        "rejected call must not retain fusion tags"
                    );
                }
            }
        }
    }

    #[test]
    fn canonical_matrix_product_graph_preserves_exact_arity() {
        for (arguments, outputs, expected) in [
            (2, 1, AccelOpCategory::MatMul),
            (1, 1, AccelOpCategory::Other),
            (3, 1, AccelOpCategory::Other),
            (2, 0, AccelOpCategory::Other),
            (2, 2, AccelOpCategory::Other),
        ] {
            let mut instructions = (0..arguments).map(Instr::LoadVar).collect::<Vec<_>>();
            instructions.push(Instr::CallBuiltinMulti("mtimes".into(), arguments, outputs));
            let graph = build_accel_graph(&instructions, &vec![Type::Unknown; arguments]);
            let node = graph.nodes.iter().find(|node| {
                matches!(&node.label, AccelNodeLabel::Builtin { name, .. } if name == "mtimes")
            }).unwrap();
            assert_eq!(node.category, expected, "{arguments}/{outputs}");
            assert_eq!(
                node.tags.contains(&AccelGraphTag::MatMul),
                expected == AccelOpCategory::MatMul
            );
        }
    }

    #[test]
    fn rebuilt_graph_distinguishes_primitive_matrix_product_from_named_call() {
        use runmat_accelerate::graph::FusionCallOrigin;
        let origin = |instruction| {
            let graph = build_accel_graph(
                &[Instr::LoadVar(0), Instr::LoadVar(1), instruction],
                &[Type::Unknown, Type::Unknown],
            );
            graph
                .nodes
                .into_iter()
                .find_map(|node| match node.label {
                    AccelNodeLabel::Builtin { name, origin } if name == "mtimes" => Some(origin),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(
            origin(Instr::Mul),
            FusionCallOrigin::Operator(runmat_types::OperatorKind::MatrixMultiply)
        );
        assert_eq!(
            origin(Instr::CallBuiltinMulti("mtimes".into(), 2, 1)),
            FusionCallOrigin::named("mtimes")
        );
    }

    #[test]
    fn accel_graph_max_with_scalar_second_arg_is_elementwise() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::CallBuiltinMulti("single".into(), 1, 1),
            Instr::CallBuiltinMulti("max".into(), 2, 1),
        ];
        let var_types = vec![Type::Tensor {
            shape: Some(vec![Some(2), Some(4), Some(5)]),
        }];
        let graph = build_accel_graph(&instructions, &var_types);

        let node = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.label,
                    AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("max")
                )
            })
            .expect("max node");

        assert_eq!(node.category, AccelOpCategory::Elementwise);
        let out_id = node.outputs.first().copied().expect("max output");
        let out = graph.value(out_id).expect("max output value");
        assert_eq!(
            out.ty,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4), Some(5)])
            }
        );
    }

    #[test]
    fn accel_graph_max_with_variable_scalar_constant_is_elementwise() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadConst(0.0),
            Instr::CallBuiltinMulti("single".into(), 1, 1),
            Instr::StoreVar(1),
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::CallBuiltinMulti("max".into(), 2, 1),
        ];
        let var_types = vec![
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4), Some(5)]),
            },
            Type::Unknown,
        ];
        let graph = build_accel_graph(&instructions, &var_types);

        let node = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.label,
                    AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("max")
                )
            })
            .expect("max node");

        assert_eq!(node.category, AccelOpCategory::Elementwise);
        let out_id = node.outputs.first().copied().expect("max output");
        let out = graph.value(out_id).expect("max output value");
        assert_eq!(
            out.ty,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4), Some(5)])
            }
        );
    }

    #[test]
    fn accel_graph_max_with_empty_placeholder_is_not_elementwise() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::CreateMatrix(0, 0),
            Instr::CallBuiltinMulti("max".into(), 2, 1),
        ];
        let var_types = vec![Type::Tensor {
            shape: Some(vec![Some(2), Some(4), Some(5)]),
        }];
        let graph = build_accel_graph(&instructions, &var_types);

        let node = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.label,
                    AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("max")
                )
            })
            .expect("max node");

        assert_eq!(node.category, AccelOpCategory::Reduction);
        let out_id = node.outputs.first().copied().expect("max output");
        let out = graph.value(out_id).expect("max output value");
        assert_eq!(out.ty, Type::Num);
    }

    #[test]
    fn accel_graph_max_with_runtime_unknown_second_arg_is_conservative() {
        let instructions = vec![
            Instr::LoadVar(0),
            Instr::LoadVar(1),
            Instr::CallBuiltinMulti("max".into(), 2, 1),
        ];
        let var_types = vec![
            Type::Tensor {
                shape: Some(vec![Some(2), Some(4), Some(5)]),
            },
            Type::Unknown,
        ];
        let graph = build_accel_graph(&instructions, &var_types);

        let node = graph
            .nodes
            .iter()
            .find(|node| {
                matches!(
                    &node.label,
                    AccelNodeLabel::Builtin { name, .. } if name.eq_ignore_ascii_case("max")
                )
            })
            .expect("max node");

        assert_eq!(node.category, AccelOpCategory::Reduction);
        let out_id = node.outputs.first().copied().expect("max output");
        let out = graph.value(out_id).expect("max output value");
        assert_eq!(out.ty, Type::Num);
    }
}
