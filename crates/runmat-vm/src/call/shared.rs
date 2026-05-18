use crate::bytecode::ArgSpec;
use crate::bytecode::EndExpr;
use runmat_builtins::Value;
use runmat_hir::{CallableFallbackPolicy, CallableIdentity, QualifiedName, SymbolName};
use runmat_runtime::RuntimeError;
use std::future::Future;

const OBJECT_PROTOCOL_SUBSREF: &str = runmat_runtime::OBJECT_SUBSREF_METHOD;
const OBJECT_PROTOCOL_SUBSASGN: &str = runmat_runtime::OBJECT_SUBSASGN_METHOD;
const OBJECT_PROTOCOL_KIND_PAREN: &str = runmat_runtime::OBJECT_INDEX_PAREN;
const OBJECT_PROTOCOL_KIND_BRACE: &str = runmat_runtime::OBJECT_INDEX_BRACE;
const OBJECT_PROTOCOL_KIND_MEMBER: &str = runmat_runtime::OBJECT_INDEX_MEMBER;
const OBJECT_SELECTOR_COLON: &str = ":";
const OBJECT_SELECTOR_END: &str = "end";
const OBJECT_END_RANGE_TAG: &str = "end_expr";

pub fn expand_cell_indices(
    cell: &runmat_builtins::CellArray,
    indices: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    crate::ops::cells::expand_cell_indices(cell, indices)
}

pub fn expand_all_cell(cell: &runmat_builtins::CellArray) -> Result<Vec<Value>, RuntimeError> {
    crate::ops::cells::expand_all_cell_values(cell)
}

pub(crate) fn external_qualified_identity(base: &str, member: &str) -> CallableIdentity {
    let mut segments: Vec<SymbolName> = base
        .split('.')
        .filter(|segment| !segment.is_empty())
        .map(|segment| SymbolName(segment.to_string()))
        .collect();
    segments.push(SymbolName(member.to_string()));
    CallableIdentity::ExternalName(QualifiedName(segments))
}

pub(crate) fn external_qualified_display_name(base: &str, member: &str) -> String {
    external_qualified_identity(base, member)
        .display_name()
        .unwrap_or_else(|| format!("{base}.{member}"))
}

pub(crate) fn object_property_getter_name(field: &str) -> String {
    runmat_runtime::object_property_getter_name(field)
}

pub(crate) fn object_property_setter_name(field: &str) -> String {
    runmat_runtime::object_property_setter_name(field)
}

pub(crate) async fn expand_brace_values(
    base: Value,
    raw_indices: &[Value],
    pad_to_outputs: Option<usize>,
) -> Result<Vec<Value>, RuntimeError> {
    async fn expand_object_brace_values(
        base: Value,
        raw_indices: &[Value],
        pad_to_outputs: Option<usize>,
    ) -> Result<Vec<Value>, RuntimeError> {
        let value = call_object_index_descriptor_method_with_outputs(
            ObjectIndexDescriptor::subsref_brace(
                base,
                ObjectIndexSelector::IndexValues {
                    values: raw_indices.to_vec(),
                },
            ),
            pad_to_outputs.unwrap_or(1),
        )
        .await?;
        Ok(match value {
            Value::OutputList(values) => values,
            other => vec![other],
        })
    }

    let mut values = match base {
        Value::Cell(ca) => {
            if raw_indices.is_empty() {
                if let Some(out_count) = pad_to_outputs {
                    crate::ops::cells::expand_cell_values(&ca, &[], out_count)?
                } else {
                    expand_all_cell(&ca)?
                }
            } else {
                expand_cell_indices(&ca, raw_indices)?
            }
        }
        Value::Object(obj) => {
            expand_object_brace_values(Value::Object(obj), raw_indices, pad_to_outputs).await?
        }
        Value::HandleObject(handle) => {
            expand_object_brace_values(Value::HandleObject(handle), raw_indices, pad_to_outputs)
                .await?
        }
        _ => {
            return Err(crate::interpreter::errors::mex(
                "CellExpansionOnNonCell",
                "Cell expansion on non-cell",
            ))
        }
    };
    if let Some(out_count) = pad_to_outputs {
        if values.len() > out_count {
            values.truncate(out_count);
        } else {
            values.resize(out_count, Value::Num(0.0));
        }
    }
    Ok(values)
}

#[derive(Clone, Copy)]
pub(crate) enum ObjectIndexOp {
    Subsref,
    Subsasgn,
}

impl ObjectIndexOp {
    pub(crate) fn protocol_name(self) -> &'static str {
        match self {
            Self::Subsref => OBJECT_PROTOCOL_SUBSREF,
            Self::Subsasgn => OBJECT_PROTOCOL_SUBSASGN,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ObjectIndexKind {
    Paren,
    Brace,
    Member,
}

impl ObjectIndexKind {
    pub(crate) fn protocol_name(self) -> &'static str {
        match self {
            Self::Paren => OBJECT_PROTOCOL_KIND_PAREN,
            Self::Brace => OBJECT_PROTOCOL_KIND_BRACE,
            Self::Member => OBJECT_PROTOCOL_KIND_MEMBER,
        }
    }
}

pub(crate) enum ObjectIndexSelector {
    ScalarIndices { indices: Vec<usize> },
    IndexValues { values: Vec<Value> },
    Member(String),
}

pub(crate) struct ObjectIndexDescriptor {
    base: Value,
    op: ObjectIndexOp,
    kind: ObjectIndexKind,
    selector: ObjectIndexSelector,
    rhs: Option<Value>,
}

impl ObjectIndexDescriptor {
    pub(crate) fn subsref_paren(base: Value, selector: ObjectIndexSelector) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Paren,
            selector,
            rhs: None,
        }
    }

    pub(crate) fn subsref_brace(base: Value, selector: ObjectIndexSelector) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Brace,
            selector,
            rhs: None,
        }
    }

    pub(crate) fn subsasgn_paren(base: Value, selector: ObjectIndexSelector, rhs: Value) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Paren,
            selector,
            rhs: Some(rhs),
        }
    }

    pub(crate) fn subsasgn_brace(base: Value, selector: ObjectIndexSelector, rhs: Value) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Brace,
            selector,
            rhs: Some(rhs),
        }
    }

    pub(crate) fn subsref_paren_from_slice(
        base: Value,
        dims: usize,
        colon_mask: u32,
        end_mask: u32,
        numeric: &[Value],
    ) -> Result<Self, RuntimeError> {
        let values = build_object_paren_selector_values(dims, colon_mask, end_mask, numeric)?;
        Ok(Self::subsref_paren(
            base,
            ObjectIndexSelector::IndexValues { values },
        ))
    }

    pub(crate) fn subsasgn_paren_from_slice(
        base: Value,
        dims: usize,
        colon_mask: u32,
        end_mask: u32,
        numeric: &[Value],
        rhs: Value,
    ) -> Result<Self, RuntimeError> {
        let values = build_object_paren_selector_values(dims, colon_mask, end_mask, numeric)?;
        Ok(Self::subsasgn_paren(
            base,
            ObjectIndexSelector::IndexValues { values },
            rhs,
        ))
    }

    pub(crate) fn subsasgn_paren_from_expr_slice(
        base: Value,
        dims: usize,
        colon_mask: u32,
        end_mask: u32,
        range_dims: &[usize],
        range_params: &[(f64, f64)],
        range_start_exprs: &[Option<EndExpr>],
        range_step_exprs: &[Option<EndExpr>],
        range_end_exprs: &[EndExpr],
        numeric: &[Value],
        rhs: Value,
    ) -> Result<Self, RuntimeError> {
        let values = build_object_paren_expr_selector_values(
            dims,
            colon_mask,
            end_mask,
            range_dims,
            range_params,
            range_start_exprs,
            range_step_exprs,
            range_end_exprs,
            numeric,
        )?;
        Ok(Self::subsasgn_paren(
            base,
            ObjectIndexSelector::IndexValues { values },
            rhs,
        ))
    }

    pub(crate) fn member(
        base: Value,
        op: ObjectIndexOp,
        field: String,
        rhs: Option<Value>,
    ) -> Self {
        Self {
            base,
            op,
            kind: ObjectIndexKind::Member,
            selector: ObjectIndexSelector::Member(field),
            rhs,
        }
    }

    fn into_method_invocation(self) -> Result<(Value, String, Vec<Value>), RuntimeError> {
        let selector = match self.selector {
            ObjectIndexSelector::ScalarIndices { indices } => {
                let values = indices
                    .into_iter()
                    .map(|index| Value::Num(index as f64))
                    .collect();
                build_protocol_index_cell(values)?
            }
            ObjectIndexSelector::IndexValues { values } => build_protocol_index_cell(values)?,
            ObjectIndexSelector::Member(field) => Value::String(field),
        };
        let mut args = vec![
            Value::String(self.kind.protocol_name().to_string()),
            selector,
        ];
        if let Some(rhs) = self.rhs {
            args.push(rhs);
        }
        Ok((self.base, self.op.protocol_name().to_string(), args))
    }
}

fn build_protocol_index_cell(values: Vec<Value>) -> Result<Value, RuntimeError> {
    let cols = values.len();
    let cell = runmat_builtins::CellArray::new(values, 1, cols)
        .map_err(|e| format!("object index descriptor build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub(crate) async fn call_getfield_with_indices(
    base: Value,
    field: String,
    indices: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    let mut getfield_args = Vec::with_capacity(3);
    getfield_args.push(base);
    getfield_args.push(Value::String(field));
    if !indices.is_empty() {
        let idx_count = indices.len();
        let idx_cell = runmat_builtins::CellArray::new(indices, 1, idx_count)
            .map_err(|e| format!("getfield idx build: {e}"))?;
        getfield_args.push(Value::Cell(idx_cell));
    }
    runmat_runtime::call_builtin_async_with_outputs("getfield", &getfield_args, requested_outputs)
        .await
}

pub(crate) async fn call_object_operator_method(
    base: Value,
    method: &str,
    arg: Value,
) -> Result<Value, RuntimeError> {
    crate::call::closures::call_method_or_member_index_with_outputs(
        base,
        CallableIdentity::DynamicName(SymbolName(method.to_string())),
        Some(method.to_string()),
        vec![arg],
        1,
        CallableFallbackPolicy::ObjectDispatch,
    )
    .await
}

pub(crate) async fn call_object_named_method_with_outputs(
    base: Value,
    method: String,
    args: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    crate::call::closures::call_method_or_member_index_with_outputs(
        base,
        CallableIdentity::DynamicName(SymbolName(method.clone())),
        Some(method),
        args,
        requested_outputs,
        CallableFallbackPolicy::ObjectDispatch,
    )
    .await
}

pub(crate) async fn call_object_property_getter_with_outputs(
    base: Value,
    field: &str,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    call_object_named_method_with_outputs(
        base,
        object_property_getter_name(field),
        vec![],
        requested_outputs,
    )
    .await
}

pub(crate) async fn call_object_property_setter_with_outputs(
    base: Value,
    field: &str,
    value: Value,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    call_object_named_method_with_outputs(
        base,
        object_property_setter_name(field),
        vec![value],
        requested_outputs,
    )
    .await
}

async fn call_object_member_method(
    base: Value,
    op: ObjectIndexOp,
    field: String,
    rhs: Option<Value>,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method(ObjectIndexDescriptor::member(base, op, field, rhs)).await
}

pub(crate) async fn call_object_member_subsref(
    base: Value,
    field: String,
) -> Result<Value, RuntimeError> {
    call_object_member_method(base, ObjectIndexOp::Subsref, field, None).await
}

pub(crate) async fn call_object_member_subsasgn(
    base: Value,
    field: String,
    rhs: Value,
) -> Result<Value, RuntimeError> {
    call_object_member_method(base, ObjectIndexOp::Subsasgn, field, Some(rhs)).await
}

pub(crate) fn class_defines_member_subsref(class: &runmat_builtins::ClassDef) -> bool {
    class
        .methods
        .contains_key(ObjectIndexOp::Subsref.protocol_name())
}

pub(crate) fn class_defines_member_subsasgn(class: &runmat_builtins::ClassDef) -> bool {
    class
        .methods
        .contains_key(ObjectIndexOp::Subsasgn.protocol_name())
}

pub(crate) async fn call_object_index_descriptor_method(
    descriptor: ObjectIndexDescriptor,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method_with_outputs(descriptor, 1).await
}

pub(crate) async fn call_object_index_descriptor_method_with_outputs(
    descriptor: ObjectIndexDescriptor,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    let (base, method, args) = descriptor.into_method_invocation()?;
    crate::call::closures::call_method_or_member_index_with_outputs(
        base,
        CallableIdentity::DynamicName(SymbolName(method.clone())),
        Some(method),
        args,
        requested_outputs,
        CallableFallbackPolicy::ObjectDispatch,
    )
    .await
}

fn encode_end_expr_value(expr: &EndExpr) -> Result<Value, RuntimeError> {
    fn mk_cell(items: Vec<Value>) -> Result<Value, RuntimeError> {
        let cols = items.len();
        let cell = runmat_builtins::CellArray::new(items, 1, cols)
            .map_err(|e| format!("end expression encoding: {e}"))?;
        Ok(Value::Cell(cell))
    }

    match expr {
        EndExpr::End => Ok(Value::String("end".to_string())),
        EndExpr::Const(v) => Ok(Value::Num(*v)),
        EndExpr::Var(i) => Ok(Value::String(format!("var:{i}"))),
        EndExpr::ResolvedCall {
            identity,
            display_name,
            args,
            ..
        } => {
            let name = display_name
                .clone()
                .or_else(|| identity.display_name())
                .unwrap_or_else(|| "<unnamed>".to_string());
            let mut items = vec![Value::String("call".to_string()), Value::String(name)];
            for a in args {
                items.push(encode_end_expr_value(a)?);
            }
            mk_cell(items)
        }
        EndExpr::Add(a, b) => mk_cell(vec![
            Value::String("+".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Sub(a, b) => mk_cell(vec![
            Value::String("-".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Mul(a, b) => mk_cell(vec![
            Value::String("*".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Div(a, b) => mk_cell(vec![
            Value::String("/".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::LeftDiv(a, b) => mk_cell(vec![
            Value::String("\\".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Pow(a, b) => mk_cell(vec![
            Value::String("^".to_string()),
            encode_end_expr_value(a)?,
            encode_end_expr_value(b)?,
        ]),
        EndExpr::Neg(a) => mk_cell(vec![
            Value::String("neg".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Pos(a) => mk_cell(vec![
            Value::String("pos".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Floor(a) => mk_cell(vec![
            Value::String("floor".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Ceil(a) => mk_cell(vec![
            Value::String("ceil".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Round(a) => mk_cell(vec![
            Value::String("round".to_string()),
            encode_end_expr_value(a)?,
        ]),
        EndExpr::Fix(a) => mk_cell(vec![
            Value::String("fix".to_string()),
            encode_end_expr_value(a)?,
        ]),
    }
}

fn build_end_range_descriptor(
    start: Value,
    step: Value,
    end_expr: &EndExpr,
) -> Result<Value, RuntimeError> {
    let encoded_end = encode_end_expr_value(end_expr)?;
    let cell = runmat_builtins::CellArray::new(
        vec![
            start,
            step,
            Value::String(OBJECT_END_RANGE_TAG.to_string()),
            encoded_end,
        ],
        1,
        4,
    )
    .map_err(|e| format!("obj range: {e}"))?;
    Ok(Value::Cell(cell))
}

fn normalize_object_numeric_selector(selector: &Value) -> Result<Value, RuntimeError> {
    match selector {
        Value::Num(n) => Ok(Value::Num(*n)),
        Value::Int(i) => Ok(Value::Num(i.to_f64())),
        Value::Tensor(t) => Ok(Value::Tensor(t.clone())),
        other => Err(format!("Unsupported index type for object selector: {other:?}").into()),
    }
}

fn validate_object_range_selector_plan(
    dims: usize,
    range_dims: &[usize],
    range_params: &[(f64, f64)],
    range_start_exprs: &[Option<EndExpr>],
    range_step_exprs: &[Option<EndExpr>],
    range_end_exprs: &[EndExpr],
) -> Result<Vec<Option<usize>>, RuntimeError> {
    let count = range_dims.len();
    if range_params.len() != count
        || range_start_exprs.len() != count
        || range_step_exprs.len() != count
        || range_end_exprs.len() != count
    {
        return Err(crate::interpreter::errors::mex(
            "InvalidRangeSelectorPlan",
            "inconsistent object range selector metadata",
        ));
    }

    let mut range_pos_by_dim = vec![None; dims];
    for (pos, &dim) in range_dims.iter().enumerate() {
        if dim >= dims {
            return Err(crate::interpreter::errors::mex(
                "InvalidRangeSelectorDim",
                "object range selector dimension is out of bounds",
            ));
        }
        if range_pos_by_dim[dim].replace(pos).is_some() {
            return Err(crate::interpreter::errors::mex(
                "DuplicateRangeSelectorDim",
                "object range selector dimension appears more than once",
            ));
        }
    }
    Ok(range_pos_by_dim)
}

pub(crate) fn build_object_paren_selector_values(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    let mut values = Vec::with_capacity(dims);
    let mut numeric_iter = 0usize;
    for d in 0..dims {
        let is_colon = (colon_mask & (1u32 << d)) != 0;
        let is_end = (end_mask & (1u32 << d)) != 0;
        if is_colon {
            values.push(Value::String(OBJECT_SELECTOR_COLON.to_string()));
            continue;
        }
        if is_end {
            values.push(Value::String(OBJECT_SELECTOR_END.to_string()));
            continue;
        }
        let selector = numeric
            .get(numeric_iter)
            .ok_or(crate::interpreter::errors::mex(
                "MissingNumericIndex",
                "missing numeric index",
            ))?;
        values.push(selector.clone());
        numeric_iter += 1;
    }
    if numeric_iter != numeric.len() {
        return Err(crate::interpreter::errors::mex(
            "UnexpectedNumericIndex",
            "unexpected extra numeric index values",
        ));
    }
    Ok(values)
}

pub(crate) fn build_object_paren_expr_selector_values(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    range_dims: &[usize],
    range_params: &[(f64, f64)],
    range_start_exprs: &[Option<EndExpr>],
    range_step_exprs: &[Option<EndExpr>],
    range_end_exprs: &[EndExpr],
    numeric: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    let range_pos_by_dim = validate_object_range_selector_plan(
        dims,
        range_dims,
        range_params,
        range_start_exprs,
        range_step_exprs,
        range_end_exprs,
    )?;
    let mut values = Vec::with_capacity(dims);
    let mut num_iter = 0usize;
    for d in 0..dims {
        let is_colon = (colon_mask & (1u32 << d)) != 0;
        let is_end = (end_mask & (1u32 << d)) != 0;
        if is_colon {
            values.push(Value::String(OBJECT_SELECTOR_COLON.to_string()));
            continue;
        }
        if is_end {
            values.push(Value::String(OBJECT_SELECTOR_END.to_string()));
            continue;
        }
        if let Some(pos) = range_pos_by_dim[d] {
            let (raw_st, raw_sp) = range_params[pos];
            let st = if let Some(expr) = &range_start_exprs[pos] {
                encode_end_expr_value(expr)?
            } else {
                Value::Num(raw_st)
            };
            let sp = if let Some(expr) = &range_step_exprs[pos] {
                encode_end_expr_value(expr)?
            } else {
                Value::Num(raw_sp)
            };
            let off = &range_end_exprs[pos];
            values.push(build_end_range_descriptor(st, sp, off)?);
            continue;
        }
        let selector = numeric
            .get(num_iter)
            .ok_or(crate::interpreter::errors::mex(
                "MissingNumericIndex",
                "missing numeric index",
            ))?;
        num_iter += 1;
        values.push(normalize_object_numeric_selector(selector)?);
    }
    if num_iter != numeric.len() {
        return Err(crate::interpreter::errors::mex(
            "UnexpectedNumericIndex",
            "unexpected extra numeric index values",
        ));
    }
    Ok(values)
}

pub async fn build_expanded_args_from_specs<ExpandObjectAll, ExpandObjectIndices, FutAll, FutIdx>(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
    invalid_expand_all_msg: &str,
    invalid_expand_msg: &str,
    mut expand_object_all: ExpandObjectAll,
    mut expand_object_indices: ExpandObjectIndices,
) -> Result<Vec<Value>, RuntimeError>
where
    ExpandObjectAll: FnMut(Value) -> FutAll,
    ExpandObjectIndices: FnMut(Value, Vec<Value>) -> FutIdx,
    FutAll: Future<Output = Result<Vec<Value>, RuntimeError>>,
    FutIdx: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let mut temp: Vec<Value> = Vec::new();
    for spec in specs.iter().rev() {
        if spec.is_expand {
            let mut indices = Vec::with_capacity(spec.num_indices);
            for _ in 0..spec.num_indices {
                indices.push(stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?);
            }
            indices.reverse();
            let base = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?;

            let expanded = if spec.expand_all {
                match base {
                    Value::OutputList(outputs) => outputs,
                    Value::Cell(ca) => expand_all_cell(&ca)?,
                    other @ Value::Object(_) | other @ Value::HandleObject(_) => {
                        expand_object_all(other).await?
                    }
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "InvalidExpandAllTarget",
                            invalid_expand_all_msg,
                        ))
                    }
                }
            } else {
                match (base, indices.len()) {
                    (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => {
                        expand_cell_indices(&ca, &indices)?
                    }
                    (other @ Value::Object(_), _) | (other @ Value::HandleObject(_), _) => {
                        expand_object_indices(other, indices).await?
                    }
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "ExpandError",
                            invalid_expand_msg,
                        ))
                    }
                }
            };
            temp.extend(expanded.into_iter().rev());
        } else {
            temp.push(stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?);
        }
    }
    temp.reverse();
    Ok(temp)
}

#[cfg(test)]
mod tests {
    use super::{
        build_expanded_args_from_specs, build_object_paren_expr_selector_values,
        build_object_paren_selector_values, ObjectIndexDescriptor, ObjectIndexOp,
        ObjectIndexSelector, OBJECT_END_RANGE_TAG, OBJECT_PROTOCOL_KIND_BRACE,
        OBJECT_PROTOCOL_KIND_MEMBER, OBJECT_PROTOCOL_SUBSASGN, OBJECT_PROTOCOL_SUBSREF,
        OBJECT_SELECTOR_COLON, OBJECT_SELECTOR_END,
    };
    use crate::bytecode::ArgSpec;
    use crate::bytecode::EndExpr;
    use futures::executor::block_on;
    use runmat_builtins::{HandleRef, Value};

    #[test]
    fn object_index_descriptor_serializes_protocol_args_once() {
        let descriptor = ObjectIndexDescriptor::subsref_brace(
            Value::Num(1.0),
            ObjectIndexSelector::IndexValues {
                values: vec![Value::Num(2.0)],
            },
        );

        let (base, method, args) = descriptor
            .into_method_invocation()
            .expect("descriptor args");
        assert_eq!(base, Value::Num(1.0));
        assert_eq!(method, OBJECT_PROTOCOL_SUBSREF.to_string());
        assert_eq!(
            args[0],
            Value::String(OBJECT_PROTOCOL_KIND_BRACE.to_string())
        );
        match &args[1] {
            Value::Cell(cell) => assert_eq!((*cell.data[0]).clone(), Value::Num(2.0)),
            other => panic!("expected selector cell, got {other:?}"),
        }
    }

    #[test]
    fn object_member_descriptor_carries_rhs() {
        let descriptor = ObjectIndexDescriptor::member(
            Value::Num(1.0),
            ObjectIndexOp::Subsasgn,
            "field".to_string(),
            Some(Value::Num(9.0)),
        );

        let (base, method, args) = descriptor
            .into_method_invocation()
            .expect("descriptor args");
        assert_eq!(base, Value::Num(1.0));
        assert_eq!(method, OBJECT_PROTOCOL_SUBSASGN.to_string());
        assert_eq!(
            args[0],
            Value::String(OBJECT_PROTOCOL_KIND_MEMBER.to_string())
        );
        assert_eq!(args[1], Value::String("field".to_string()));
        assert_eq!(args[2], Value::Num(9.0));
    }

    #[test]
    fn object_paren_selector_values_preserve_colon_end_and_numeric_order() {
        let selectors = build_object_paren_selector_values(3, 0b001, 0b010, &[Value::Num(9.0)])
            .expect("selector values");
        assert_eq!(selectors.len(), 3);
        assert_eq!(
            selectors[0],
            Value::String(OBJECT_SELECTOR_COLON.to_string())
        );
        assert_eq!(selectors[1], Value::String(OBJECT_SELECTOR_END.to_string()));
        assert_eq!(selectors[2], Value::Num(9.0));
    }

    #[test]
    fn object_paren_selector_values_validate_numeric_arity() {
        let missing = build_object_paren_selector_values(2, 0, 0, &[Value::Num(1.0)])
            .expect_err("missing selector should fail");
        assert_eq!(missing.identifier(), Some("RunMat:MissingNumericIndex"));

        let extra =
            build_object_paren_selector_values(2, 0b01, 0, &[Value::Num(2.0), Value::Num(3.0)])
                .expect_err("extra selector should fail");
        assert_eq!(extra.identifier(), Some("RunMat:UnexpectedNumericIndex"));
    }

    #[test]
    fn object_paren_expr_selector_values_encode_end_expression_range_descriptors() {
        let selectors = build_object_paren_expr_selector_values(
            2,
            0,
            0,
            &[0],
            &[(1.0, 2.0)],
            &[Some(EndExpr::Sub(
                Box::new(EndExpr::End),
                Box::new(EndExpr::Const(1.0)),
            ))],
            &[None],
            &[EndExpr::End],
            &[Value::Num(4.0)],
        )
        .expect("expr selector values");

        assert_eq!(selectors.len(), 2);
        match &selectors[0] {
            Value::Cell(cell) => {
                assert_eq!(
                    (*cell.data[2]).clone(),
                    Value::String(OBJECT_END_RANGE_TAG.to_string())
                );
                assert_eq!((*cell.data[1]).clone(), Value::Num(2.0));
                assert_eq!(
                    (*cell.data[3]).clone(),
                    Value::String(OBJECT_SELECTOR_END.to_string())
                );
            }
            other => panic!("expected range descriptor cell, got {other:?}"),
        }
        assert_eq!(selectors[1], Value::Num(4.0));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_invalid_range_plan_metadata() {
        let err = build_object_paren_expr_selector_values(
            2,
            0,
            0,
            &[0],
            &[(1.0, 2.0)],
            &[None],
            &[None],
            &[],
            &[Value::Num(4.0)],
        )
        .expect_err("inconsistent range metadata should fail");
        assert_eq!(err.identifier(), Some("RunMat:InvalidRangeSelectorPlan"));
    }

    #[test]
    fn object_paren_expr_selector_values_reject_duplicate_range_dims() {
        let err = build_object_paren_expr_selector_values(
            2,
            0,
            0,
            &[0, 0],
            &[(1.0, 1.0), (2.0, 1.0)],
            &[None, None],
            &[None, None],
            &[EndExpr::End, EndExpr::End],
            &[],
        )
        .expect_err("duplicate range dimensions should fail");
        assert_eq!(err.identifier(), Some("RunMat:DuplicateRangeSelectorDim"));
    }

    #[test]
    fn build_expanded_args_from_specs_accepts_handle_object_expansion() {
        let target = runmat_gc::gc_allocate(Value::Num(7.0)).expect("handle target");
        let handle = HandleRef {
            class_name: "HandleThing".to_string(),
            target,
            valid: true,
        };
        let mut stack = vec![Value::HandleObject(handle)];
        let specs = vec![ArgSpec {
            is_expand: true,
            num_indices: 0,
            expand_all: true,
        }];
        let expanded = block_on(build_expanded_args_from_specs(
            &mut stack,
            &specs,
            "expand-all failed",
            "expand-indices failed",
            |base| async move {
                match base {
                    Value::HandleObject(_) => Ok(vec![Value::Num(42.0)]),
                    other => panic!("expected handle object expansion path, got {other:?}"),
                }
            },
            |_base, _indices| async move { Ok(vec![]) },
        ))
        .expect("expanded args");
        assert_eq!(expanded, vec![Value::Num(42.0)]);
    }
}
