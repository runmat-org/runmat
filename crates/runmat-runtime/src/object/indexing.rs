use crate::call::identity::strict_callable_display_name;
use crate::indexing::EndExpr;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
use runmat_value::Value;

pub const OBJECT_PROTOCOL_SUBSREF: &str = crate::OBJECT_SUBSREF_METHOD;
pub const OBJECT_PROTOCOL_SUBSASGN: &str = crate::OBJECT_SUBSASGN_METHOD;
pub const OBJECT_PROTOCOL_KIND_PAREN: &str = crate::OBJECT_INDEX_PAREN;
pub const OBJECT_PROTOCOL_KIND_BRACE: &str = crate::OBJECT_INDEX_BRACE;
pub const OBJECT_PROTOCOL_KIND_MEMBER: &str = crate::OBJECT_INDEX_MEMBER;
pub const OBJECT_SELECTOR_COLON: &str = ":";
pub const OBJECT_SELECTOR_END: &str = "end";
pub const OBJECT_END_RANGE_TAG: &str = "end_expr";

#[derive(Clone, Copy)]
pub enum ObjectIndexOp {
    Subsref,
    Subsasgn,
}

impl ObjectIndexOp {
    pub fn protocol_name(self) -> &'static str {
        match self {
            Self::Subsref => OBJECT_PROTOCOL_SUBSREF,
            Self::Subsasgn => OBJECT_PROTOCOL_SUBSASGN,
        }
    }
}

#[derive(Clone, Copy)]
pub enum ObjectIndexKind {
    Paren,
    Brace,
    Member,
}

impl ObjectIndexKind {
    pub fn protocol_name(self) -> &'static str {
        match self {
            Self::Paren => OBJECT_PROTOCOL_KIND_PAREN,
            Self::Brace => OBJECT_PROTOCOL_KIND_BRACE,
            Self::Member => OBJECT_PROTOCOL_KIND_MEMBER,
        }
    }
}

#[derive(Clone)]
pub enum ObjectIndexSelector {
    ScalarIndices { indices: Vec<usize> },
    IndexValues { values: Vec<Value> },
    Member(String),
}

#[derive(Clone)]
pub struct ObjectIndexDescriptor {
    base: Value,
    op: ObjectIndexOp,
    kind: ObjectIndexKind,
    selector: ObjectIndexSelector,
    rhs: Option<Value>,
}

#[derive(Debug, Clone, Copy)]
pub struct ObjectParenExprSelectorSpec<'a> {
    pub dims: usize,
    pub colon_mask: u32,
    pub end_mask: u32,
    pub range_dims: &'a [usize],
    pub range_params: &'a [(f64, f64)],
    pub range_start_exprs: &'a [Option<EndExpr>],
    pub range_step_exprs: &'a [Option<EndExpr>],
    pub range_end_exprs: &'a [EndExpr],
    pub end_numeric_exprs: &'a [(usize, EndExpr)],
    pub numeric: &'a [Value],
}

impl ObjectIndexDescriptor {
    pub fn subsref_paren(base: Value, selector: ObjectIndexSelector) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Paren,
            selector,
            rhs: None,
        }
    }

    pub fn subsref_brace(base: Value, selector: ObjectIndexSelector) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Brace,
            selector,
            rhs: None,
        }
    }

    pub fn subsasgn_paren(base: Value, selector: ObjectIndexSelector, rhs: Value) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Paren,
            selector,
            rhs: Some(rhs),
        }
    }

    pub fn subsasgn_brace(base: Value, selector: ObjectIndexSelector, rhs: Value) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Brace,
            selector,
            rhs: Some(rhs),
        }
    }

    pub fn subsref_paren_from_slice(
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

    pub fn subsasgn_paren_from_slice(
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

    pub fn subsasgn_paren_from_expr_slice(
        base: Value,
        spec: ObjectParenExprSelectorSpec<'_>,
        rhs: Value,
    ) -> Result<Self, RuntimeError> {
        let values = build_object_paren_expr_selector_values(spec)?;
        Ok(Self::subsasgn_paren(
            base,
            ObjectIndexSelector::IndexValues { values },
            rhs,
        ))
    }

    pub fn subsref_paren_from_expr_slice(
        base: Value,
        spec: ObjectParenExprSelectorSpec<'_>,
    ) -> Result<Self, RuntimeError> {
        let values = build_object_paren_expr_selector_values(spec)?;
        Ok(Self::subsref_paren(
            base,
            ObjectIndexSelector::IndexValues { values },
        ))
    }

    pub fn member(base: Value, op: ObjectIndexOp, field: String, rhs: Option<Value>) -> Self {
        Self {
            base,
            op,
            kind: ObjectIndexKind::Member,
            selector: ObjectIndexSelector::Member(field),
            rhs,
        }
    }

    pub fn base(&self) -> &Value {
        &self.base
    }

    pub fn operation(&self) -> ObjectIndexOp {
        self.op
    }

    pub fn rhs(&self) -> Option<&Value> {
        self.rhs.as_ref()
    }

    pub fn into_method_invocation(self) -> Result<(Value, String, Vec<Value>), RuntimeError> {
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
    let cell = build_cell_array_with_shape(values, 1, cols, "object index descriptor build")?;
    Ok(Value::Cell(cell))
}

fn matlab_index_type(kind: ObjectIndexKind) -> &'static str {
    match kind {
        ObjectIndexKind::Paren => "()",
        ObjectIndexKind::Brace => "{}",
        ObjectIndexKind::Member => ".",
    }
}

pub fn class_name_from_base(base: &Value) -> Option<&str> {
    match base {
        Value::Object(obj) => Some(obj.class_name.as_str()),
        Value::HandleObject(handle) => Some(handle.class_name.as_str()),
        _ => None,
    }
}

pub fn build_matlab_substruct_arg(
    descriptor: &ObjectIndexDescriptor,
) -> Result<Value, RuntimeError> {
    let subs_value = match &descriptor.selector {
        ObjectIndexSelector::ScalarIndices { indices } => {
            let values = indices
                .iter()
                .map(|index| Value::Num(*index as f64))
                .collect();
            build_protocol_index_cell(values)?
        }
        ObjectIndexSelector::IndexValues { values } => build_protocol_index_cell(values.clone())?,
        ObjectIndexSelector::Member(field) => Value::String(field.clone()),
    };
    let mut value = runmat_value::StructValue::new();
    value.fields.insert(
        "type".to_string(),
        Value::String(matlab_index_type(descriptor.kind).to_string()),
    );
    value.fields.insert("subs".to_string(), subs_value);
    Ok(Value::Struct(value))
}

fn encode_end_expr_value(expr: &EndExpr) -> Result<Value, RuntimeError> {
    fn mk_cell(items: Vec<Value>) -> Result<Value, RuntimeError> {
        let cols = items.len();
        let cell = build_cell_array_with_shape(items, 1, cols, "end expression encoding")?;
        Ok(Value::Cell(cell))
    }

    match expr {
        EndExpr::End => Ok(Value::String("end".to_string())),
        EndExpr::Const(v) => Ok(Value::Num(*v)),
        EndExpr::Var(i) => Ok(Value::String(format!("var:{i}"))),
        EndExpr::ResolvedCall { identity, args, .. } => {
            let name = strict_callable_display_name(identity).ok_or_else(|| {
                semantic_error(
                    "UndefinedFunction",
                    "end expression call missing callable name",
                )
            })?;
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
    let cell = build_cell_array_with_shape(
        vec![
            start,
            step,
            Value::String(OBJECT_END_RANGE_TAG.to_string()),
            encoded_end,
        ],
        1,
        4,
        "obj range",
    )?;
    Ok(Value::Cell(cell))
}

fn normalize_object_numeric_selector(selector: &Value) -> Result<Value, RuntimeError> {
    match selector {
        Value::Num(n) => Ok(Value::Num(*n)),
        Value::Int(i) => Ok(Value::Int(i.clone())),
        Value::Tensor(t) => Ok(Value::Tensor(t.clone())),
        Value::Bool(value) => Ok(Value::Bool(*value)),
        Value::LogicalArray(array) => Ok(Value::LogicalArray(array.clone())),
        Value::String(value) => Ok(Value::String(value.clone())),
        Value::StringArray(array) => Ok(Value::StringArray(array.clone())),
        Value::CharArray(array) => Ok(Value::CharArray(array.clone())),
        Value::Cell(cell) => Ok(Value::Cell(cell.clone())),
        _ => Err(semantic_error(
            "ObjectSelectorTypeUnsupported",
            "unsupported index type for object selector",
        )),
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
        return Err(semantic_error(
            "InvalidRangeSelectorPlan",
            "inconsistent object range selector metadata",
        ));
    }

    let mut range_pos_by_dim = vec![None; dims];
    for (pos, &dim) in range_dims.iter().enumerate() {
        if dim >= dims {
            return Err(semantic_error(
                "InvalidRangeSelectorDim",
                "object range selector dimension is out of bounds",
            ));
        }
        if range_pos_by_dim[dim].replace(pos).is_some() {
            return Err(semantic_error(
                "InvalidRangeSelectorPlan",
                "object range selector dimension appears more than once",
            ));
        }
    }
    Ok(range_pos_by_dim)
}

fn validate_object_end_numeric_selector_plan(
    slot_count: usize,
    end_numeric_exprs: &[(usize, EndExpr)],
) -> Result<Vec<Option<&EndExpr>>, RuntimeError> {
    let mut end_expr_by_slot = vec![None; slot_count];
    for (position, expr) in end_numeric_exprs {
        if *position >= slot_count {
            return Err(semantic_error(
                "InvalidEndSelectorPlan",
                "object end-selector position is out of bounds",
            ));
        }
        if end_expr_by_slot[*position].is_some() {
            return Err(semantic_error(
                "InvalidEndSelectorPlan",
                "object end-selector position appears more than once",
            ));
        }
        end_expr_by_slot[*position] = Some(expr);
    }
    Ok(end_expr_by_slot)
}

fn validate_object_selector_masks(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
) -> Result<(), RuntimeError> {
    if (colon_mask & end_mask) != 0 {
        return Err(semantic_error(
            "InvalidSelectorMaskPlan",
            "object selector masks overlap on the same dimension",
        ));
    }

    if dims < u32::BITS as usize {
        let allowed_mask = if dims == 0 { 0 } else { (1u32 << dims) - 1 };
        if ((colon_mask | end_mask) & !allowed_mask) != 0 {
            return Err(semantic_error(
                "InvalidSelectorMaskPlan",
                "object selector mask dimension is out of bounds",
            ));
        }
    }

    Ok(())
}

fn object_selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

pub fn build_object_paren_selector_values(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    validate_object_selector_masks(dims, colon_mask, end_mask)?;
    let mut values = Vec::with_capacity(dims);
    let mut numeric_iter = 0usize;
    for d in 0..dims {
        let is_colon = object_selector_mask_has_dim(colon_mask, d);
        let is_end = object_selector_mask_has_dim(end_mask, d);
        if is_colon {
            values.push(Value::String(OBJECT_SELECTOR_COLON.to_string()));
            continue;
        }
        if is_end {
            values.push(Value::String(OBJECT_SELECTOR_END.to_string()));
            continue;
        }
        let selector = numeric.get(numeric_iter).ok_or(semantic_error(
            "MissingNumericIndex",
            "missing numeric index",
        ))?;
        values.push(normalize_object_numeric_selector(selector)?);
        numeric_iter += 1;
    }
    if numeric_iter != numeric.len() {
        return Err(semantic_error(
            "UnexpectedNumericIndex",
            "unexpected extra numeric index values",
        ));
    }
    Ok(values)
}

pub fn build_object_paren_expr_selector_values(
    spec: ObjectParenExprSelectorSpec<'_>,
) -> Result<Vec<Value>, RuntimeError> {
    validate_object_selector_masks(spec.dims, spec.colon_mask, spec.end_mask)?;
    let range_pos_by_dim = validate_object_range_selector_plan(
        spec.dims,
        spec.range_dims,
        spec.range_params,
        spec.range_start_exprs,
        spec.range_step_exprs,
        spec.range_end_exprs,
    )?;
    for (d, range_pos) in range_pos_by_dim.iter().enumerate().take(spec.dims) {
        if range_pos.is_some() {
            let is_colon = object_selector_mask_has_dim(spec.colon_mask, d);
            let is_end = object_selector_mask_has_dim(spec.end_mask, d);
            if is_colon || is_end {
                return Err(semantic_error(
                    "InvalidRangeSelectorPlan",
                    "object range selector conflicts with colon/end selector masks",
                ));
            }
        }
    }
    let slot_count = (0..spec.dims)
        .filter(|&d| {
            let is_colon = object_selector_mask_has_dim(spec.colon_mask, d);
            let is_end = object_selector_mask_has_dim(spec.end_mask, d);
            !is_colon && !is_end && range_pos_by_dim[d].is_none()
        })
        .count();
    let end_expr_by_slot =
        validate_object_end_numeric_selector_plan(slot_count, spec.end_numeric_exprs)?;
    let mut values = Vec::with_capacity(spec.dims);
    let mut num_iter = 0usize;
    for (d, range_pos) in range_pos_by_dim.iter().enumerate().take(spec.dims) {
        let is_colon = object_selector_mask_has_dim(spec.colon_mask, d);
        let is_end = object_selector_mask_has_dim(spec.end_mask, d);
        if is_colon {
            values.push(Value::String(OBJECT_SELECTOR_COLON.to_string()));
            continue;
        }
        if is_end {
            values.push(Value::String(OBJECT_SELECTOR_END.to_string()));
            continue;
        }
        if let Some(pos) = *range_pos {
            let (raw_st, raw_sp) = spec.range_params[pos];
            let st = if let Some(expr) = &spec.range_start_exprs[pos] {
                encode_end_expr_value(expr)?
            } else {
                Value::Num(raw_st)
            };
            let sp = if let Some(expr) = &spec.range_step_exprs[pos] {
                encode_end_expr_value(expr)?
            } else {
                Value::Num(raw_sp)
            };
            let off = &spec.range_end_exprs[pos];
            values.push(build_end_range_descriptor(st, sp, off)?);
            continue;
        }
        if let Some(expr) = end_expr_by_slot[num_iter] {
            values.push(encode_end_expr_value(expr)?);
            num_iter += 1;
            continue;
        }
        let selector = spec.numeric.get(num_iter).ok_or(semantic_error(
            "MissingNumericIndex",
            "missing numeric index",
        ))?;
        num_iter += 1;
        values.push(normalize_object_numeric_selector(selector)?);
    }
    if num_iter != spec.numeric.len() {
        return Err(semantic_error(
            "UnexpectedNumericIndex",
            "unexpected extra numeric index values",
        ));
    }
    Ok(values)
}

fn build_cell_array_with_shape(
    values: Vec<Value>,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<runmat_value::CellArray, RuntimeError> {
    runmat_value::CellArray::new(values, rows, cols)
        .map_err(|error| semantic_error("ShapeMismatch", format!("{context}: {error}")))
}
