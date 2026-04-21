use crate::{HirClassMember, HirExpr, HirExprKind, HirLValue, HirProgram, HirStmt, Type, VarId};
use runmat_parser as parser;
use std::collections::HashMap;
use std::sync::OnceLock;

static CONST_NUM_LOOKUP: OnceLock<HashMap<String, f64>> = OnceLock::new();

pub(crate) type FuncDef = (Vec<VarId>, Vec<VarId>, Vec<HirStmt>);

#[derive(Clone)]
pub(crate) struct Analysis {
    pub(crate) exits: Vec<HashMap<VarId, Type>>,
    pub(crate) fallthrough: Option<HashMap<VarId, Type>>,
}

pub(crate) fn join_env(a: &HashMap<VarId, Type>, b: &HashMap<VarId, Type>) -> HashMap<VarId, Type> {
    let mut out = a.clone();
    for (k, v) in b {
        out.entry(*k)
            .and_modify(|t| *t = t.unify(v))
            .or_insert_with(|| v.clone());
    }
    out
}

pub(crate) fn collect_function_defs(prog: &HirProgram) -> HashMap<String, FuncDef> {
    let mut func_defs: HashMap<String, FuncDef> = HashMap::new();
    for stmt in &prog.body {
        if let HirStmt::Function {
            name,
            params,
            outputs,
            body,
            ..
        } = stmt
        {
            func_defs.insert(
                name.clone(),
                (params.clone(), outputs.clone(), body.clone()),
            );
        } else if let HirStmt::ClassDef { members, .. } = stmt {
            for m in members {
                if let HirClassMember::Methods { body, .. } = m {
                    for s in body {
                        if let HirStmt::Function {
                            name,
                            params,
                            outputs,
                            body,
                            ..
                        } = s
                        {
                            func_defs.insert(
                                name.clone(),
                                (params.clone(), outputs.clone(), body.clone()),
                            );
                        }
                    }
                }
            }
        }
    }
    func_defs
}

pub(crate) fn refine_multi_assign_outputs_from_func(
    name: &str,
    per_out: &mut Vec<Type>,
    returns: &HashMap<String, Vec<Type>>,
    func_defs: &HashMap<String, FuncDef>,
    infer_expr: impl Fn(&HirExpr, &HashMap<VarId, Type>, &HashMap<String, Vec<Type>>) -> Type,
) {
    let needs_fallback = per_out.is_empty() || per_out.iter().any(|t| matches!(t, Type::Unknown));
    if !needs_fallback {
        return;
    }

    if let Some((params, outs, body)) = func_defs.get(name).cloned() {
        let mut penv: HashMap<VarId, Type> = HashMap::new();
        for p in params {
            penv.insert(p, Type::Num);
        }
        let mut out_types: Vec<Type> = vec![Type::Unknown; outs.len()];
        for s in &body {
            if let HirStmt::Assign(var, rhs, _, _) = s {
                if let Some(pos) = outs.iter().position(|o| o == var) {
                    let t = infer_expr(rhs, &penv, returns);
                    out_types[pos] = out_types[pos].unify(&t);
                }
            }
        }
        if per_out.is_empty() {
            *per_out = out_types;
        } else {
            for (i, t) in out_types.into_iter().enumerate() {
                if matches!(per_out.get(i), Some(Type::Unknown)) {
                    if let Some(slot) = per_out.get_mut(i) {
                        *slot = t;
                    }
                }
            }
        }
    }
}

pub(crate) fn apply_struct_field_assertions(
    env: &mut HashMap<VarId, Type>,
    assertions: impl IntoIterator<Item = (VarId, String)>,
) {
    for (vid, field) in assertions {
        let mut known = match env.get(&vid) {
            Some(Type::Struct { known_fields }) => known_fields.clone(),
            _ => Some(Vec::new()),
        };
        if let Some(list) = &mut known {
            if !list.iter().any(|f| f == &field) {
                list.push(field);
                list.sort();
                list.dedup();
            }
        }
        env.insert(
            vid,
            Type::Struct {
                known_fields: known,
            },
        );
    }
}

pub(crate) fn collect_struct_field_assertions(e: &HirExpr, out: &mut Vec<(VarId, String)>) {
    use HirExprKind as K;

    fn trim_quotes(s: &str) -> String {
        let t = s.trim();
        t.trim_matches('\'').to_string()
    }

    fn extract_field_literal(e: &HirExpr) -> Option<String> {
        match &e.kind {
            HirExprKind::String(s) => Some(trim_quotes(s)),
            _ => None,
        }
    }

    fn extract_field_list(e: &HirExpr) -> Vec<String> {
        match &e.kind {
            HirExprKind::String(s) => vec![trim_quotes(s)],
            HirExprKind::Cell(rows) => {
                let mut out = Vec::new();
                for row in rows {
                    for it in row {
                        if let Some(v) = extract_field_literal(it) {
                            out.push(v);
                        }
                    }
                }
                out
            }
            _ => Vec::new(),
        }
    }

    match &e.kind {
        K::Unary(parser::UnOp::Not, _inner) => {}
        K::Binary(left, parser::BinOp::AndAnd, right)
        | K::Binary(left, parser::BinOp::BitAnd, right) => {
            collect_struct_field_assertions(left, out);
            collect_struct_field_assertions(right, out);
        }
        K::FuncCall(name, args) => {
            let lname = name.as_str();
            if lname.eq_ignore_ascii_case("isfield") && args.len() >= 2 {
                if let HirExprKind::Var(vid) = args[0].kind {
                    if let Some(f) = extract_field_literal(&args[1]) {
                        out.push((vid, f));
                    }
                }
            }
            if lname.eq_ignore_ascii_case("ismember") && args.len() >= 2 {
                let mut fields: Vec<String> = Vec::new();
                let mut target: Option<VarId> = None;
                if let HirExprKind::FuncCall(ref n0, ref a0) = args[0].kind {
                    if n0.eq_ignore_ascii_case("fieldnames") && a0.len() == 1 {
                        if let HirExprKind::Var(vid) = a0[0].kind {
                            target = Some(vid);
                        }
                    }
                }
                if let HirExprKind::FuncCall(ref n1, ref a1) = args[1].kind {
                    if n1.eq_ignore_ascii_case("fieldnames") && a1.len() == 1 {
                        if let HirExprKind::Var(vid) = a1[0].kind {
                            target = Some(vid);
                        }
                    }
                }
                if fields.is_empty() {
                    fields.extend(extract_field_list(&args[0]));
                }
                if fields.is_empty() {
                    fields.extend(extract_field_list(&args[1]));
                }
                if let Some(vid) = target {
                    for f in fields {
                        out.push((vid, f));
                    }
                }
            }
            if (lname.eq_ignore_ascii_case("any") || lname.eq_ignore_ascii_case("all"))
                && !args.is_empty()
            {
                collect_struct_field_assertions(&args[0], out);
            }
            if (lname.eq_ignore_ascii_case("strcmp") || lname.eq_ignore_ascii_case("strcmpi"))
                && args.len() >= 2
            {
                let mut target: Option<VarId> = None;
                if let HirExprKind::FuncCall(ref n0, ref a0) = args[0].kind {
                    if n0.eq_ignore_ascii_case("fieldnames") && a0.len() == 1 {
                        if let HirExprKind::Var(vid) = a0[0].kind {
                            target = Some(vid);
                        }
                    }
                }
                if let HirExprKind::FuncCall(ref n1, ref a1) = args[1].kind {
                    if n1.eq_ignore_ascii_case("fieldnames") && a1.len() == 1 {
                        if let HirExprKind::Var(vid) = a1[0].kind {
                            target = Some(vid);
                        }
                    }
                }
                let mut fields = Vec::new();
                fields.extend(extract_field_list(&args[0]));
                fields.extend(extract_field_list(&args[1]));
                if let Some(vid) = target {
                    for f in fields {
                        out.push((vid, f));
                    }
                }
            }
        }
        _ => {}
    }
}

pub(crate) fn apply_lvalue_type_effects(env: &mut HashMap<VarId, Type>, lv: &HirLValue) {
    if let HirLValue::Member(base, field) = lv {
        if let HirExprKind::Var(vid) = base.kind {
            apply_struct_field_assertions(env, [(vid, field.clone())]);
        }
    }
}

pub(crate) fn logical_binary_result(lhs: &Type, rhs: &Type) -> Type {
    match (lhs, rhs) {
        (Type::Tensor { shape: Some(a) }, Type::Tensor { shape: Some(b) })
        | (Type::Logical { shape: Some(a) }, Type::Logical { shape: Some(b) })
        | (Type::Tensor { shape: Some(a) }, Type::Logical { shape: Some(b) })
        | (Type::Logical { shape: Some(a) }, Type::Tensor { shape: Some(b) }) => Type::Logical {
            shape: Some(runmat_builtins::shape_rules::broadcast_shapes(a, b)),
        },
        (Type::Tensor { shape: Some(a) }, Type::Num)
        | (Type::Tensor { shape: Some(a) }, Type::Int)
        | (Type::Tensor { shape: Some(a) }, Type::Bool)
        | (Type::Logical { shape: Some(a) }, Type::Num)
        | (Type::Logical { shape: Some(a) }, Type::Int)
        | (Type::Logical { shape: Some(a) }, Type::Bool)
        | (Type::Num, Type::Tensor { shape: Some(a) })
        | (Type::Int, Type::Tensor { shape: Some(a) })
        | (Type::Bool, Type::Tensor { shape: Some(a) })
        | (Type::Num, Type::Logical { shape: Some(a) })
        | (Type::Int, Type::Logical { shape: Some(a) })
        | (Type::Bool, Type::Logical { shape: Some(a) }) => Type::Logical {
            shape: Some(a.clone()),
        },
        (Type::Tensor { .. }, _)
        | (_, Type::Tensor { .. })
        | (Type::Logical { .. }, _)
        | (_, Type::Logical { .. }) => Type::logical(),
        _ => Type::Bool,
    }
}

pub fn eval_const_num(expr: &HirExpr) -> Option<f64> {
    fn numeric_value(value: &runmat_builtins::Value) -> Option<f64> {
        use runmat_builtins::IntValue;

        match value {
            runmat_builtins::Value::Num(v) => Some(*v),
            runmat_builtins::Value::Int(int_value) => match int_value {
                IntValue::I8(v) => Some(*v as f64),
                IntValue::I16(v) => Some(*v as f64),
                IntValue::I32(v) => Some(*v as f64),
                IntValue::I64(v) => Some(*v as f64),
                IntValue::U8(v) => Some(*v as f64),
                IntValue::U16(v) => Some(*v as f64),
                IntValue::U32(v) => Some(*v as f64),
                IntValue::U64(v) => Some(*v as f64),
            },
            runmat_builtins::Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    fn const_numeric_value(name: &str) -> Option<f64> {
        let map = CONST_NUM_LOOKUP.get_or_init(|| {
            let mut out: HashMap<String, f64> = HashMap::new();
            for constant in runmat_builtins::constants() {
                if let Some(value) = numeric_value(&constant.value) {
                    out.insert(constant.name.to_ascii_lowercase(), value);
                }
            }
            out
        });
        map.get(&name.to_ascii_lowercase()).copied()
    }

    match &expr.kind {
        HirExprKind::Number(text) => text.parse::<f64>().ok(),
        HirExprKind::Constant(name) => const_numeric_value(name),
        HirExprKind::Unary(op, inner) => {
            let value = eval_const_num(inner)?;
            match op {
                parser::UnOp::Plus => Some(value),
                parser::UnOp::Minus => Some(-value),
                _ => None,
            }
        }
        HirExprKind::Binary(lhs, op, rhs) => {
            let a = eval_const_num(lhs)?;
            let b = eval_const_num(rhs)?;
            match op {
                parser::BinOp::Add => Some(a + b),
                parser::BinOp::Sub => Some(a - b),
                parser::BinOp::Mul | parser::BinOp::ElemMul => Some(a * b),
                parser::BinOp::RightDiv | parser::BinOp::ElemDiv => Some(a / b),
                parser::BinOp::LeftDiv | parser::BinOp::ElemLeftDiv => Some(b / a),
                parser::BinOp::Pow | parser::BinOp::ElemPow => Some(a.powf(b)),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(crate) fn literal_value_from_expr(expr: &HirExpr) -> runmat_builtins::LiteralValue {
    use runmat_builtins::LiteralValue;

    match &expr.kind {
        HirExprKind::String(text) => LiteralValue::String(text.clone()),
        HirExprKind::Constant(name) => match name.to_ascii_lowercase().as_str() {
            "true" => LiteralValue::Bool(true),
            "false" => LiteralValue::Bool(false),
            _ => eval_const_num(expr)
                .map(LiteralValue::Number)
                .unwrap_or(LiteralValue::Unknown),
        },
        HirExprKind::Tensor(rows) => literal_vector_from_tensor(rows),
        _ => eval_const_num(expr)
            .map(LiteralValue::Number)
            .unwrap_or(LiteralValue::Unknown),
    }
}

fn literal_vector_from_tensor(rows: &[Vec<HirExpr>]) -> runmat_builtins::LiteralValue {
    use runmat_builtins::LiteralValue;

    if rows.is_empty() {
        return LiteralValue::Vector(Vec::new());
    }
    let is_row = rows.len() == 1;
    let is_column = rows.iter().all(|row| row.len() == 1);

    if is_row {
        let values = rows[0].iter().map(literal_value_from_expr).collect();
        return LiteralValue::Vector(values);
    }

    if is_column {
        let values = rows
            .iter()
            .map(|row| literal_value_from_expr(&row[0]))
            .collect();
        return LiteralValue::Vector(values);
    }

    let values = rows
        .iter()
        .map(|row| LiteralValue::Vector(row.iter().map(literal_value_from_expr).collect()))
        .collect();
    LiteralValue::Vector(values)
}

pub(crate) fn resolve_context_from_args(args: &[HirExpr]) -> runmat_builtins::ResolveContext {
    let literal_args = args.iter().map(literal_value_from_expr).collect();
    runmat_builtins::ResolveContext::new(literal_args)
}

fn unquote_matlab_string(text: &str) -> String {
    if text.len() >= 2 {
        let bytes = text.as_bytes();
        if (bytes[0] == b'\'' && bytes[text.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[text.len() - 1] == b'"')
        {
            return text[1..text.len() - 1].to_string();
        }
    }
    text.to_string()
}

pub(crate) fn literal_path_arg(expr: &HirExpr) -> Option<String> {
    if let HirExprKind::String(text) = &expr.kind {
        return Some(unquote_matlab_string(text));
    }
    None
}

pub(crate) fn infer_dataset_type_from_literal_path(path: &str) -> Option<Type> {
    let manifest_path = if path.ends_with(".json") {
        std::path::PathBuf::from(path)
    } else {
        std::path::PathBuf::from(path).join("manifest.json")
    };
    let bytes = std::fs::read(&manifest_path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let arrays = value.get("arrays")?.as_object()?;
    let mut out = std::collections::BTreeMap::new();
    for (name, meta) in arrays {
        let shape = meta.get("shape").and_then(|v| v.as_array()).map(|dims| {
            dims.iter()
                .map(|d| d.as_u64().map(|n| n as usize))
                .collect::<Vec<_>>()
        });
        let chunk_shape = meta
            .get("chunk_shape")
            .or_else(|| meta.get("chunkShape"))
            .and_then(|v| v.as_array())
            .map(|dims| {
                dims.iter()
                    .map(|d| d.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            });
        let dtype = meta
            .get("dtype")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let codec = meta
            .get("codec")
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        out.insert(
            name.clone(),
            runmat_builtins::DataArrayTypeInfo {
                dtype,
                shape,
                chunk_shape,
                codec,
            },
        );
    }
    Some(Type::DataDataset { arrays: Some(out) })
}

pub(crate) fn shape_rank(shape: &Option<Vec<Option<usize>>>) -> Option<usize> {
    shape.as_ref().map(Vec::len)
}

pub(crate) fn infer_slice_result_shape(
    array_shape: &Option<Vec<Option<usize>>>,
    slice_expr: &HirExpr,
) -> Option<Vec<Option<usize>>> {
    let dims = array_shape.as_ref()?;
    match &slice_expr.kind {
        HirExprKind::Cell(rows) => {
            let selectors = rows.iter().flat_map(|row| row.iter()).collect::<Vec<_>>();
            let mut out = Vec::new();
            for (idx, selector) in selectors.iter().enumerate() {
                let base_dim = dims.get(idx).cloned().unwrap_or(None);
                match &selector.kind {
                    HirExprKind::Number(_) => {}
                    HirExprKind::Colon => out.push(base_dim),
                    HirExprKind::Tensor(trows) => {
                        let flat = trows.iter().flat_map(|r| r.iter()).collect::<Vec<_>>();
                        if flat.len() == 2 {
                            let start = eval_const_num(flat[0]);
                            let end = eval_const_num(flat[1]);
                            if let (Some(s), Some(e)) = (start, end) {
                                if e >= s {
                                    out.push(Some((e - s + 1.0) as usize));
                                    continue;
                                }
                            }
                        }
                        out.push(None);
                    }
                    _ => out.push(None),
                }
            }
            for dim in dims.iter().skip(selectors.len()) {
                out.push(*dim);
            }
            if out.is_empty() {
                Some(vec![Some(1), Some(1)])
            } else {
                Some(out)
            }
        }
        HirExprKind::Colon => Some(dims.clone()),
        HirExprKind::Number(_) => Some(vec![Some(1), Some(1)]),
        _ => Some(dims.clone()),
    }
}

#[cfg(test)]
mod literal_context_tests {
    use super::*;
    use crate::Span;
    use runmat_builtins::LiteralValue;

    fn expr(kind: HirExprKind) -> HirExpr {
        HirExpr {
            kind,
            ty: Type::Unknown,
            span: Span::default(),
        }
    }

    #[test]
    fn literal_from_number() {
        let value = expr(HirExprKind::Number("3.5".to_string()));
        assert_eq!(literal_value_from_expr(&value), LiteralValue::Number(3.5));
    }

    #[test]
    fn literal_from_string() {
        let value = expr(HirExprKind::String("All".to_string()));
        assert_eq!(
            literal_value_from_expr(&value),
            LiteralValue::String("All".to_string())
        );
    }

    #[test]
    fn literal_from_bool_constant() {
        let value = expr(HirExprKind::Constant("true".to_string()));
        assert_eq!(literal_value_from_expr(&value), LiteralValue::Bool(true));
    }

    #[test]
    fn literal_from_row_vector() {
        let value = expr(HirExprKind::Tensor(vec![vec![
            expr(HirExprKind::Number("1".to_string())),
            expr(HirExprKind::Number("2".to_string())),
        ]]));
        assert_eq!(
            literal_value_from_expr(&value),
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)])
        );
    }

    #[test]
    fn literal_from_column_vector() {
        let value = expr(HirExprKind::Tensor(vec![
            vec![expr(HirExprKind::Number("1".to_string()))],
            vec![expr(HirExprKind::Number("2".to_string()))],
        ]));
        assert_eq!(
            literal_value_from_expr(&value),
            LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)])
        );
    }

    #[test]
    fn literal_from_matrix_literal_is_nested_vector() {
        let value = expr(HirExprKind::Tensor(vec![
            vec![
                expr(HirExprKind::Number("1".to_string())),
                expr(HirExprKind::Number("2".to_string())),
            ],
            vec![
                expr(HirExprKind::Number("3".to_string())),
                expr(HirExprKind::Number("4".to_string())),
            ],
        ]));
        assert_eq!(
            literal_value_from_expr(&value),
            LiteralValue::Vector(vec![
                LiteralValue::Vector(vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)]),
                LiteralValue::Vector(vec![LiteralValue::Number(3.0), LiteralValue::Number(4.0)]),
            ])
        );
    }

    #[test]
    fn resolve_context_tracks_literals() {
        let args = vec![
            expr(HirExprKind::Number("4".to_string())),
            expr(HirExprKind::String("omitnan".to_string())),
            expr(HirExprKind::Constant("false".to_string())),
        ];
        let ctx = resolve_context_from_args(&args);
        assert_eq!(
            ctx.literal_args,
            vec![
                LiteralValue::Number(4.0),
                LiteralValue::String("omitnan".to_string()),
                LiteralValue::Bool(false),
            ]
        );
    }

    #[test]
    fn resolve_context_exposes_boolean_literals() {
        let args = vec![expr(HirExprKind::Constant("true".to_string()))];
        let ctx = resolve_context_from_args(&args);
        assert_eq!(ctx.literal_bool_at(0), Some(true));
    }
}
