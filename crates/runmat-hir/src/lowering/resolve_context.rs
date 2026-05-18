use crate::hir::{CompatibilityHirExpr as HirExpr, CompatibilityHirExprKind as HirExprKind};
#[cfg(test)]
use crate::Type;
use runmat_parser as parser;
use std::collections::HashMap;
use std::sync::OnceLock;

static CONST_NUM_LOOKUP: OnceLock<HashMap<String, f64>> = OnceLock::new();

pub(crate) fn resolve_context_from_args(args: &[HirExpr]) -> runmat_builtins::ResolveContext {
    let literal_args = args.iter().map(literal_value_from_expr).collect();
    runmat_builtins::ResolveContext::new(literal_args)
}

fn literal_value_from_expr(expr: &HirExpr) -> runmat_builtins::LiteralValue {
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

fn eval_const_num(expr: &HirExpr) -> Option<f64> {
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
    fn resolve_context_preserves_argument_literals() {
        let args = vec![
            expr(HirExprKind::Number("3".to_string())),
            expr(HirExprKind::String("all".to_string())),
        ];
        let ctx = resolve_context_from_args(&args);
        assert_eq!(
            ctx.literal_args,
            vec![
                LiteralValue::Number(3.0),
                LiteralValue::String("all".to_string())
            ]
        );
    }
}
