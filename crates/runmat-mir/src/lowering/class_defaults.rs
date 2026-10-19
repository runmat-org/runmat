use crate::{MirClassDefaultLiteral, MirClassPropertyDefault};
use runmat_builtins::BuiltinConstantLiteral;
use runmat_hir::{HirAssembly, HirExpr, HirExprKind};

pub(super) fn project(hir: &HirAssembly) -> Vec<MirClassPropertyDefault> {
    let mut defaults = Vec::new();
    for class in &hir.classes {
        for property in &class.declaration.properties {
            if let Some(literal) = class
                .property_default(&property.name)
                .and_then(project_literal)
            {
                defaults.push(MirClassPropertyDefault {
                    class: class.declaration.id,
                    property: property.name.clone(),
                    literal,
                });
            }
        }
    }
    defaults
        .sort_by(|left, right| (left.class, &left.property).cmp(&(right.class, &right.property)));
    defaults
}

fn project_literal(expr: &HirExpr) -> Option<MirClassDefaultLiteral> {
    fn eval_numeric(expr: &HirExpr) -> Option<f64> {
        match &expr.kind {
            HirExprKind::Number(text) => text.parse::<f64>().ok().filter(|value| value.is_finite()),
            HirExprKind::Unary(runmat_hir::OperatorKind::UnaryPlus, inner) => eval_numeric(inner),
            HirExprKind::Unary(runmat_hir::OperatorKind::UnaryMinus, inner) => {
                eval_numeric(inner).map(|value| -value)
            }
            HirExprKind::Binary(left, op, right) => {
                let left = eval_numeric(left)?;
                let right = eval_numeric(right)?;
                let value = match op {
                    runmat_hir::OperatorKind::Add => Some(left + right),
                    runmat_hir::OperatorKind::Subtract => Some(left - right),
                    runmat_hir::OperatorKind::MatrixMultiply
                    | runmat_hir::OperatorKind::ElementwiseMultiply => Some(left * right),
                    runmat_hir::OperatorKind::Mrdivide
                    | runmat_hir::OperatorKind::ElementwiseDivide => Some(left / right),
                    runmat_hir::OperatorKind::Mldivide
                    | runmat_hir::OperatorKind::ElementwiseLeftDivide => Some(right / left),
                    runmat_hir::OperatorKind::MatrixPower
                    | runmat_hir::OperatorKind::ElementwisePower
                        if left < 0.0 && right.fract() != 0.0 =>
                    {
                        None
                    }
                    runmat_hir::OperatorKind::MatrixPower
                    | runmat_hir::OperatorKind::ElementwisePower => Some(left.powf(right)),
                    _ => None,
                };
                value.filter(|value| value.is_finite())
            }
            _ => None,
        }
    }

    match &expr.kind {
        HirExprKind::Number(_) => eval_numeric(expr).map(MirClassDefaultLiteral::Num),
        HirExprKind::IntegerLiteral(value) => Some(MirClassDefaultLiteral::Int(value.clone())),
        HirExprKind::String(text) => Some(MirClassDefaultLiteral::String(text.0.clone())),
        HirExprKind::Constant(name) => {
            runmat_builtins::builtin_constant_catalog_entry_by_name(&name.0)
                .and_then(|entry| entry.literal)
                .map(|literal| match literal {
                    BuiltinConstantLiteral::Logical(value) => MirClassDefaultLiteral::Bool(value),
                })
        }
        _ => eval_numeric(expr).map(MirClassDefaultLiteral::Num),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_division_default_uses_left_divisor() {
        let ast = runmat_parser::parse(
            "classdef NumericDefault\n properties\n left = 8 \\ 2;\n right = 8 / 2;\n end\nend",
        )
        .expect("class syntax");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("class lowering")
            .assembly;
        let defaults = project(&hir);
        assert!(defaults.iter().any(|default| {
            default.property.display_name() == "left"
                && default.literal == MirClassDefaultLiteral::Num(0.25)
        }));
        assert!(defaults.iter().any(|default| {
            default.property.display_name() == "right"
                && default.literal == MirClassDefaultLiteral::Num(4.0)
        }));
    }

    #[test]
    fn unsupported_complex_or_nonfinite_numeric_defaults_are_not_folded_to_real_values() {
        let ast = runmat_parser::parse(
            "classdef NumericDefault\n properties\n complexValue = (-1)^0.5;\n overflowValue = 1e308 * 1e308;\n realValue = (-2)^2;\n end\nend",
        )
        .expect("class syntax");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("class lowering")
            .assembly;
        let defaults = project(&hir);
        assert!(!defaults.iter().any(|default| {
            matches!(
                default.property.display_name(),
                "complexValue" | "overflowValue"
            )
        }));
        assert!(defaults.iter().any(|default| {
            default.property.display_name() == "realValue"
                && default.literal == MirClassDefaultLiteral::Num(4.0)
        }));
    }

    #[test]
    fn logical_defaults_use_the_catalog_literal_not_constant_name_spelling() {
        let ast = runmat_parser::parse(
            "classdef LogicalDefault\n properties\n enabled = true;\n disabled = false;\n end\nend",
        )
        .expect("class syntax");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("class lowering")
            .assembly;
        let defaults = project(&hir);
        assert!(defaults.iter().any(|default| {
            default.property.display_name() == "enabled"
                && default.literal == MirClassDefaultLiteral::Bool(true)
        }));
        assert!(defaults.iter().any(|default| {
            default.property.display_name() == "disabled"
                && default.literal == MirClassDefaultLiteral::Bool(false)
        }));
    }
}
