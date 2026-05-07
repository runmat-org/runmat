use std::collections::HashMap;

use runmat_builtins::Value;

use crate::{
    approximate_size_bytes, matlab_class_name, numeric_dtype_label, preview_numeric_values,
    value_shape,
};

use super::{WorkspaceEntry, WorkspacePreview, WorkspaceResidency};

const WORKSPACE_PREVIEW_LIMIT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FinalStmtEmitDisposition {
    Inline,
    #[allow(dead_code)]
    NeedsFallback,
    Suppressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LegacyDisplayContext {
    pub single_assign_var: Option<usize>,
    pub single_stmt_non_assign: bool,
    pub final_stmt_emit: FinalStmtEmitDisposition,
    pub last_assign_var: Option<usize>,
    pub last_expr_emits: bool,
}

pub(crate) fn determine_display_label_from_context(
    single_assign_var: Option<usize>,
    id_to_name: &HashMap<usize, String>,
    is_expression_stmt: bool,
    single_stmt_non_assign: bool,
) -> Option<String> {
    if let Some(var_id) = single_assign_var {
        id_to_name.get(&var_id).cloned()
    } else if is_expression_stmt || single_stmt_non_assign {
        Some("ans".to_string())
    } else {
        None
    }
}

/// Format value type information like MATLAB (e.g., "1000x1 vector", "3x3 matrix")
pub(crate) fn format_type_info(value: &Value) -> String {
    match value {
        Value::Int(_) => "scalar".to_string(),
        Value::Num(_) => "scalar".to_string(),
        Value::Bool(_) => "logical scalar".to_string(),
        Value::String(_) => "string".to_string(),
        Value::StringArray(sa) => {
            // MATLAB displays string arrays as m x n string array; for test's purpose, we classify scalar string arrays as "string"
            if sa.shape == vec![1, 1] {
                "string".to_string()
            } else if sa.shape.len() > 2 {
                let dims: Vec<String> = sa.shape.iter().map(|d| d.to_string()).collect();
                format!("{} string array", dims.join("x"))
            } else {
                format!("{}x{} string array", sa.rows(), sa.cols())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 && ca.cols == 1 {
                "char".to_string()
            } else {
                format!("{}x{} char array", ca.rows, ca.cols)
            }
        }
        Value::Tensor(m) => {
            if m.rows() == 1 && m.cols() == 1 {
                "scalar".to_string()
            } else if m.rows() == 1 || m.cols() == 1 {
                format!("{}x{} vector", m.rows(), m.cols())
            } else {
                format!("{}x{} matrix", m.rows(), m.cols())
            }
        }
        Value::Cell(cells) => {
            if cells.data.len() == 1 {
                "1x1 cell".to_string()
            } else {
                format!("{}x1 cell array", cells.data.len())
            }
        }
        Value::GpuTensor(h) => {
            if h.shape.len() == 2 {
                let r = h.shape[0];
                let c = h.shape[1];
                if r == 1 && c == 1 {
                    "scalar (gpu)".to_string()
                } else if r == 1 || c == 1 {
                    format!("{r}x{c} vector (gpu)")
                } else {
                    format!("{r}x{c} matrix (gpu)")
                }
            } else {
                format!("Tensor{:?} (gpu)", h.shape)
            }
        }
        _ => "value".to_string(),
    }
}

pub(crate) fn legacy_display_context(body: &[runmat_hir::LegacyHirStmt]) -> LegacyDisplayContext {
    let (single_assign_var, single_stmt_non_assign) = if body.len() == 1 {
        match &body[0] {
            runmat_hir::LegacyHirStmt::Assign(var_id, _, _, _) => (Some(var_id.0), false),
            _ => (None, true),
        }
    } else {
        (None, false)
    };

    LegacyDisplayContext {
        single_assign_var,
        single_stmt_non_assign,
        final_stmt_emit: legacy_last_displayable_statement_emit_disposition(body),
        last_assign_var: legacy_last_unsuppressed_assign_var(body),
        last_expr_emits: legacy_last_expr_emits_value(body),
    }
}

pub(crate) fn legacy_first_assign_var(body: &[runmat_hir::LegacyHirStmt]) -> Option<usize> {
    match body.first() {
        Some(runmat_hir::LegacyHirStmt::Assign(var_id, _, _, _)) => Some(var_id.0),
        _ => None,
    }
}

pub(crate) fn legacy_display_var_ids(body: &[runmat_hir::LegacyHirStmt]) -> Vec<usize> {
    use runmat_hir::{LegacyHirExprKind as HirExprKind, LegacyHirStmt as HirStmt};

    body.iter()
        .filter_map(|stmt| match stmt {
            HirStmt::Assign(var_id, _, suppressed, _) if !*suppressed => Some(var_id.0),
            HirStmt::ExprStmt(expr, suppressed, _) if !*suppressed => match &expr.kind {
                HirExprKind::Var(var_id) => Some(var_id.0),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn legacy_last_displayable_statement_emit_disposition(
    body: &[runmat_hir::LegacyHirStmt],
) -> FinalStmtEmitDisposition {
    use runmat_hir::LegacyHirStmt as HirStmt;

    for stmt in body.iter().rev() {
        match stmt {
            HirStmt::ExprStmt(expr, _, _) => return expr_emit_disposition(expr),
            HirStmt::Assign(_, _, _, _) | HirStmt::MultiAssign(_, _, _, _) => {
                return FinalStmtEmitDisposition::Suppressed
            }
            HirStmt::AssignLValue(_, _, _, _) => return FinalStmtEmitDisposition::Suppressed,

            _ => continue,
        }
    }
    FinalStmtEmitDisposition::Suppressed
}

fn legacy_last_unsuppressed_assign_var(body: &[runmat_hir::LegacyHirStmt]) -> Option<usize> {
    use runmat_hir::LegacyHirStmt as HirStmt;

    for stmt in body.iter().rev() {
        match stmt {
            HirStmt::Assign(var_id, _, suppressed, _) => {
                return if *suppressed { None } else { Some(var_id.0) };
            }
            HirStmt::ExprStmt(_, _, _)
            | HirStmt::MultiAssign(_, _, _, _)
            | HirStmt::AssignLValue(_, _, _, _) => return None,
            _ => continue,
        }
    }
    None
}

fn legacy_last_expr_emits_value(body: &[runmat_hir::LegacyHirStmt]) -> bool {
    use runmat_hir::LegacyHirStmt as HirStmt;

    for stmt in body.iter().rev() {
        match stmt {
            HirStmt::ExprStmt(expr, suppressed, _) => {
                if *suppressed {
                    return false;
                }
                return matches!(
                    expr_emit_disposition(expr),
                    FinalStmtEmitDisposition::Inline
                );
            }
            HirStmt::Assign(_, _, _, _)
            | HirStmt::MultiAssign(_, _, _, _)
            | HirStmt::AssignLValue(_, _, _, _) => return false,
            _ => continue,
        }
    }
    false
}

pub(crate) fn last_emit_var_index(bytecode: &runmat_vm::Bytecode) -> Option<usize> {
    for instr in bytecode.instructions.iter().rev() {
        if let runmat_vm::Instr::EmitVar { var_index, .. } = instr {
            return Some(*var_index);
        }
    }
    None
}

pub(crate) fn expr_emit_disposition(expr: &runmat_hir::LegacyHirExpr) -> FinalStmtEmitDisposition {
    use runmat_hir::LegacyHirExprKind as HirExprKind;
    if let HirExprKind::FuncCall(name, _) = &expr.kind {
        if runmat_builtins::suppresses_auto_output(name) {
            return FinalStmtEmitDisposition::Suppressed;
        }
    }
    FinalStmtEmitDisposition::Inline
}

pub(crate) fn workspace_entry(name: &str, value: &Value) -> WorkspaceEntry {
    let dtype = numeric_dtype_label(value).map(|label| label.to_string());
    let preview = preview_numeric_values(value, WORKSPACE_PREVIEW_LIMIT)
        .map(|(values, truncated)| WorkspacePreview { values, truncated });
    let residency = if matches!(value, Value::GpuTensor(_)) {
        WorkspaceResidency::Gpu
    } else {
        WorkspaceResidency::Cpu
    };
    WorkspaceEntry {
        name: name.to_string(),
        class_name: matlab_class_name(value),
        dtype,
        shape: value_shape(value).unwrap_or_default(),
        is_gpu: matches!(value, Value::GpuTensor(_)),
        size_bytes: approximate_size_bytes(value),
        preview,
        residency,
        preview_token: None,
    }
}
