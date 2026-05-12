use std::collections::HashMap;

use runmat_builtins::Value;
use runmat_hir::{HirAssembly, HirExprKind, HirPlace, HirStmtKind};

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
pub(crate) struct DisplayContext {
    pub first_assign_var: Option<usize>,
    pub single_assign_var: Option<usize>,
    pub single_stmt_non_assign: bool,
    pub final_stmt_emit: FinalStmtEmitDisposition,
    pub last_assign_var: Option<usize>,
    pub last_expr_emits: bool,
}

pub(crate) struct ExecutionDisplayContext {
    pub context: DisplayContext,
    pub display_var_ids: Vec<usize>,
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

pub(crate) fn execution_display_context(
    assembly: &HirAssembly,
    layout: Option<&runmat_vm::VmAssemblyLayout>,
) -> ExecutionDisplayContext {
    semantic_display_context(assembly, layout).unwrap_or_else(|| ExecutionDisplayContext {
        display_var_ids: Vec::new(),
        context: DisplayContext {
            first_assign_var: None,
            single_assign_var: None,
            single_stmt_non_assign: false,
            final_stmt_emit: FinalStmtEmitDisposition::Suppressed,
            last_assign_var: None,
            last_expr_emits: false,
        },
    })
}

fn semantic_display_context(
    assembly: &HirAssembly,
    layout: Option<&runmat_vm::VmAssemblyLayout>,
) -> Option<ExecutionDisplayContext> {
    let entrypoint = assembly.entrypoints.first()?;
    let function = assembly
        .functions
        .iter()
        .find(|f| f.id == entrypoint.target)?;
    let function_layout = layout?.functions.get(&function.id)?;
    let slot_for_place = |place: &HirPlace| match place {
        HirPlace::Binding(binding) => function_layout
            .binding_slots
            .get(binding)
            .map(|slot| slot.0),
        _ => None,
    };
    let slot_for_binding_expr = |expr: &runmat_hir::HirExpr| match &expr.kind {
        HirExprKind::Binding(binding) => function_layout
            .binding_slots
            .get(binding)
            .map(|slot| slot.0),
        _ => None,
    };

    let statements = &function.body.statements;
    let (single_assign_var, single_stmt_non_assign) = if statements.len() == 1 {
        match &statements[0].kind {
            HirStmtKind::Assign(place, _, _) => (slot_for_place(place), false),
            _ => (None, true),
        }
    } else {
        (None, false)
    };

    let first_assign_var = statements.first().and_then(|stmt| match &stmt.kind {
        HirStmtKind::Assign(place, _, _) => slot_for_place(place),
        _ => None,
    });

    let mut final_stmt_emit = FinalStmtEmitDisposition::Suppressed;
    for stmt in statements.iter().rev() {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, suppressed) => {
                final_stmt_emit = semantic_expr_emit_disposition(expr, *suppressed);
                break;
            }
            HirStmtKind::Assign(_, _, _) | HirStmtKind::MultiAssign(_, _, _) => break,
            _ => continue,
        }
    }

    let mut last_assign_var = None;
    for stmt in statements.iter().rev() {
        match &stmt.kind {
            HirStmtKind::Assign(place, _, suppressed) => {
                last_assign_var = if *suppressed {
                    None
                } else {
                    slot_for_place(place)
                };
                break;
            }
            HirStmtKind::ExprStmt(_, _) | HirStmtKind::MultiAssign(_, _, _) => break,
            _ => continue,
        }
    }

    let mut last_expr_emits = false;
    for stmt in statements.iter().rev() {
        match &stmt.kind {
            HirStmtKind::ExprStmt(expr, suppressed) => {
                last_expr_emits = matches!(
                    semantic_expr_emit_disposition(expr, *suppressed),
                    FinalStmtEmitDisposition::Inline
                );
                break;
            }
            HirStmtKind::Assign(_, _, _) | HirStmtKind::MultiAssign(_, _, _) => break,
            _ => continue,
        }
    }

    let mut display_var_ids = Vec::new();
    for stmt in statements {
        match &stmt.kind {
            HirStmtKind::Assign(place, _, suppressed) if !*suppressed => {
                display_var_ids.extend(slot_for_place(place));
            }
            HirStmtKind::ExprStmt(expr, suppressed) if !*suppressed => {
                display_var_ids.extend(slot_for_binding_expr(expr));
            }
            HirStmtKind::MultiAssign(targets, _, suppressed) if !*suppressed => {
                display_var_ids.extend(targets.targets.iter().filter_map(|target| match target {
                    runmat_hir::OutputTarget::Place(place) => slot_for_place(place),
                    _ => None,
                }));
            }
            _ => {}
        }
    }

    Some(ExecutionDisplayContext {
        context: DisplayContext {
            first_assign_var,
            single_assign_var,
            single_stmt_non_assign,
            final_stmt_emit,
            last_assign_var,
            last_expr_emits,
        },
        display_var_ids,
    })
}

fn semantic_expr_emit_disposition(
    expr: &runmat_hir::HirExpr,
    suppressed: bool,
) -> FinalStmtEmitDisposition {
    if suppressed {
        return FinalStmtEmitDisposition::Suppressed;
    }
    if let HirExprKind::Call(call) = &expr.kind {
        if let runmat_hir::HirCallableRef::Builtin(builtin) = &call.callee {
            if runmat_builtins::suppresses_auto_output(builtin.0.as_str()) {
                return FinalStmtEmitDisposition::Suppressed;
            }
        }
    }
    FinalStmtEmitDisposition::Inline
}

pub(crate) fn last_emit_var_index(bytecode: &runmat_vm::Bytecode) -> Option<usize> {
    for instr in bytecode.instructions.iter().rev() {
        if let runmat_vm::Instr::EmitVar { var_index, .. } = instr {
            return Some(*var_index);
        }
    }
    None
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
