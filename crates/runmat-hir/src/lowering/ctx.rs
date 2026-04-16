use crate::inference::function_outputs::infer_function_output_types;
use crate::inference::function_vars::infer_function_variable_types;
use crate::inference::globals::infer_global_variable_types;
use crate::validation::classdefs::validate_classdefs;
use crate::{HirProgram, HirStmt, LoweringContext, LoweringResult, SemanticError, Type, VarId};
use runmat_parser::Program as AstProgram;
use std::collections::HashMap;

pub(crate) struct Scope {
    pub(crate) parent: Option<usize>,
    pub(crate) bindings: HashMap<String, VarId>,
}

pub(crate) struct Ctx {
    pub(crate) scopes: Vec<Scope>,
    pub(crate) var_types: Vec<Type>,
    pub(crate) next_var: usize,
    pub(crate) functions: HashMap<String, HirStmt>,
    pub(crate) var_names: Vec<Option<String>>,
    pub(crate) allow_unqualified_statics: bool,
}

pub fn lower(
    prog: &AstProgram,
    context: &LoweringContext<'_>,
) -> Result<LoweringResult, SemanticError> {
    let mut ctx = Ctx::new();

    for (name, var_id) in context.variables {
        ctx.scopes[0].bindings.insert(name.clone(), VarId(*var_id));
        while ctx.var_types.len() <= *var_id {
            ctx.var_types.push(Type::Unknown);
        }
        while ctx.var_names.len() <= *var_id {
            ctx.var_names.push(None);
        }
        ctx.var_names[*var_id] = Some(name.clone());
        if *var_id >= ctx.next_var {
            ctx.next_var = var_id + 1;
        }
    }

    for (name, func_stmt) in context.functions {
        ctx.functions.insert(name.clone(), func_stmt.clone());
    }

    let body = ctx.lower_stmts(&prog.body)?;
    let var_types = ctx.var_types.clone();
    let hir = HirProgram { body, var_types };
    let _ = infer_function_output_types(&hir);
    validate_classdefs(&hir)?;

    let mut variables: HashMap<String, usize> = HashMap::new();
    for (name, var_id) in &ctx.scopes[0].bindings {
        variables.insert(name.clone(), var_id.0);
    }
    let mut var_names = HashMap::new();
    for (idx, name_opt) in ctx.var_names.iter().enumerate() {
        if let Some(name) = name_opt {
            var_names.insert(VarId(idx), name.clone());
        }
    }

    let inferred_function_returns = infer_function_output_types(&hir);
    let inferred_function_envs = infer_function_variable_types(&hir);
    let inferred_globals = infer_global_variable_types(&hir, &inferred_function_returns);

    Ok(LoweringResult {
        hir,
        variables,
        functions: ctx.functions,
        var_types: ctx.var_types,
        var_names,
        inferred_globals,
        inferred_function_envs,
        inferred_function_returns,
    })
}

impl Ctx {
    pub(crate) fn new() -> Self {
        Self {
            scopes: vec![Scope {
                parent: None,
                bindings: HashMap::new(),
            }],
            var_types: Vec::new(),
            next_var: 0,
            functions: HashMap::new(),
            var_names: Vec::new(),
            allow_unqualified_statics: false,
        }
    }

    pub(crate) fn push_scope(&mut self) -> usize {
        let parent = Some(self.scopes.len() - 1);
        self.scopes.push(Scope {
            parent,
            bindings: HashMap::new(),
        });
        self.scopes.len() - 1
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn define(&mut self, name: String) -> VarId {
        let id = VarId(self.next_var);
        self.next_var += 1;
        let current = self.scopes.len() - 1;
        self.scopes[current].bindings.insert(name.clone(), id);
        self.var_types.push(Type::Unknown);
        self.var_names.push(Some(name));
        id
    }

    pub(crate) fn lookup_current_scope(&self, name: &str) -> Option<VarId> {
        let current = self.scopes.len() - 1;
        self.scopes[current].bindings.get(name).copied()
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<VarId> {
        let mut scope_idx = Some(self.scopes.len() - 1);
        while let Some(idx) = scope_idx {
            if let Some(id) = self.scopes[idx].bindings.get(name) {
                return Some(*id);
            }
            scope_idx = self.scopes[idx].parent;
        }
        None
    }

    pub(crate) fn is_constant(&self, name: &str) -> bool {
        runmat_builtins::constants().iter().any(|c| c.name == name)
    }

    pub(crate) fn is_builtin_function(&self, name: &str) -> bool {
        runmat_builtins::builtin_functions()
            .iter()
            .any(|b| b.name == name)
    }

    pub(crate) fn is_user_defined_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    pub(crate) fn is_function(&self, name: &str) -> bool {
        self.is_user_defined_function(name) || self.is_builtin_function(name)
    }

    pub(crate) fn is_static_method_class(&self, name: &str) -> bool {
        matches!(
            name,
            "gpuArray"
                | "logical"
                | "double"
                | "single"
                | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "char"
                | "string"
                | "cell"
                | "struct"
        )
    }
}
