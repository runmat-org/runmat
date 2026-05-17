use crate::compatibility::HirStmt as CompatibilityHirStmt;
use crate::{CompatibilityMode, FunctionId};
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct LoweringContext<'a> {
    pub variables: &'a HashMap<String, usize>,
    compatibility_functions: &'a HashMap<String, CompatibilityHirStmt>,
    pub semantic_functions: &'a HashMap<String, FunctionId>,
    pub compatibility_mode: Option<CompatibilityMode>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(variables: &'a HashMap<String, usize>) -> Self {
        Self {
            variables,
            compatibility_functions: empty_compatibility_functions(),
            semantic_functions: empty_semantic_functions(),
            compatibility_mode: None,
        }
    }

    pub fn with_compatibility_functions(
        mut self,
        functions: &'a HashMap<String, CompatibilityHirStmt>,
    ) -> Self {
        self.compatibility_functions = functions;
        self
    }

    pub fn with_semantic_functions(
        mut self,
        semantic_functions: &'a HashMap<String, FunctionId>,
    ) -> Self {
        self.semantic_functions = semantic_functions;
        self
    }

    pub fn with_compatibility_mode(mut self, compatibility_mode: CompatibilityMode) -> Self {
        self.compatibility_mode = Some(compatibility_mode);
        self
    }

    pub fn empty() -> Self {
        static EMPTY_VARS: OnceLock<HashMap<String, usize>> = OnceLock::new();
        Self {
            variables: EMPTY_VARS.get_or_init(HashMap::new),
            compatibility_functions: empty_compatibility_functions(),
            semantic_functions: empty_semantic_functions(),
            compatibility_mode: None,
        }
    }

    pub(crate) fn compatibility_functions(&self) -> &'a HashMap<String, CompatibilityHirStmt> {
        self.compatibility_functions
    }
}

fn empty_compatibility_functions() -> &'static HashMap<String, CompatibilityHirStmt> {
    static EMPTY_COMPAT_FUNCS: OnceLock<HashMap<String, CompatibilityHirStmt>> = OnceLock::new();
    EMPTY_COMPAT_FUNCS.get_or_init(HashMap::new)
}

fn empty_semantic_functions() -> &'static HashMap<String, FunctionId> {
    static EMPTY_SEMANTIC_FUNCS: OnceLock<HashMap<String, FunctionId>> = OnceLock::new();
    EMPTY_SEMANTIC_FUNCS.get_or_init(HashMap::new)
}
