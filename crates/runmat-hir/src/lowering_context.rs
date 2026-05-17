use crate::{CompatibilityHirStmt as HirStmt, CompatibilityMode, FunctionId};
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct LoweringContext<'a> {
    pub variables: &'a HashMap<String, usize>,
    pub functions: &'a HashMap<String, HirStmt>,
    pub semantic_functions: &'a HashMap<String, FunctionId>,
    pub compatibility_mode: Option<CompatibilityMode>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(
        variables: &'a HashMap<String, usize>,
        functions: &'a HashMap<String, HirStmt>,
    ) -> Self {
        Self {
            variables,
            functions,
            semantic_functions: empty_semantic_functions(),
            compatibility_mode: None,
        }
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
        static EMPTY_FUNCS: OnceLock<HashMap<String, HirStmt>> = OnceLock::new();
        Self {
            variables: EMPTY_VARS.get_or_init(HashMap::new),
            functions: EMPTY_FUNCS.get_or_init(HashMap::new),
            semantic_functions: empty_semantic_functions(),
            compatibility_mode: None,
        }
    }
}

fn empty_semantic_functions() -> &'static HashMap<String, FunctionId> {
    static EMPTY_SEMANTIC_FUNCS: OnceLock<HashMap<String, FunctionId>> = OnceLock::new();
    EMPTY_SEMANTIC_FUNCS.get_or_init(HashMap::new)
}
