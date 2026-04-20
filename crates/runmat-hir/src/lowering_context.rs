use crate::HirStmt;
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct LoweringContext<'a> {
    pub variables: &'a HashMap<String, usize>,
    pub functions: &'a HashMap<String, HirStmt>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(
        variables: &'a HashMap<String, usize>,
        functions: &'a HashMap<String, HirStmt>,
    ) -> Self {
        Self {
            variables,
            functions,
        }
    }

    pub fn empty() -> Self {
        static EMPTY_VARS: OnceLock<HashMap<String, usize>> = OnceLock::new();
        static EMPTY_FUNCS: OnceLock<HashMap<String, HirStmt>> = OnceLock::new();
        Self {
            variables: EMPTY_VARS.get_or_init(HashMap::new),
            functions: EMPTY_FUNCS.get_or_init(HashMap::new),
        }
    }
}
