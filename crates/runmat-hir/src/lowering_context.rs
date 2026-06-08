use crate::FunctionId;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

pub struct LoweringContext<'a> {
    pub variables: &'a HashMap<String, usize>,
    pub bound_functions: &'a HashMap<String, FunctionId>,
    pub known_project_symbols: &'a HashSet<String>,
    pub private_function_owners: &'a HashMap<String, String>,
    pub private_function_aliases: &'a HashMap<String, HashMap<String, String>>,
    pub runmat_extensions_enabled: bool,
    pub top_level_await_enabled: bool,
}

impl<'a> LoweringContext<'a> {
    pub fn new(variables: &'a HashMap<String, usize>) -> Self {
        Self {
            variables,
            bound_functions: empty_bound_functions(),
            known_project_symbols: empty_project_symbols(),
            private_function_owners: empty_private_function_owners(),
            private_function_aliases: empty_private_function_aliases(),
            runmat_extensions_enabled: true,
            top_level_await_enabled: true,
        }
    }

    pub fn with_bound_functions(
        mut self,
        bound_functions: &'a HashMap<String, FunctionId>,
    ) -> Self {
        self.bound_functions = bound_functions;
        self
    }

    pub fn with_known_project_symbols(mut self, symbols: &'a HashSet<String>) -> Self {
        self.known_project_symbols = symbols;
        self
    }

    pub fn with_private_functions(
        mut self,
        owners: &'a HashMap<String, String>,
        aliases: &'a HashMap<String, HashMap<String, String>>,
    ) -> Self {
        self.private_function_owners = owners;
        self.private_function_aliases = aliases;
        self
    }

    pub fn with_runmat_extensions_enabled(mut self, enabled: bool) -> Self {
        self.runmat_extensions_enabled = enabled;
        self
    }

    pub fn with_top_level_await_enabled(mut self, enabled: bool) -> Self {
        self.top_level_await_enabled = enabled;
        self
    }

    pub fn empty() -> Self {
        static EMPTY_VARS: OnceLock<HashMap<String, usize>> = OnceLock::new();
        Self {
            variables: EMPTY_VARS.get_or_init(HashMap::new),
            bound_functions: empty_bound_functions(),
            known_project_symbols: empty_project_symbols(),
            private_function_owners: empty_private_function_owners(),
            private_function_aliases: empty_private_function_aliases(),
            runmat_extensions_enabled: true,
            top_level_await_enabled: true,
        }
    }
}

fn empty_bound_functions() -> &'static HashMap<String, FunctionId> {
    static EMPTY_SEMANTIC_FUNCS: OnceLock<HashMap<String, FunctionId>> = OnceLock::new();
    EMPTY_SEMANTIC_FUNCS.get_or_init(HashMap::new)
}

fn empty_project_symbols() -> &'static HashSet<String> {
    static EMPTY_PROJECT_SYMBOLS: OnceLock<HashSet<String>> = OnceLock::new();
    EMPTY_PROJECT_SYMBOLS.get_or_init(HashSet::new)
}

fn empty_private_function_owners() -> &'static HashMap<String, String> {
    static EMPTY_PRIVATE_FUNCTION_OWNERS: OnceLock<HashMap<String, String>> = OnceLock::new();
    EMPTY_PRIVATE_FUNCTION_OWNERS.get_or_init(HashMap::new)
}

fn empty_private_function_aliases() -> &'static HashMap<String, HashMap<String, String>> {
    static EMPTY_PRIVATE_FUNCTION_ALIASES: OnceLock<HashMap<String, HashMap<String, String>>> =
        OnceLock::new();
    EMPTY_PRIVATE_FUNCTION_ALIASES.get_or_init(HashMap::new)
}
