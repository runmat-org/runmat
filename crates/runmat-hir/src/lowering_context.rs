use crate::FunctionId;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionOutputArity {
    pub fixed_outputs: usize,
    pub has_varargout: bool,
}

impl FunctionOutputArity {
    pub fn new(fixed_outputs: usize, has_varargout: bool) -> Self {
        Self {
            fixed_outputs,
            has_varargout,
        }
    }

    pub fn from_declared_outputs(outputs: &[String]) -> Self {
        Self {
            fixed_outputs: outputs
                .iter()
                .filter(|output| output.as_str() != "varargout")
                .count(),
            has_varargout: outputs.iter().any(|output| output.as_str() == "varargout"),
        }
    }

    pub fn is_declared_zero_output(self) -> bool {
        self.fixed_outputs == 0 && !self.has_varargout
    }
}

pub struct LoweringContext<'a> {
    pub variables: &'a HashMap<String, usize>,
    pub bound_functions: &'a HashMap<String, FunctionId>,
    pub function_output_arities: &'a HashMap<FunctionId, FunctionOutputArity>,
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
            function_output_arities: empty_function_output_arities(),
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

    pub fn with_function_output_arities(
        mut self,
        arities: &'a HashMap<FunctionId, FunctionOutputArity>,
    ) -> Self {
        self.function_output_arities = arities;
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
            function_output_arities: empty_function_output_arities(),
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

fn empty_function_output_arities() -> &'static HashMap<FunctionId, FunctionOutputArity> {
    static EMPTY_FUNCTION_OUTPUT_ARITIES: OnceLock<HashMap<FunctionId, FunctionOutputArity>> =
        OnceLock::new();
    EMPTY_FUNCTION_OUTPUT_ARITIES.get_or_init(HashMap::new)
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
