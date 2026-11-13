/// Builtins whose language semantics require an exact workspace execution
/// target rather than an ordinary registered implementation body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicWorkspaceBuiltin {
    Eval,
    Evalc,
    Evalin,
    Assignin,
    Run,
}

impl DynamicWorkspaceBuiltin {
    /// Classify one already-canonicalized builtin catalog name.
    pub fn for_canonical_name(name: &str) -> Option<Self> {
        match name {
            runmat_types::EVAL_BUILTIN_NAME => Some(Self::Eval),
            runmat_types::EVALC_BUILTIN_NAME => Some(Self::Evalc),
            runmat_types::EVALIN_BUILTIN_NAME => Some(Self::Evalin),
            runmat_types::ASSIGNIN_BUILTIN_NAME => Some(Self::Assignin),
            runmat_types::RUN_BUILTIN_NAME => Some(Self::Run),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Eval => runmat_types::EVAL_BUILTIN_NAME,
            Self::Evalc => runmat_types::EVALC_BUILTIN_NAME,
            Self::Evalin => runmat_types::EVALIN_BUILTIN_NAME,
            Self::Assignin => runmat_types::ASSIGNIN_BUILTIN_NAME,
            Self::Run => runmat_types::RUN_BUILTIN_NAME,
        }
    }
}
