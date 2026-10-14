use std::ops::Deref;

use crate::{LexicalEnvironment, Value};

use super::CallableInstanceId;

/// Semantic callable data. This is read-only through a constructed Closure;
/// changing its target or captures requires a new physical instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosureData {
    pub function_name: String,
    pub binding: Option<crate::BoundFunctionBinding>,
    /// Compiler/source-owned classification; display spelling is never authority.
    pub source_role: runmat_types::FunctionSourceRole,
    /// Compiler-registered origin retained with the callable across sessions.
    pub source_file: Option<Box<str>>,
    pub captures: Vec<Value>,
    /// A live ancestor link for nested functions or anonymous frames carrying
    /// a descendant's shared binding. Anonymous captures remain by value.
    pub lexical_environment: Option<LexicalEnvironment>,
}

#[derive(Debug, Clone)]
pub struct Closure {
    data: ClosureData,
    instance: CallableInstanceId,
}

impl Closure {
    pub fn new(
        function_name: String,
        binding: Option<crate::BoundFunctionBinding>,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
        captures: Vec<Value>,
        lexical_environment: Option<LexicalEnvironment>,
    ) -> Self {
        Self::from_data(ClosureData {
            function_name,
            binding,
            source_role,
            source_file: source_file.map(String::into_boxed_str),
            captures,
            lexical_environment,
        })
    }

    pub fn from_data(data: ClosureData) -> Self {
        Self {
            data,
            instance: CallableInstanceId::mint(),
        }
    }

    pub fn into_data(self) -> ClosureData {
        self.data
    }

    pub fn instance_id(&self) -> CallableInstanceId {
        self.instance
    }

    /// A resolved binding changes the target. Mint a new instance so an
    /// unbound clone cannot retain the same ID with different semantic data.
    pub fn bind_resolved(
        mut self,
        binding: crate::BoundFunctionBinding,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
    ) -> Self {
        assert!(
            self.data.binding.is_none(),
            "only an unresolved closure may be refined"
        );
        self.data.binding = Some(binding);
        if self.data.source_role == runmat_types::FunctionSourceRole::Unknown {
            self.data.source_role = source_role;
        }
        if self.data.source_file.is_none() {
            self.data.source_file = source_file.map(String::into_boxed_str);
        }
        self.instance = CallableInstanceId::mint();
        self
    }

    pub fn bound_function(&self) -> Option<usize> {
        self.data.binding.as_ref().map(|binding| binding.function)
    }
}

impl Deref for Closure {
    type Target = ClosureData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
