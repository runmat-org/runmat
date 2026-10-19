//! Retained function and call contracts shared by the frontend and executors.
//!
//! These are serialized in HIR, MIR, and native products. Keep their wire
//! representation independent of a parser or frontend crate.

use crate::{
    BindingId, ClassMethodRole, FunctionArgDefaultValue, FunctionArgSizeSpec, FunctionArgValidator,
    FunctionId,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum FunctionKind {
    Named,
    Anonymous,
    SyntheticEntrypoint,
    ClassMethod {
        is_static: bool,
        role: ClassMethodRole,
    },
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct FunctionModifiers {
    pub isolated: bool,
    pub is_async: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize, Default)]
pub struct FunctionAbi {
    pub fixed_inputs: Vec<BindingId>,
    pub varargin: Option<BindingId>,
    pub fixed_outputs: Vec<BindingId>,
    pub varargout: Option<BindingId>,
    pub implicit_nargin: Option<BindingId>,
    pub implicit_nargout: Option<BindingId>,
}

impl FunctionAbi {
    /// Fixed result bindings in declaration order, excluding the separately
    /// expanded `varargout` binding retained in the stored declaration list.
    pub fn fixed_output_bindings(&self) -> impl Iterator<Item = &BindingId> {
        self.fixed_outputs
            .iter()
            .filter(|binding| Some(**binding) != self.varargout)
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct FunctionArgumentValidation {
    pub binding: BindingId,
    pub size: Option<FunctionArgSizeSpec>,
    pub class_name: Option<String>,
    pub validators: Vec<FunctionArgValidator>,
    pub default_value: Option<FunctionArgDefaultValue>,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct CapturedBinding {
    pub binding: BindingId,
    pub from_function: FunctionId,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum CallSyntax {
    Plain,
    Method,
    DottedInvoke,
    Command,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum WorkspaceEffect {
    None,
    ReadsWorkspace,
    CreatesBinding,
    ClearsBinding,
    ClearsFunctionCache,
    MutatesGlobal,
    MutatesPersistent,
    LoadsExternalBindings,
    DynamicEval,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum EnvironmentEffect {
    PathMutation,
    WorkingDirectoryMutation,
    FunctionCacheInvalidation,
    DynamicLookupInvalidation,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct StringLiteral(pub String);

impl StringLiteral {
    /// Decode the source spelling into the text observed by execution.
    pub fn runtime_text(&self) -> String {
        if self.0.starts_with('"') && self.0.ends_with('"') && self.0.len() >= 2 {
            self.0[1..self.0.len() - 1].replace("\"\"", "\"")
        } else if self.0.starts_with('\'') && self.0.ends_with('\'') && self.0.len() >= 2 {
            self.0[1..self.0.len() - 1].replace("''", "'")
        } else {
            self.0.clone()
        }
    }

    pub fn is_character_row(&self) -> bool {
        self.0.starts_with('\'') && self.0.ends_with('\'') && self.0.len() >= 2
    }
}
