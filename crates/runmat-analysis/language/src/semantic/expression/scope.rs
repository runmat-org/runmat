use crate::FeaType;
use runmat_design_syntax::FeaReference;
use runmat_types::{OperationParameter, OperationTypePattern};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpressionScope {
    references: BTreeMap<Vec<String>, FeaType>,
    functions: BTreeMap<Vec<String>, FunctionSignature>,
    declarations: BTreeMap<Vec<String>, DeclarationSignature>,
}

impl ExpressionScope {
    pub fn new() -> Self {
        Self {
            references: BTreeMap::new(),
            functions: BTreeMap::new(),
            declarations: BTreeMap::new(),
        }
    }

    pub fn insert_reference(&mut self, path: impl IntoIterator<Item = String>, value: FeaType) {
        self.references.insert(path.into_iter().collect(), value);
    }

    pub fn insert_function(
        &mut self,
        path: impl IntoIterator<Item = String>,
        signature: FunctionSignature,
    ) {
        self.functions.insert(path.into_iter().collect(), signature);
    }

    pub fn insert_declaration(
        &mut self,
        path: impl IntoIterator<Item = String>,
        signature: DeclarationSignature,
    ) {
        self.declarations
            .insert(path.into_iter().collect(), signature);
    }

    pub(super) fn reference(&self, reference: &FeaReference) -> Option<&FeaType> {
        self.references.get(&reference_path(reference))
    }

    pub(super) fn function(&self, reference: &FeaReference) -> Option<&FunctionSignature> {
        self.functions.get(&reference_path(reference))
    }

    pub(super) fn declaration(&self, reference: &FeaReference) -> Option<&DeclarationSignature> {
        self.declarations.get(&reference_path(reference))
    }
}

impl Default for ExpressionScope {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    pub inputs: Vec<OperationParameter>,
    pub outputs: Vec<OperationTypePattern>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclarationSignature {
    pub result: FeaType,
    pub inputs: BTreeMap<String, DeclarationInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclarationInput {
    pub value_type: FeaType,
    pub required: bool,
}

fn reference_path(reference: &FeaReference) -> Vec<String> {
    reference
        .segments
        .iter()
        .map(|segment| segment.text.clone())
        .collect()
}
