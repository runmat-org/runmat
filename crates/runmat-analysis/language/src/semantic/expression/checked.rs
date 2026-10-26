use super::CheckedExpression;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCadParameterValue {
    pub geometry_module_path: String,
    pub geometry_id: String,
    pub parameter_id: String,
    pub source_entity_id: u64,
    pub persistent_id: Option<String>,
    pub measure: runmat_types::physical::SourceQuantity,
}

impl ResolvedCadParameterValue {
    pub fn value_si(&self) -> f64 {
        self.measure.canonical_value()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckedCallArgument {
    Positional(CheckedExpression),
    Named {
        name: String,
        name_span: runmat_types::Span,
        value: CheckedExpression,
    },
}
