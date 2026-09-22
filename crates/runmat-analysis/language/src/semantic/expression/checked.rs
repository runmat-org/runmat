use super::CheckedExpression;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCadParameterValue {
    pub geometry_module_path: String,
    pub geometry_id: String,
    pub parameter_id: String,
    pub source_entity_id: u64,
    pub persistent_id: Option<String>,
    pub value_si_bits: u64,
    pub source_value_bits: u64,
    pub source_unit: String,
}

impl ResolvedCadParameterValue {
    pub fn value_si(&self) -> f64 {
        f64::from_bits(self.value_si_bits)
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
