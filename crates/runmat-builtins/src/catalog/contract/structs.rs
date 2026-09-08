use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StructInferenceRule {
    Fieldnames,
    Isfield,
    Rmfield,
    Structfun,
}
