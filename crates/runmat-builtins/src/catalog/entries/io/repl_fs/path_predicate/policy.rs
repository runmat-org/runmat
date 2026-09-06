#[derive(Clone, Copy)]
pub(super) struct PathPredicateInferencePolicy {
    pub arity_code: &'static str,
    pub arity_message: &'static str,
    pub path_code: &'static str,
    pub path_message: &'static str,
}
