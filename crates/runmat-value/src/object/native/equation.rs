//! Equation native data has one storage generation, no property mirror.
use super::*;
impl ObjectInstance {
    /// Original-context Runtime producers validate the complete equality DAG.
    /// Finite native promotion is refused by the common native owner until
    /// complete model-DAG storage admission is implemented.
    pub fn with_equation_problem_payload(self, data: EquationProblemData) -> Self {
        self.with_native(NativeObjectPayload::EquationProblem(
            SharedNativeState::new(data),
        ))
    }
    pub fn equation_problem_payload(&self) -> Option<&EquationProblemData> {
        match self.generation().native_payload.as_ref() {
            Some(NativeObjectPayload::EquationProblem(value)) => Some(value.get()),
            _ => None,
        }
    }
}
