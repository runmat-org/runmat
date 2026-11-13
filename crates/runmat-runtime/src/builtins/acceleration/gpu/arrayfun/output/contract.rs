use runmat_value::NumericDType;

use super::super::{callback::Callable, input::ArrayInput};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in crate::builtins::acceleration::gpu::arrayfun) enum OutputContract {
    #[default]
    Dynamic,
    Logical,
    Numeric(NumericDType),
    Complex(NumericDType),
    Character,
}

impl OutputContract {
    pub(in crate::builtins::acceleration::gpu::arrayfun) fn infer(
        callable: &Callable,
        inputs: &[ArrayInput],
    ) -> Self {
        let Some(identity) = callable.builtin_identity() else {
            return Self::Dynamic;
        };
        let inference = crate::call::catalog::infer_builtin_identity_callback(
            identity,
            inputs.iter().map(ArrayInput::scalar_fact).collect(),
            1,
        );
        if inference
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == runmat_types::InferenceSeverity::Error)
        {
            return Self::Dynamic;
        }
        match inference.outputs.first().map(|output| &output.kind) {
            Some(runmat_types::ValueKindFact::Logical) => Self::Logical,
            Some(runmat_types::ValueKindFact::Character) => Self::Character,
            Some(runmat_types::ValueKindFact::Numeric(numeric)) => match numeric.domain {
                runmat_types::NumericDomain::Real => Self::Numeric(numeric.class.into()),
                runmat_types::NumericDomain::Complex => Self::Complex(numeric.class.into()),
            },
            _ => Self::Dynamic,
        }
    }
}
