//! Source embedding and live argument owners share one deoptimization driver.
use runmat_value::Value;

pub(in crate::generic_native) enum Inputs {
    Source {
        captures: Vec<runmat_runtime::call::lexical::LexicalCapture>,
        arguments: Vec<Value>,
    },
    Retained(runmat_runtime::call::RootedFunctionArguments),
}

impl Inputs {
    /// A retained input already owns its registrations. Only source ingress
    /// needs the ordinary embedding root acquisition performed by this driver.
    pub(super) fn source_values(&self) -> impl Iterator<Item = &Value> {
        let (arguments, captures): (&[Value], &[runmat_runtime::call::lexical::LexicalCapture]) =
            match self {
                Self::Source {
                    captures,
                    arguments,
                } => (arguments, captures),
                Self::Retained(_) => (&[], &[]),
            };
        arguments
            .iter()
            .chain(captures.iter().map(|capture| &capture.value))
    }
    pub(super) fn has_captures(&self) -> bool {
        match self {
            Self::Source { captures, .. } => !captures.is_empty(),
            Self::Retained(arguments) => arguments.lexical_capture_count() != 0,
        }
    }
}
