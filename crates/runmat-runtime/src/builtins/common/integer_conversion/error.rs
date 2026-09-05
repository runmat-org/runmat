use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UnsupportedValueKind {
    Text,
    Symbolic,
    Cell,
    Struct,
    Object(String),
    Listener,
    FunctionHandle,
    ClassReference,
    RuntimeHandle,
    Foreign,
    OutputList,
    Complex,
}

impl fmt::Display for UnsupportedValueKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => formatter.write_str("string"),
            Self::Symbolic => formatter.write_str("sym"),
            Self::Cell => formatter.write_str("cell"),
            Self::Struct => formatter.write_str("struct"),
            Self::Object(class) => formatter.write_str(class),
            Self::Listener => formatter.write_str("event.listener"),
            Self::FunctionHandle => formatter.write_str("function_handle"),
            Self::ClassReference => formatter.write_str("meta.class"),
            Self::RuntimeHandle => formatter.write_str("runtime handle"),
            Self::Foreign => formatter.write_str("foreign"),
            Self::OutputList => formatter.write_str("OutputList"),
            Self::Complex => formatter.write_str("complex"),
        }
    }
}

#[derive(Debug)]
pub(crate) enum CastError {
    Unsupported(UnsupportedValueKind),
    Internal(String),
}
