/// Implemented public exception properties. Name decoding belongs at the
/// member-access boundary; inference and execution dispatch on this identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExceptionProperty {
    Identifier,
    Message,
    Stack,
}

impl ExceptionProperty {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "identifier" => Some(Self::Identifier),
            "message" => Some(Self::Message),
            "stack" => Some(Self::Stack),
            _ => None,
        }
    }
}
