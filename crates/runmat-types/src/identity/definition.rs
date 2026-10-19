use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct DefPath {
    pub package: PackageName,
    pub module: QualifiedName,
    pub item: Vec<DefPathSegment>,
}

impl DefPath {
    pub fn display_name(&self) -> Option<String> {
        self.item.last().map(DefPathSegment::display_name)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub enum DefPathSegment {
    Function(SymbolName),
    Class(SymbolName),
    Method(SymbolName),
    ScriptSection { ordinal: u32, title: String },
}

impl DefPathSegment {
    pub fn display_name(&self) -> String {
        match self {
            Self::Function(name) | Self::Class(name) | Self::Method(name) => name.0.clone(),
            Self::ScriptSection { ordinal, title } if title.is_empty() => {
                format!("section-{ordinal}")
            }
            Self::ScriptSection { ordinal, title } => format!("section-{ordinal}:{title}"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct QualifiedName(pub Vec<SymbolName>);

impl QualifiedName {
    pub fn display_name(&self) -> Option<String> {
        (!self.0.is_empty() && self.0.iter().all(|part| !part.0.is_empty())).then(|| {
            self.0
                .iter()
                .map(|part| part.0.as_str())
                .collect::<Vec<_>>()
                .join(".")
        })
    }
}

macro_rules! string_identity {
    ($name:ident) => {
        #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Serialize, Deserialize)]
        pub struct $name(pub String);
    };
}

string_identity!(SymbolName);
string_identity!(BindingName);
string_identity!(FunctionName);
string_identity!(EntrypointName);
string_identity!(MemberName);
string_identity!(MethodName);
string_identity!(PackageName);
string_identity!(BuiltinId);
string_identity!(MethodId);

impl MemberName {
    pub fn display_name(&self) -> &str {
        &self.0
    }
}

impl From<String> for MemberName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl From<&str> for MemberName {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl std::borrow::Borrow<str> for MemberName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MemberName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl MethodName {
    pub fn display_name(&self) -> &str {
        &self.0
    }

    pub fn property_getter(property: &MemberName) -> Self {
        Self(format!("get.{}", property.0))
    }

    pub fn property_setter(property: &MemberName) -> Self {
        Self(format!("set.{}", property.0))
    }

    pub fn is_property_getter(&self) -> bool {
        self.0
            .strip_prefix("get.")
            .is_some_and(|name| !name.is_empty())
    }

    pub fn is_property_setter(&self) -> bool {
        self.0
            .strip_prefix("set.")
            .is_some_and(|name| !name.is_empty())
    }
}

impl From<String> for MethodName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl From<&str> for MethodName {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl std::borrow::Borrow<str> for MethodName {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MethodName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A method identity whose spelling is fixed at compile time.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct StaticMethodName(&'static str);

impl StaticMethodName {
    pub const fn new(name: &'static str) -> Self {
        assert_static_member_name(name);
        Self(name)
    }

    pub const fn display_name(self) -> &'static str {
        self.0
    }

    pub fn owned(self) -> MethodName {
        MethodName(self.0.to_owned())
    }

    pub fn is(self, name: &MethodName) -> bool {
        self.0 == name.0
    }

    /// Compare with text that has not yet crossed an identity boundary.
    pub fn matches_text(self, name: &str) -> bool {
        self.0 == name
    }
}

impl std::fmt::Display for StaticMethodName {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl From<StaticMethodName> for MethodName {
    fn from(name: StaticMethodName) -> Self {
        name.owned()
    }
}

const fn assert_static_member_name(name: &str) {
    let bytes = name.as_bytes();
    assert!(!bytes.is_empty(), "member identity must not be empty");
    let mut index = 0;
    while index < bytes.len() {
        assert!(
            bytes[index] != b'.',
            "member identity must not be qualified"
        );
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{MethodName, StaticMethodName};

    #[test]
    fn static_method_identity_preserves_the_semantic_name() {
        const METHOD: StaticMethodName = StaticMethodName::new("subsref");

        assert_eq!(METHOD.display_name(), "subsref");
        assert!(METHOD.is(&MethodName::from("subsref")));
        assert!(METHOD.matches_text("subsref"));
        assert_eq!(METHOD.owned(), MethodName::from("subsref"));
    }

    #[test]
    #[should_panic(expected = "member identity must not be qualified")]
    fn static_method_identity_rejects_qualified_names() {
        let _ = StaticMethodName::new("Class.method");
    }
}
