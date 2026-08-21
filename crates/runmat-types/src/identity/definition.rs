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
        #[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
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
