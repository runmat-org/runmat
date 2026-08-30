use super::{QualifiedName, SymbolName};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Canonical, portable identity of a runtime class.
///
/// Class spellings enter the system through parsers, registries, or foreign
/// interfaces. Semantic code carries this type and compares identities; it
/// does not reinterpret source strings at each use site.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClassIdentity(String);

/// A canonical class identity known at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StaticClassIdentity(&'static str);

/// A canonical namespace used to classify related runtime classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StaticClassNamespace(&'static str);

impl ClassIdentity {
    pub fn new(name: impl Into<String>) -> Result<Self, InvalidClassIdentity> {
        let name = name.into();
        validate(&name)?;
        Ok(Self(name))
    }

    pub fn from_qualified_name(name: &QualifiedName) -> Result<Self, InvalidClassIdentity> {
        Self::new(
            name.display_name()
                .ok_or(InvalidClassIdentity::EmptySegment)?,
        )
    }

    pub fn qualified_name(&self) -> QualifiedName {
        QualifiedName(
            self.0
                .split('.')
                .map(|part| SymbolName(part.to_owned()))
                .collect(),
        )
    }

    /// Source/display spelling for diagnostics and external interfaces.
    pub fn display_name(&self) -> &str {
        &self.0
    }

    pub fn is(&self, expected: StaticClassIdentity) -> bool {
        self == &expected
    }

    pub fn is_in_namespace(&self, namespace: StaticClassNamespace) -> bool {
        self.0
            .strip_prefix(namespace.0)
            .is_some_and(|suffix| suffix.starts_with('.'))
    }
}

impl StaticClassIdentity {
    pub const fn new(name: &'static str) -> Self {
        assert_valid_static(name);
        Self(name)
    }

    pub const fn display_name(self) -> &'static str {
        self.0
    }

    pub fn owned(self) -> ClassIdentity {
        ClassIdentity(self.0.to_owned())
    }
}

impl StaticClassNamespace {
    pub const fn new(name: &'static str) -> Self {
        assert_valid_static(name);
        Self(name)
    }

    pub const fn display_name(self) -> &'static str {
        self.0
    }
}

impl PartialEq<StaticClassIdentity> for ClassIdentity {
    fn eq(&self, other: &StaticClassIdentity) -> bool {
        self.0 == other.0
    }
}

impl PartialEq<ClassIdentity> for StaticClassIdentity {
    fn eq(&self, other: &ClassIdentity) -> bool {
        other == self
    }
}

impl fmt::Display for ClassIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ClassIdentity {
    fn as_ref(&self) -> &str {
        self.display_name()
    }
}

impl fmt::Display for StaticClassIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl AsRef<str> for StaticClassIdentity {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl From<String> for ClassIdentity {
    fn from(value: String) -> Self {
        Self::new(value).expect("class identity must be canonical")
    }
}

impl From<&str> for ClassIdentity {
    fn from(value: &str) -> Self {
        Self::new(value).expect("class identity must be canonical")
    }
}

impl From<StaticClassIdentity> for ClassIdentity {
    fn from(value: StaticClassIdentity) -> Self {
        value.owned()
    }
}

impl From<StaticClassIdentity> for String {
    fn from(value: StaticClassIdentity) -> Self {
        value.display_name().to_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidClassIdentity {
    Empty,
    EmptySegment,
}

impl fmt::Display for InvalidClassIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("class identity must not be empty"),
            Self::EmptySegment => f.write_str("class identity contains an empty name segment"),
        }
    }
}

impl std::error::Error for InvalidClassIdentity {}

fn validate(name: &str) -> Result<(), InvalidClassIdentity> {
    if name.is_empty() {
        return Err(InvalidClassIdentity::Empty);
    }
    if name.split('.').any(str::is_empty) {
        return Err(InvalidClassIdentity::EmptySegment);
    }
    Ok(())
}

const fn assert_valid_static(name: &str) {
    let bytes = name.as_bytes();
    assert!(!bytes.is_empty(), "class identity must not be empty");
    assert!(bytes[0] != b'.', "class identity must not start with a dot");
    assert!(
        bytes[bytes.len() - 1] != b'.',
        "class identity must not end with a dot"
    );
    let mut index = 1;
    while index < bytes.len() {
        assert!(
            !(bytes[index - 1] == b'.' && bytes[index] == b'.'),
            "class identity must not contain an empty segment"
        );
        index += 1;
    }
}

pub mod standard {
    use super::{StaticClassIdentity, StaticClassNamespace};

    pub const LANGUAGE_CORRECTION_NAMESPACE: StaticClassNamespace =
        StaticClassNamespace::new("matlab.lang.correction");

    pub const DOUBLE: StaticClassIdentity = StaticClassIdentity::new("double");
    pub const SINGLE: StaticClassIdentity = StaticClassIdentity::new("single");
    pub const LOGICAL: StaticClassIdentity = StaticClassIdentity::new("logical");
    pub const CHAR: StaticClassIdentity = StaticClassIdentity::new("char");
    pub const STRING: StaticClassIdentity = StaticClassIdentity::new("string");
    pub const CELL: StaticClassIdentity = StaticClassIdentity::new("cell");
    pub const STRUCT: StaticClassIdentity = StaticClassIdentity::new("struct");
    pub const FUNCTION_HANDLE: StaticClassIdentity = StaticClassIdentity::new("function_handle");
    pub const META_CLASS: StaticClassIdentity = StaticClassIdentity::new("meta.class");
    pub const SPARSE: StaticClassIdentity = StaticClassIdentity::new("sparse");
    pub const SYMBOLIC: StaticClassIdentity = StaticClassIdentity::new("sym");
    pub const INT8: StaticClassIdentity = StaticClassIdentity::new("int8");
    pub const UINT8: StaticClassIdentity = StaticClassIdentity::new("uint8");
    pub const INT16: StaticClassIdentity = StaticClassIdentity::new("int16");
    pub const UINT16: StaticClassIdentity = StaticClassIdentity::new("uint16");
    pub const INT32: StaticClassIdentity = StaticClassIdentity::new("int32");
    pub const UINT32: StaticClassIdentity = StaticClassIdentity::new("uint32");
    pub const INT64: StaticClassIdentity = StaticClassIdentity::new("int64");
    pub const UINT64: StaticClassIdentity = StaticClassIdentity::new("uint64");
    pub const TABLE: StaticClassIdentity = StaticClassIdentity::new("table");
    pub const TIMETABLE: StaticClassIdentity = StaticClassIdentity::new("timetable");
    pub const CATEGORICAL: StaticClassIdentity = StaticClassIdentity::new("categorical");
    pub const DICTIONARY: StaticClassIdentity = StaticClassIdentity::new("dictionary");
    pub const DATETIME: StaticClassIdentity = StaticClassIdentity::new("datetime");
    pub const DURATION: StaticClassIdentity = StaticClassIdentity::new("duration");
    pub const CALENDAR_DURATION: StaticClassIdentity = StaticClassIdentity::new("calendarDuration");
    pub const TRANSFER_FUNCTION: StaticClassIdentity = StaticClassIdentity::new("tf");
    pub const STATE_SPACE: StaticClassIdentity = StaticClassIdentity::new("ss");
    pub const HANDLE: StaticClassIdentity = StaticClassIdentity::new("handle");
    pub const DYNAMIC_PROPERTIES: StaticClassIdentity = StaticClassIdentity::new("dynamicprops");
    pub const METADATA_PROPERTY: StaticClassIdentity =
        StaticClassIdentity::new("matlab.metadata.Property");
    pub const METADATA_DYNAMIC_PROPERTY: StaticClassIdentity =
        StaticClassIdentity::new("matlab.metadata.DynamicProperty");
    pub const UNIT_TEST_CASE: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.TestCase");
    pub const GPU_ARRAY: StaticClassIdentity = StaticClassIdentity::new("gpuArray");
    pub const EVENT_LISTENER: StaticClassIdentity = StaticClassIdentity::new("event.listener");
    pub const MEXCEPTION: StaticClassIdentity = StaticClassIdentity::new("MException");
    pub const TOKENIZED_DOCUMENT: StaticClassIdentity =
        StaticClassIdentity::new("tokenizedDocument");
    pub const BAG_OF_WORDS: StaticClassIdentity = StaticClassIdentity::new("bagOfWords");
    pub const BAG_OF_NGRAMS: StaticClassIdentity = StaticClassIdentity::new("bagOfNgrams");
    pub const LDA_MODEL: StaticClassIdentity = StaticClassIdentity::new("ldaModel");
    pub const PYTHON_ARGUMENTS: StaticClassIdentity =
        StaticClassIdentity::new("RunMat.PythonArguments");
    pub const UNIT_TEST_DIAGNOSTIC: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.diagnostics.Diagnostic");
    pub const UNIT_TEST_CONSTRAINT: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.constraints.Constraint");
    pub const UNIT_TEST_RUNNER_PLUGIN: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.plugins.TestRunnerPlugin");
    pub const UNIT_TEST_CODE_COVERAGE_PLUGIN: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.plugins.CodeCoveragePlugin");
    pub const UNIT_TEST_IS_EQUAL_TO: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.constraints.IsEqualTo");
    pub const UNIT_TEST_IS_TRUE: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.constraints.IsTrue");
    pub const UNIT_TEST_IS_FALSE: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.constraints.IsFalse");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_identity_round_trips_qualified_names() {
        let qualified = QualifiedName(vec![
            SymbolName("package".into()),
            SymbolName("Widget".into()),
        ]);
        let identity = ClassIdentity::from_qualified_name(&qualified).unwrap();
        assert_eq!(identity.display_name(), "package.Widget");
        assert_eq!(identity.qualified_name(), qualified);
    }

    #[test]
    fn class_identity_rejects_noncanonical_names() {
        assert_eq!(ClassIdentity::new(""), Err(InvalidClassIdentity::Empty));
        assert_eq!(
            ClassIdentity::new("package..Widget"),
            Err(InvalidClassIdentity::EmptySegment)
        );
    }

    #[test]
    fn standard_identities_compare_without_source_string_matching() {
        let table = ClassIdentity::new("table").unwrap();
        assert!(table.is(standard::TABLE));
        assert!(!table.is(standard::TIMETABLE));
    }
}
