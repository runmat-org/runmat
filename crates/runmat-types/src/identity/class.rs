use super::{QualifiedName, SymbolName};
use serde::{Deserialize, Serialize};
use std::fmt;

#[path = "class/graphics.rs"]
mod graphics;

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

    /// Compare the existing qualified-name spelling projection without allocating
    /// its joined display buffer. This mirrors from_qualified_name exactly.
    pub fn matches_qualified_name(&self, name: &QualifiedName) -> bool {
        if name.0.is_empty() {
            return false;
        }
        let mut remaining = self.0.as_str();
        for (index, part) in name.0.iter().enumerate() {
            if part.0.is_empty() {
                return false;
            }
            if index > 0 {
                let Some(tail) = remaining.strip_prefix('.') else {
                    return false;
                };
                remaining = tail;
            }
            let Some(tail) = remaining.strip_prefix(part.0.as_str()) else {
                return false;
            };
            remaining = tail;
        }
        remaining.is_empty()
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

    /// Physical UTF-8 allocation capacity. Value-domain retained owners use
    /// this only to quote the original backing they share; semantic code
    /// should continue to use [`Self::display_name`].
    pub fn allocation_capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn is(&self, expected: StaticClassIdentity) -> bool {
        self == &expected
    }

    /// Compatibility comparison for public class-query designators. Registry
    /// identities themselves retain their original case and source spelling.
    pub fn eq_ignore_ascii_case(&self, other: &ClassIdentity) -> bool {
        self.0.eq_ignore_ascii_case(&other.0)
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

// Borrow the canonical identity spelling for allocation-free registry queries.
impl std::borrow::Borrow<str> for ClassIdentity {
    fn borrow(&self) -> &str {
        self.display_name()
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
    pub use super::graphics::*;
    use super::{ClassIdentity, StaticClassIdentity, StaticClassNamespace};

    pub const LANGUAGE_CORRECTION_NAMESPACE: StaticClassNamespace =
        StaticClassNamespace::new("matlab.lang.correction");

    pub const DOUBLE: StaticClassIdentity = StaticClassIdentity::new("double");
    pub const SINGLE: StaticClassIdentity = StaticClassIdentity::new("single");
    pub const LOGICAL: StaticClassIdentity = StaticClassIdentity::new("logical");
    pub const CHAR: StaticClassIdentity = StaticClassIdentity::new("char");
    pub const STRING: StaticClassIdentity = StaticClassIdentity::new("string");
    pub const PATTERN: StaticClassIdentity = StaticClassIdentity::new("pattern");
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
    pub const DECOMPOSITION: StaticClassIdentity = StaticClassIdentity::new("decomposition");
    pub const DATETIME: StaticClassIdentity = StaticClassIdentity::new("datetime");
    pub const DURATION: StaticClassIdentity = StaticClassIdentity::new("duration");
    pub const DL_ARRAY: StaticClassIdentity = StaticClassIdentity::new("dlarray");
    pub const DL_NETWORK: StaticClassIdentity = StaticClassIdentity::new("dlnetwork");
    pub const SERIES_NETWORK: StaticClassIdentity = StaticClassIdentity::new("SeriesNetwork");
    pub const DAG_NETWORK: StaticClassIdentity = StaticClassIdentity::new("DAGNetwork");
    pub const LINEAR_MODEL: StaticClassIdentity = StaticClassIdentity::new("LinearModel");
    pub const OPTIMIZABLE_VARIABLE: StaticClassIdentity =
        StaticClassIdentity::new("optimizableVariable");
    pub const OPTIMIZATION_EXPRESSION: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationExpression");
    pub const OPTIMIZATION_CONSTRAINT: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationConstraint");
    pub const OPTIMIZATION_EQUALITY: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationEquality");
    pub const OPTIMIZATION_INEQUALITY: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationInequality");
    pub const OPTIMIZATION_EXITFLAG: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationExitflag");
    pub const OPTIMIZATION_PROBLEM: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationProblem");
    pub const OPTIMIZATION_VARIABLE: StaticClassIdentity =
        StaticClassIdentity::new("optim.problemdef.OptimizationVariable");
    pub const INTEGER_CONSTRAINT: StaticClassIdentity =
        StaticClassIdentity::new("IntegerConstraint");
    pub const BAYESIAN_OPTIMIZATION: StaticClassIdentity =
        StaticClassIdentity::new("BayesianOptimization");
    pub const CLASSIFICATION_TREE: StaticClassIdentity =
        StaticClassIdentity::new("ClassificationTree");
    pub const CLASSIFICATION_LINEAR: StaticClassIdentity =
        StaticClassIdentity::new("ClassificationLinear");
    pub const CVPARTITION: StaticClassIdentity = StaticClassIdentity::new("cvpartition");
    pub const PROBABILITY_DISTRIBUTION: StaticClassIdentity =
        StaticClassIdentity::new("ProbabilityDistribution");
    pub const CALENDAR_DURATION: StaticClassIdentity = StaticClassIdentity::new("calendarDuration");
    pub const TRANSFER_FUNCTION: StaticClassIdentity = StaticClassIdentity::new("tf");
    pub const STATE_SPACE: StaticClassIdentity = StaticClassIdentity::new("ss");
    pub const HANDLE: StaticClassIdentity = StaticClassIdentity::new("handle");
    pub const DYNAMIC_PROPERTIES: StaticClassIdentity = StaticClassIdentity::new("dynamicprops");
    pub const EVENT_DATA: StaticClassIdentity = StaticClassIdentity::new("event.EventData");
    pub const FIGURE: StaticClassIdentity = StaticClassIdentity::new("matlab.ui.Figure");
    pub const AXES: StaticClassIdentity = StaticClassIdentity::new("matlab.graphics.axis.Axes");
    pub const ZOOM_CONTROL: StaticClassIdentity =
        StaticClassIdentity::new("matlab.graphics.interaction.internal.zoom");
    pub const PAN_CONTROL: StaticClassIdentity =
        StaticClassIdentity::new("matlab.graphics.interaction.internal.pan");
    pub const DATA_CURSOR_CONTROL: StaticClassIdentity =
        StaticClassIdentity::new("matlab.graphics.shape.internal.DataCursorManager");

    pub fn is_graphics_property_control(identity: &ClassIdentity) -> bool {
        identity.is(ZOOM_CONTROL) || identity.is(PAN_CONTROL) || identity.is(DATA_CURSOR_CONTROL)
    }
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
    pub const UNIT_TEST_STRING_DIAGNOSTIC: StaticClassIdentity =
        StaticClassIdentity::new("matlab.unittest.diagnostics.StringDiagnostic");
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

    /// Returns whether an identity belongs to the standard tabular family.
    ///
    /// Semantic consumers use this predicate instead of reconstructing the
    /// family from display spellings or repeating its member list.
    pub fn is_tabular(identity: &ClassIdentity) -> bool {
        identity.is(TABLE) || identity.is(TIMETABLE)
    }
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
        assert!(identity.matches_qualified_name(&qualified));
        assert!(identity
            .matches_qualified_name(&QualifiedName(vec![SymbolName("package.Widget".into())])));
        assert!(!identity.matches_qualified_name(&QualifiedName(vec![SymbolName(String::new())])));
        assert!(!identity.matches_qualified_name(&QualifiedName(vec![
            SymbolName("other".into()),
            SymbolName("Widget".into())
        ])));
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
        assert!(standard::is_tabular(&table));
    }
}
