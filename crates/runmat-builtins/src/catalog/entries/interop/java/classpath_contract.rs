//! Public classpath option identity and the bounded runtime-to-adapter wire.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JavaClasspathQuery {
    Dynamic,
    Static,
    All,
}

impl JavaClasspathQuery {
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Dynamic => "dynamic",
            Self::Static => "static",
            Self::All => "all",
        }
    }

    pub fn parse_wire(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("dynamic") {
            Some(Self::Dynamic)
        } else if name.eq_ignore_ascii_case("static") || name.eq_ignore_ascii_case("bootstrap") {
            Some(Self::Static)
        } else if name.eq_ignore_ascii_case("all") {
            Some(Self::All)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JavaClasspathControl {
    Query(JavaClasspathQuery),
    Verbosity,
    Path,
}

impl JavaClasspathControl {
    pub fn parse_public(name: &str) -> Self {
        if name.eq_ignore_ascii_case("-dynamic") {
            Self::Query(JavaClasspathQuery::Dynamic)
        } else if name.eq_ignore_ascii_case("-static") {
            Self::Query(JavaClasspathQuery::Static)
        } else if name.eq_ignore_ascii_case("-all") {
            Self::Query(JavaClasspathQuery::All)
        } else if name.eq_ignore_ascii_case("-v0") || name.eq_ignore_ascii_case("-v1") {
            Self::Verbosity
        } else {
            Self::Path
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JavaInsertionPosition {
    Begin,
    End,
}

impl JavaInsertionPosition {
    pub fn from_public_terminal(name: &str) -> Option<Self> {
        (name == "-end").then_some(Self::End)
    }

    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Begin => "begin",
            Self::End => "end",
        }
    }

    pub fn parse_wire(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("begin") {
            Some(Self::Begin)
        } else if name.eq_ignore_ascii_case("end") {
            Some(Self::End)
        } else {
            None
        }
    }
}
