#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableSource {
    pub owner_identity: String,
    pub relative_path: String,
    pub text: String,
    lookup_path: Option<String>,
}

impl ExecutableSource {
    pub fn new(
        owner_identity: impl Into<String>,
        relative_path: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            owner_identity: owner_identity.into(),
            relative_path: relative_path.into(),
            text: text.into(),
            lookup_path: None,
        }
    }

    /// Construct a source whose portable identity is logical while project discovery reads the
    /// already-frozen physical file selected by the caller.
    pub fn frozen(
        owner_identity: impl Into<String>,
        relative_path: impl Into<String>,
        lookup_path: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            owner_identity: owner_identity.into(),
            relative_path: relative_path.into(),
            text: text.into(),
            lookup_path: Some(lookup_path.into()),
        }
    }

    pub(crate) fn lookup_path(&self) -> &str {
        self.lookup_path.as_deref().unwrap_or(&self.relative_path)
    }
}
