#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableSource {
    pub owner_identity: String,
    pub relative_path: String,
    pub text: String,
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
        }
    }
}
