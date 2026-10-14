#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableSource {
    pub owner_identity: String,
    pub relative_path: String,
    pub text: String,
}

impl ExecutableSource {
    /// Bind source text to an exact frozen project descriptor. This pure
    /// admission is shared by native and browser producers; it performs no I/O.
    pub fn from_project_source(
        source: &runmat_package::StableSourceId,
        text: impl Into<String>,
    ) -> Result<Self, String> {
        let text = text.into();
        if runmat_package::ContentDigest::sha256(text.as_bytes()) != source.content_digest {
            return Err("source text differs from its frozen project descriptor".into());
        }
        Ok(Self::new(
            source.package_instance.to_string(),
            source.relative_path.to_string(),
            text,
        ))
    }

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
