use url::Url;

/// The public source spelling is decoded once and retained across catalog
/// admission and runtime dispatch. Non-HTTP URL spellings historically name
/// local paths for this identity; changing that rule needs separate review.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnzipSourceRoute {
    LocalPath(String),
    Http(Url),
}

impl UnzipSourceRoute {
    pub fn parse(source: String) -> Self {
        if let Ok(url) = Url::parse(&source) {
            if matches!(url.scheme(), "http" | "https") {
                return Self::Http(url);
            }
        }
        Self::LocalPath(source)
    }
}
