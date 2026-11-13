use url::Url;

/// One parsed source identity for catalog admission and runtime dispatch.
/// A failed URL parse and a single-letter drive scheme retain the historical
/// local-path route; no second lexical URL parser is used for literals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImreadSourceRoute {
    LocalPath,
    FileUrl(Url),
    InvalidFileUrlHost { host: String },
    Http(Url),
    UnsupportedScheme { scheme: String },
}

impl ImreadSourceRoute {
    pub fn parse(source: &str) -> Self {
        let Ok(url) = Url::parse(source) else {
            return Self::LocalPath;
        };
        let scheme = url.scheme();
        if scheme.len() == 1 {
            return Self::LocalPath;
        }
        match scheme {
            "http" | "https" => Self::Http(url),
            "file" => match url.host_str() {
                Some(host) if !host.is_empty() && !host.eq_ignore_ascii_case("localhost") => {
                    Self::InvalidFileUrlHost {
                        host: host.to_string(),
                    }
                }
                _ => Self::FileUrl(url),
            },
            _ => Self::UnsupportedScheme {
                scheme: scheme.to_string(),
            },
        }
    }
}
