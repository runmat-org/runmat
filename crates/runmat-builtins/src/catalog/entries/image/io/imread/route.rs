/// The source transport identity after public URL spelling has been parsed.
/// Runtime and catalog use this one scheme decision; a drive letter is a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImreadSourceRoute {
    LocalPath,
    FileUrl,
    Http,
    UnsupportedScheme,
}

impl ImreadSourceRoute {
    pub fn from_parsed_scheme(scheme: &str) -> Self {
        if scheme.len() == 1 {
            return Self::LocalPath;
        }
        match scheme {
            "http" | "https" => Self::Http,
            "file" => Self::FileUrl,
            _ => Self::UnsupportedScheme,
        }
    }

    /// Only establish a URL route from a literal when its authority is a
    /// conservative, certainly parseable ASCII host. Other spellings remain
    /// unknown to inference; runtime's URL parser remains authoritative.
    pub fn from_proven_literal(source: &str) -> Option<Self> {
        let (scheme, rest) = source.split_once("://")?;
        if scheme.is_empty()
            || !scheme.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_alphabetic()
                    || (index > 0 && (byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')))
            })
        {
            return None;
        }
        let route = Self::from_parsed_scheme(&scheme.to_ascii_lowercase());
        if route == Self::FileUrl {
            return Some(route);
        }
        let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
        if authority.is_empty()
            || !authority
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
        {
            return None;
        }
        Some(route)
    }
}
