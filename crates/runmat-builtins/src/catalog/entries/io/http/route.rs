//! One HTTP/HTTPS URL decoding boundary shared by static facts and runtime.

use std::error::Error;
use std::fmt;
use url::Url;

#[derive(Debug)]
pub enum HttpUrlError {
    Parse(url::ParseError),
    UnsupportedScheme(String),
    MissingHost,
}

impl fmt::Display for HttpUrlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(source) => write!(formatter, "{source}"),
            Self::UnsupportedScheme(scheme) => write!(
                formatter,
                "unsupported URL scheme '{scheme}'; expected HTTP or HTTPS"
            ),
            Self::MissingHost => formatter.write_str("HTTP URL must have a host"),
        }
    }
}

impl Error for HttpUrlError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Parse(source) => Some(source),
            _ => None,
        }
    }
}

/// Preserve the parsed `Url` for request construction; do not reparse after
/// operand materialization or network admission.
pub fn parse_http_url(source: &str) -> Result<Url, HttpUrlError> {
    let url = Url::parse(source).map_err(HttpUrlError::Parse)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(HttpUrlError::UnsupportedScheme(url.scheme().to_owned()));
    }
    if url.host().is_none() {
        return Err(HttpUrlError::MissingHost);
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_parser_rejects_unsupported_schemes_and_preserves_parsed_url() {
        assert_eq!(
            parse_http_url("HTTP://example.test/p?a=1")
                .unwrap()
                .scheme(),
            "http"
        );
        assert!(matches!(
            parse_http_url("ftp://example.test/a"),
            Err(HttpUrlError::UnsupportedScheme(_))
        ));
        assert!(matches!(
            parse_http_url("mailto:alice@example.test"),
            Err(HttpUrlError::UnsupportedScheme(_))
        ));
        assert!(matches!(
            parse_http_url("relative/path"),
            Err(HttpUrlError::Parse(_))
        ));
        assert!(matches!(
            parse_http_url("http://"),
            Err(HttpUrlError::Parse(_))
        ));
    }
}
