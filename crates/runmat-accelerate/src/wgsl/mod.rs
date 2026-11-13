//! Device-independent WGSL source text shared by fusion and WGPU.

#[cfg(any(feature = "wgpu", test))]
use std::borrow::Cow;

pub(crate) mod ieee_words;
#[cfg(any(feature = "wgpu", test))]
pub(crate) mod f64_math;
#[cfg(feature = "wgpu")]
pub(crate) mod integer;
pub(crate) mod nan_predicate;
#[cfg(test)]
pub(crate) mod validation_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericPrecision {
    F32,
    F64,
}

pub(crate) mod host_buffers;

#[cfg(any(feature = "wgpu", test))]
#[derive(Debug)]
pub(crate) enum TokenSource<'a> {
    Borrowed(&'a str),
    Owned(host_buffers::ClaimedString),
}
#[cfg(any(feature = "wgpu", test))]
impl AsRef<str> for TokenSource<'_> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
#[cfg(any(feature = "wgpu", test))]
impl std::ops::Deref for TokenSource<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
#[cfg(any(feature = "wgpu", test))]
impl std::fmt::Display for TokenSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
#[cfg(any(feature = "wgpu", test))]
impl<'a> TokenSource<'a> {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            Self::Borrowed(source) => source,
            Self::Owned(source) => source.as_str(),
        }
    }
    /// Coupled transfer; any owned source must keep its claim in the successor.
    pub(crate) fn into_parts(self) -> (Cow<'a, str>, host_buffers::Claim) {
        match self {
            Self::Borrowed(source) => (Cow::Borrowed(source), None),
            Self::Owned(source) => {
                let (source, claim) = source.into_parts();
                (Cow::Owned(source), claim)
            }
        }
    }
}

/// One checked token writer; fixed sources remain borrowed without allocation.
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn substitute_tokens<'source>(
    src: &'source str,
    wg: u32,
    tile: u32,
    metadata: Option<&dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
) -> Result<TokenSource<'source>, host_buffers::Error> {
    use host_buffers::{formatted_len, ClaimedString, Error};
    if !src.contains("@WG@") && !src.contains("@MT@") {
        return Ok(TokenSource::Borrowed(src));
    }
    let wg_count = src.matches("@WG@").count();
    let tile_count = src.matches("@MT@").count();
    let replaced = wg_count
        .checked_add(tile_count)
        .and_then(|count| count.checked_mul(4))
        .ok_or(Error::Overflow)?;
    let wg_bytes = formatted_len(format_args!("{wg}"))?;
    let tile_bytes = formatted_len(format_args!("{tile}"))?;
    let added = wg_count
        .checked_mul(wg_bytes)
        .and_then(|bytes| {
            tile_count
                .checked_mul(tile_bytes)
                .and_then(|tile_bytes| bytes.checked_add(tile_bytes))
        })
        .ok_or(Error::Overflow)?;
    let len = src
        .len()
        .checked_sub(replaced)
        .and_then(|bytes| bytes.checked_add(added))
        .ok_or(Error::Overflow)?;
    let mut result = ClaimedString::with_capacity(len, metadata)?;
    let mut rest = src;
    loop {
        let next = match (rest.find("@WG@"), rest.find("@MT@")) {
            (None, None) => break,
            (Some(index), None) => (index, wg),
            (None, Some(index)) => (index, tile),
            (Some(a), Some(b)) => {
                if a <= b {
                    (a, wg)
                } else {
                    (b, tile)
                }
            }
        };
        result.push_str(&rest[..next.0], metadata)?;
        result.append_fmt(format_args!("{}", next.1), metadata)?;
        rest = &rest[next.0 + 4..];
    }
    result.push_str(rest, metadata)?;
    if result.len() != len {
        return Err(Error::Overflow);
    }
    Ok(TokenSource::Owned(result))
}

#[cfg(test)]
mod tests {
    use super::{substitute_tokens, TokenSource};

    #[test]
    fn production_tokens_keep_borrowed_and_owned_paths() {
        assert!(matches!(
            substitute_tokens("fn fixed() {}", 64, 8, None).unwrap(),
            TokenSource::Borrowed(_)
        ));
        assert_eq!(
            substitute_tokens("@compute @workgroup_size(@WG@) // @MT@", 64, 8, None)
                .unwrap()
                .as_str(),
            "@compute @workgroup_size(64) // 8"
        );
    }
}
