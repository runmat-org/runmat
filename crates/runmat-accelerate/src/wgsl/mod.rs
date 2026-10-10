//! Device-independent WGSL source text shared by fusion and WGPU.

#[cfg(any(feature = "wgpu", test))]
use std::borrow::Cow;

pub(crate) mod ieee_words;
pub(crate) mod nan_predicate;
#[cfg(test)]
pub(crate) mod validation_tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericPrecision {
    F32,
    F64,
}

/// Substitute production launch tokens without allocating for fixed sources.
#[cfg(any(feature = "wgpu", test))]
pub(crate) fn substitute_tokens(src: &str, wg: u32, tile: u32) -> Cow<'_, str> {
    if src.contains("@WG@") || src.contains("@MT@") {
        let mut source = src.to_string();
        source = source.replace("@WG@", &wg.to_string());
        source = source.replace("@MT@", &tile.to_string());
        Cow::Owned(source)
    } else {
        Cow::Borrowed(src)
    }
}

#[cfg(test)]
mod tests {
    use super::substitute_tokens;
    use std::borrow::Cow;

    #[test]
    fn production_tokens_keep_borrowed_and_owned_paths() {
        assert!(matches!(
            substitute_tokens("fn fixed() {}", 64, 8),
            Cow::Borrowed(_)
        ));
        assert_eq!(
            substitute_tokens("@compute @workgroup_size(@WG@) // @MT@", 64, 8),
            "@compute @workgroup_size(64) // 8"
        );
    }
}
