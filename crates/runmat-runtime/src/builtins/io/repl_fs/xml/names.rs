//! XML 1.0 (fifth edition) Name productions, shared by format boundaries.
//! https://www.w3.org/TR/xml/#NT-Name

use super::*;

pub(super) struct XmlName<'a>(&'a str);

impl<'a> XmlName<'a> {
    pub(super) fn parse(name: &'a str) -> BuiltinResult<Self> {
        let mut chars = name.chars();
        if !chars.next().is_some_and(name_start) || !chars.all(name_char) {
            return Err(compat_error("xmlwrite", "xmlwrite: invalid XML name"));
        }
        Ok(Self(name))
    }

    pub(super) fn as_str(&self) -> &'a str {
        self.0
    }
}

impl std::fmt::Display for XmlName<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

pub(super) fn name_start(ch: char) -> bool {
    matches!(ch,
        ':' | '_' | 'A'..='Z' | 'a'..='z'
        | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}' | '\u{F8}'..='\u{2FF}'
        | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}'
        | '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}'
        | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}'
        | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}'
        | '\u{10000}'..='\u{EFFFF}'
    )
}

pub(super) fn name_char(ch: char) -> bool {
    name_start(ch)
        || matches!(ch, '-' | '.' | '0'..='9' | '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn xml_names_preserve_unicode_namespace_and_continuation_characters() {
        for name in [
            "root",
            "ns:node",
            "_value-1.2",
            "温度",
            "e\u{301}",
            "a\u{B7}b",
        ] {
            assert_eq!(XmlName::parse(name).expect("valid XML name").as_str(), name);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn xml_names_reject_empty_invalid_initial_and_markup_characters() {
        for name in [
            "",
            "1root",
            "-root",
            "\u{301}root",
            "two words",
            "a<b",
            "a/b",
            "a=b",
            "a&b",
        ] {
            assert!(XmlName::parse(name).is_err(), "invalid XML name {name:?}");
        }
    }
}
