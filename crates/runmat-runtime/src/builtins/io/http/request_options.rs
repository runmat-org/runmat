//! Prepared public request-option names shared by webread and webwrite.

use runmat_builtins::WebOptionField;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum RequestOptionName {
    Field(WebOptionField),
    Query(String),
}

impl RequestOptionName {
    pub(super) fn decode(name: &str) -> Self {
        match WebOptionField::decode(name) {
            // PreparedApiKey owns these only in the leading options structure.
            // Elsewhere, their original spelling remains an ordinary query name.
            Some(WebOptionField::KeyName | WebOptionField::KeyValue) => {
                Self::Query(name.to_owned())
            }
            Some(field) => Self::Field(field),
            None => Self::Query(name.to_owned()),
        }
    }
}

/// In argument order, matching the leading struct's ordered fields and each
/// trailing pair. A resident name is decoded only after its owning gather.
#[derive(Default)]
pub(super) struct PreparedRequestNames {
    pub(super) struct_fields: Vec<(String, RequestOptionName)>,
    pub(super) pair_fields: Vec<Option<RequestOptionName>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_api_key_spellings_remain_query_names() {
        assert_eq!(
            RequestOptionName::decode("KeyName"),
            RequestOptionName::Query("KeyName".into())
        );
        assert_eq!(
            RequestOptionName::decode("Timeout"),
            RequestOptionName::Field(WebOptionField::Timeout)
        );
    }
}
