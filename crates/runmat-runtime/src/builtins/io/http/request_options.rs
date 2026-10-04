//! Prepared public request-option names shared by webread and webwrite.

use runmat_builtins::WebOptionField;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum RequestOptionName {
    Field(WebOptionField),
    Query(String),
}

impl RequestOptionName {
    pub(super) fn decode(name: &str, leading_options_struct: bool) -> Self {
        match WebOptionField::decode(name) {
            // Only the leading weboptions structure gives these names header meaning.
            // Direct pairs have always been ordinary query parameters.
            Some(WebOptionField::KeyName | WebOptionField::KeyValue) if !leading_options_struct => {
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
            RequestOptionName::decode("KeyName", false),
            RequestOptionName::Query("KeyName".into())
        );
        assert_eq!(
            RequestOptionName::decode("Timeout", false),
            RequestOptionName::Field(WebOptionField::Timeout)
        );
    }
}
