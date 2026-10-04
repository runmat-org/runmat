//! The one public spelling boundary for weboptions property identities.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebOptionField {
    ContentType,
    Timeout,
    KeyName,
    KeyValue,
    HeaderFields,
    UserAgent,
    Username,
    Password,
    RequestMethod,
    MediaType,
    QueryParameters,
}

impl WebOptionField {
    pub fn decode(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "contenttype" => Some(Self::ContentType),
            "timeout" => Some(Self::Timeout),
            "keyname" => Some(Self::KeyName),
            "keyvalue" => Some(Self::KeyValue),
            "headerfields" => Some(Self::HeaderFields),
            "useragent" => Some(Self::UserAgent),
            "username" => Some(Self::Username),
            "password" => Some(Self::Password),
            "requestmethod" => Some(Self::RequestMethod),
            "mediatype" => Some(Self::MediaType),
            "queryparameters" => Some(Self::QueryParameters),
            _ => None,
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::ContentType => "ContentType",
            Self::Timeout => "Timeout",
            Self::KeyName => "KeyName",
            Self::KeyValue => "KeyValue",
            Self::HeaderFields => "HeaderFields",
            Self::UserAgent => "UserAgent",
            Self::Username => "Username",
            Self::Password => "Password",
            Self::RequestMethod => "RequestMethod",
            Self::MediaType => "MediaType",
            Self::QueryParameters => "QueryParameters",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documented_spellings_decode_case_insensitively_without_aliases() {
        for field in [
            WebOptionField::ContentType,
            WebOptionField::Timeout,
            WebOptionField::KeyName,
            WebOptionField::KeyValue,
            WebOptionField::HeaderFields,
            WebOptionField::UserAgent,
            WebOptionField::Username,
            WebOptionField::Password,
            WebOptionField::RequestMethod,
            WebOptionField::MediaType,
            WebOptionField::QueryParameters,
        ] {
            assert_eq!(WebOptionField::decode(field.canonical_name()), Some(field));
            assert_eq!(
                WebOptionField::decode(&field.canonical_name().to_ascii_uppercase()),
                Some(field)
            );
        }
        assert_eq!(WebOptionField::decode("CharacterEncoding"), None);
        assert_eq!(WebOptionField::decode("Timeout "), None);
    }
}
