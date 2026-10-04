//! Public native pointer member/method admission. Foreign structure fields
//! remain case-sensitive user names, while the reserved members are normalized
//! exactly once at the adapter boundary.

pub(super) enum PointerMember {
    Value,
    DataType,
    Field(String),
}

impl PointerMember {
    pub(super) fn parse(name: String) -> Self {
        if name.eq_ignore_ascii_case("Value") {
            Self::Value
        } else if name.eq_ignore_ascii_case("DataType") {
            Self::DataType
        } else {
            Self::Field(name)
        }
    }
}

pub(super) enum PointerMethod {
    SetDataType,
    IsNull,
    Reshape,
}

impl PointerMethod {
    pub(super) fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("setdatatype") {
            Some(Self::SetDataType)
        } else if name.eq_ignore_ascii_case("isnull") {
            Some(Self::IsNull)
        } else if name.eq_ignore_ascii_case("reshape") {
            Some(Self::Reshape)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_roles_are_case_insensitive_but_structure_fields_keep_spelling() {
        assert!(matches!(
            PointerMember::parse("vAlUe".into()),
            PointerMember::Value
        ));
        assert!(matches!(
            PointerMember::parse("dataTYPE".into()),
            PointerMember::DataType
        ));
        assert!(
            matches!(PointerMember::parse("Left".into()), PointerMember::Field(name) if name == "Left")
        );
        assert!(matches!(
            PointerMethod::parse("isNULL"),
            Some(PointerMethod::IsNull)
        ));
        assert!(PointerMethod::parse("unknown").is_none());
    }
}
