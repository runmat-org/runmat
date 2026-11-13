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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PointerMethod {
    SetDataType,
    IsNull,
    Reshape,
}

pub(super) const POINTER_TYPE_NAME: &str = "lib.pointer";
pub(super) const POINTER_TYPE_VERSION: u32 = 1;

/// Nominal dispatch classification only, not a resource/host capability.
/// The adapter still validates the live handle and required access on invoke.
pub(crate) fn declared_pointer_method(
    identity: &runmat_types::ForeignTypeIdentity,
    member: &str,
) -> Option<PointerMethod> {
    if identity.family != super::NATIVE_FFI_ADAPTER_ID
        || identity.name != POINTER_TYPE_NAME
        || identity.version != POINTER_TYPE_VERSION
    {
        return None;
    }
    PointerMethod::parse(member)
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
        let mut identity =
            super::super::pointer_view::pointer_type_identity(POINTER_TYPE_NAME.into());
        assert_eq!(
            declared_pointer_method(&identity, "setDataType"),
            Some(PointerMethod::SetDataType)
        );
        assert!(declared_pointer_method(&identity, "Value").is_none());
        identity.name = "lib.fixture_record".into();
        assert!(declared_pointer_method(&identity, "setdatatype").is_none());
        identity.name = POINTER_TYPE_NAME.into();
        identity.version += 1;
        assert!(declared_pointer_method(&identity, "setdatatype").is_none());
        identity.version = POINTER_TYPE_VERSION;
        identity.family = "another-adapter".into();
        assert!(declared_pointer_method(&identity, "setdatatype").is_none());
    }
}
