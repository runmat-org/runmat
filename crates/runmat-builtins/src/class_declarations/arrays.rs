//! Immutable representation declarations for object-backed array classes.

use super::public_instance_method;
use runmat_types::{
    standard, ClassArrayPayload, ClassArrayShape, ClassIdentity, ClassKind,
    ExternalClassDeclaration, ExternalPropertyDeclaration, PropertyAttributes, StaticMethodName,
};

pub const DURATION_DAYS_PROPERTY: &str = "__days";
pub const DATETIME_SERIAL_PROPERTY: &str = "__serial";
pub const CALENDAR_MONTHS_PROPERTY: &str = "__months";
pub const CALENDAR_DAYS_PROPERTY: &str = "__days";
pub const DLARRAY_DATA_PROPERTY: &str = "Data";

pub(super) fn declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    find(identity).cloned()
}

pub(super) fn shape(identity: &ClassIdentity) -> Option<&'static ClassArrayShape> {
    find(identity).and_then(|declaration| declaration.array_shape.as_ref())
}

fn find(identity: &ClassIdentity) -> Option<&'static ExternalClassDeclaration> {
    static DECLARATIONS: std::sync::OnceLock<
        std::collections::BTreeMap<ClassIdentity, ExternalClassDeclaration>,
    > = std::sync::OnceLock::new();
    DECLARATIONS
        .get_or_init(|| {
            [
                standard::DURATION,
                standard::DATETIME,
                standard::CALENDAR_DURATION,
                standard::DL_ARRAY,
            ]
            .into_iter()
            .map(|class| {
                (
                    class.owned(),
                    build_declaration(&class.owned()).expect("declared standard array class"),
                )
            })
            .collect()
        })
        .get(identity)
}

fn build_declaration(identity: &ClassIdentity) -> Option<ExternalClassDeclaration> {
    let (class, shape, properties, methods) = if identity.is(standard::DURATION) {
        (
            standard::DURATION,
            real_property(DURATION_DAYS_PROPERTY),
            vec!["Format"],
            elapsed_methods(),
        )
    } else if identity.is(standard::DATETIME) {
        (
            standard::DATETIME,
            real_property(DATETIME_SERIAL_PROPERTY),
            vec!["Format"],
            elapsed_methods(),
        )
    } else if identity.is(standard::CALENDAR_DURATION) {
        (
            standard::CALENDAR_DURATION,
            ClassArrayShape::ConsistentProperties {
                primary: CALENDAR_MONTHS_PROPERTY.into(),
                others: vec![CALENDAR_DAYS_PROPERTY.into()],
                payload: ClassArrayPayload::RealNumeric,
            },
            Vec::new(),
            vec!["plus", "minus", "eq", "ne"],
        )
    } else if identity.is(standard::DL_ARRAY) {
        (
            standard::DL_ARRAY,
            ClassArrayShape::Property {
                property: DLARRAY_DATA_PROPERTY.into(),
                payload: ClassArrayPayload::NumericOrLogical,
            },
            vec![DLARRAY_DATA_PROPERTY, "Format", "Labels"],
            vec!["plus", "minus", "times", "rdivide", "mtimes", "sum"],
        )
    } else {
        return None;
    };
    Some(ExternalClassDeclaration {
        name: identity.qualified_name(),
        parent: None,
        kind: ClassKind::Value,
        is_sealed: false,
        is_abstract: false,
        properties: properties
            .into_iter()
            .map(|name| ExternalPropertyDeclaration {
                name: name.into(),
                attributes: PropertyAttributes::default(),
            })
            .collect(),
        methods: methods
            .into_iter()
            .map(|name| public_instance_method(class, StaticMethodName::new(name)))
            .collect(),
        array_shape: Some(shape),
    })
}

fn elapsed_methods() -> Vec<&'static str> {
    vec![
        "subsref", "subsasgn", "plus", "minus", "eq", "ne", "lt", "le", "gt", "ge",
    ]
}

fn real_property(property: &str) -> ClassArrayShape {
    ClassArrayShape::Property {
        property: property.into(),
        payload: ClassArrayPayload::RealNumeric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_geometry_and_methods_share_the_class_declaration() {
        for class in [
            standard::DURATION,
            standard::DATETIME,
            standard::CALENDAR_DURATION,
            standard::DL_ARRAY,
        ] {
            let declaration = declaration(&class.owned()).unwrap();
            assert!(declaration.array_shape.is_some());
            assert!(!declaration.methods.is_empty());
            for method in declaration.methods {
                assert_eq!(
                    method.callable.display_name().unwrap(),
                    format!("{class}.{}", method.name.0)
                );
            }
        }
        assert!(declaration(&standard::STATE_SPACE.owned()).is_none());
    }
}
