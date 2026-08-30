use runmat_types::MemberAccess;
use runmat_value::Value;
use std::collections::HashMap;

use crate::{OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD};

use super::object::default_properties_for_class;
use super::{
    ARRAY_DATASTORE_CLASS, CATEGORICAL_CLASS, DICTIONARY_CLASS, FILE_DATASTORE_CLASS,
    PARQUET_DATASTORE_CLASS, PROPERTIES_MEMBER, ROWFILTER_CLASS, TABLE_CLASS, TIMERANGE_CLASS,
    TIMETABLE_CLASS, UITABLE_CLASS, VARTYPE_CLASS,
};

static TABLE_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(TABLE_CLASS);

pub fn ensure_table_class_registered() {
    TABLE_CLASS_REGISTERED.ensure(|| {
        register_tabular_class(TABLE_CLASS);
        register_tabular_class(TIMETABLE_CLASS);
        register_plain_object_class(CATEGORICAL_CLASS, &["Codes", "Categories", "Ordinal"]);
        register_dictionary_class();
        register_plain_object_class(TIMERANGE_CLASS, &["Start", "End", "Inclusivity"]);
        register_plain_object_class(VARTYPE_CLASS, &["Type"]);
        register_plain_object_class(ROWFILTER_CLASS, &["Variables", "Predicate"]);
        register_plain_object_class(
            ARRAY_DATASTORE_CLASS,
            &["Data", "ReadSize", "IterationDimension", "OutputType"],
        );
        register_plain_object_class(
            FILE_DATASTORE_CLASS,
            &[
                "Files",
                "ReadFcn",
                "FileExtensions",
                "IncludeSubfolders",
                "ReadMode",
            ],
        );
        register_plain_object_class(PARQUET_DATASTORE_CLASS, &["Files"]);
        register_plain_object_class(UITABLE_CLASS, &["Data", "ColumnName", "RowName"]);
    });
}

fn register_tabular_class(name: runmat_types::StaticClassIdentity) {
    let mut properties = HashMap::new();
    properties.insert(
        PROPERTIES_MEMBER.into(),
        crate::class_registry::RuntimeProperty {
            name: PROPERTIES_MEMBER.into(),
            is_static: false,
            is_constant: false,
            is_dependent: false,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            default_value: Some(Value::Struct(default_properties_for_class(
                name,
                Vec::new(),
                None,
            ))),
        },
    );

    let mut methods = HashMap::new();
    for method_name in [OBJECT_SUBSREF_METHOD, OBJECT_SUBSASGN_METHOD] {
        methods.insert(
            method_name.into(),
            crate::class_registry::RuntimeMethod {
                name: method_name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: format!("{TABLE_CLASS}.{method_name}"),
                implicit_class_argument: None,
            },
        );
    }

    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: name.into(),
        parent: None,
        properties,
        methods,
    });
}

fn register_plain_object_class(name: runmat_types::StaticClassIdentity, property_names: &[&str]) {
    let mut properties = HashMap::new();
    for property_name in property_names {
        properties.insert(
            (*property_name).into(),
            crate::class_registry::RuntimeProperty {
                name: (*property_name).into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
    }
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: name.into(),
        parent: None,
        properties,
        methods: HashMap::new(),
    });
}

fn register_dictionary_class() {
    let mut properties = HashMap::new();
    for property_name in ["Keys", "Values"] {
        properties.insert(
            property_name.into(),
            crate::class_registry::RuntimeProperty {
                name: property_name.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
    }
    let mut methods = HashMap::new();
    for method_name in [OBJECT_SUBSREF_METHOD, OBJECT_SUBSASGN_METHOD] {
        methods.insert(
            method_name.into(),
            crate::class_registry::RuntimeMethod {
                name: method_name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: format!("{DICTIONARY_CLASS}.{method_name}"),
                implicit_class_argument: None,
            },
        );
    }
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: DICTIONARY_CLASS.into(),
        parent: None,
        properties,
        methods,
    });
}
