use super::{DELIMITED_IMPORT_CLASS, SPREADSHEET_IMPORT_CLASS, VARIABLE_IMPORT_CLASS};
use runmat_types::{MemberAccess, StaticClassIdentity};
use std::collections::HashMap;

static DELIMITED_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DELIMITED_IMPORT_CLASS);
static SPREADSHEET_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(SPREADSHEET_IMPORT_CLASS);
static VARIABLE_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(VARIABLE_IMPORT_CLASS);

pub(crate) fn ensure_import_options_classes_registered() {
    DELIMITED_REGISTERED.ensure(|| register(DELIMITED_IMPORT_CLASS, IMPORT_PROPERTIES));
    SPREADSHEET_REGISTERED.ensure(|| register(SPREADSHEET_IMPORT_CLASS, IMPORT_PROPERTIES));
    VARIABLE_REGISTERED.ensure(|| register(VARIABLE_IMPORT_CLASS, VARIABLE_PROPERTIES));
}

const IMPORT_PROPERTIES: &[&str] = &[
    "FileType",
    "Delimiter",
    "NumHeaderLines",
    "ReadVariableNames",
    "ReadRowNames",
    "NumVariables",
    "VariableNames",
    "VariableTypes",
    "SelectedVariableNames",
    "Range",
    "DataRange",
    "Sheet",
    "TreatAsMissing",
    "PreserveVariableNames",
    "VariableNamingRule",
    "EmptyLineRule",
    "TextType",
    "DatetimeType",
    "Encoding",
];

const VARIABLE_PROPERTIES: &[&str] = &[
    "Name",
    "Type",
    "TreatAsMissing",
    "DecimalSeparator",
    "ThousandsSeparator",
    "EmptyFieldRule",
    "WhitespaceRule",
    "FillValue",
    "InputFormat",
];

fn register(class: StaticClassIdentity, names: &[&str]) {
    let mut properties = HashMap::new();
    for name in names {
        properties.insert(
            (*name).into(),
            crate::class_registry::RuntimeProperty {
                name: (*name).into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Public,
                default_value: None,
            },
        );
    }
    crate::class_registry::register_builtin_class(
        class,
        crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: class.into(),
            parent: None,
            properties,
            methods: HashMap::new(),
        },
    );
}
