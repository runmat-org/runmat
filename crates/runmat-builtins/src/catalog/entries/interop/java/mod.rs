//! Canonical public Java interop identities; session execution remains owned
//! by the foreign adapter.

mod add_path;
mod array;
mod classpath;
mod classpath_contract;
mod contract;
mod descriptor;
mod documentation_support;
mod environment;
mod feature;
mod inference;
mod isjava;
mod method;
mod method_edt;
mod object;
mod object_edt;
mod remove_path;
#[cfg(test)]
mod tests;

pub use add_path::JAVA_ADD_PATH_CATALOG_ENTRY;
pub use array::JAVA_ARRAY_CATALOG_ENTRY;
pub use classpath::JAVA_CLASSPATH_CATALOG_ENTRY;
pub use classpath_contract::{JavaClasspathControl, JavaClasspathQuery, JavaInsertionPosition};
pub use contract::{
    JavaEnvironmentControl, JavaExecutionMode, JavaFeature, JAVA_ENVIRONMENT_CLASS,
    JAVA_INTEGER_AUDIT,
};
pub use descriptor::{
    ISJAVA_DESCRIPTOR, JAVA_ADD_PATH_DESCRIPTOR, JAVA_ARRAY_DESCRIPTOR, JAVA_CLASSPATH_DESCRIPTOR,
    JAVA_ERROR_HOST_UNAVAILABLE, JAVA_ERROR_INVALID, JAVA_ERROR_UNAVAILABLE,
    JAVA_METHOD_DESCRIPTOR, JAVA_METHOD_EDT_DESCRIPTOR, JAVA_OBJECT_DESCRIPTOR,
    JAVA_OBJECT_EDT_DESCRIPTOR, JAVA_REMOVE_PATH_DESCRIPTOR, JENV_DESCRIPTOR, USEJAVA_DESCRIPTOR,
};
pub use environment::JENV_CATALOG_ENTRY;
pub use feature::USEJAVA_CATALOG_ENTRY;
pub use isjava::ISJAVA_CATALOG_ENTRY;
pub use method::JAVA_METHOD_CATALOG_ENTRY;
pub use method_edt::JAVA_METHOD_EDT_CATALOG_ENTRY;
pub use object::JAVA_OBJECT_CATALOG_ENTRY;
pub use object_edt::JAVA_OBJECT_EDT_CATALOG_ENTRY;
pub use remove_path::JAVA_REMOVE_PATH_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &ISJAVA_CATALOG_ENTRY,
    &JAVA_ADD_PATH_CATALOG_ENTRY,
    &JAVA_ARRAY_CATALOG_ENTRY,
    &JAVA_CLASSPATH_CATALOG_ENTRY,
    &JAVA_METHOD_CATALOG_ENTRY,
    &JAVA_METHOD_EDT_CATALOG_ENTRY,
    &JAVA_OBJECT_CATALOG_ENTRY,
    &JAVA_OBJECT_EDT_CATALOG_ENTRY,
    &JAVA_REMOVE_PATH_CATALOG_ENTRY,
    &JENV_CATALOG_ENTRY,
    &USEJAVA_CATALOG_ENTRY,
];
