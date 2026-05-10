pub mod class_def;
pub mod member_read;
pub mod member_write;
pub mod method_call;
pub mod resolve;
pub mod static_dispatch;

pub(crate) const CLASS_REF_CONSTRUCTOR_NAME: &str = "classref";
pub(crate) const PAREN_SELECTOR_NAME: &str = "()";
pub(crate) const BRACE_SELECTOR_NAME: &str = "{}";
pub(crate) const MEMBER_SELECTOR_NAME: &str = ".";
