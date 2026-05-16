pub(crate) mod class_def;
pub(crate) mod member_read;
pub(crate) mod member_write;
pub(crate) mod method_call;
pub(crate) mod resolve;
pub(crate) mod static_dispatch;

pub(crate) const PAREN_SELECTOR_NAME: &str = "()";
pub(crate) const BRACE_SELECTOR_NAME: &str = "{}";
pub(crate) const MEMBER_SELECTOR_NAME: &str = ".";
