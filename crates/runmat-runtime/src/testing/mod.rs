mod class_registry;
mod context;
mod objects;
mod services;

pub use class_registry::ensure_testing_classes;
pub use context::{
    active_test_context, install_test_context, record_runtime_teardown, record_test_command,
    ActiveTestContext, RuntimeTeardownInvocation, TestContextGuard, TestContextHandle,
};
pub use objects::{
    diagnostic_object, fixture_object, object_array_or_scalar, plugin_object, test_case_object,
    test_result_object, test_suite_object, TEST_CASE_CLASS, TEST_RESULT_CLASS, TEST_SUITE_CLASS,
};
pub use services::{install_test_services, run_test_suite, RuntimeTestServices, TestServicesGuard};
