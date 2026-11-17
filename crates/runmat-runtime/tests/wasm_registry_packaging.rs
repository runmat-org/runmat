// Exercise the actual build-script inventory in the ordinary integration suite.
// The test fixture verifies that packaging out unit tests preserves the registry
// identity while production changes still invalidate it.
include!("../build.rs");
