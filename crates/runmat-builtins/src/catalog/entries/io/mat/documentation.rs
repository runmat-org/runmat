use crate::*;

mod load;
mod matfile;
mod save;

pub(super) use load::LOAD;
pub(super) use matfile::{MATFILE, SUBSASGN, SUBSREF};
pub(super) use save::SAVE;

const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: Option<&'static str>,
    assertion: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertion },
    }
}
