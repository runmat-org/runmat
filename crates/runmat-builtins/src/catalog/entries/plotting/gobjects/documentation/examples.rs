use crate::*;
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Preallocate an array and fill one graphics handle",
        program: "h = gobjects(2,1);\nh(1) = plot(1:3);\nisgraphics(h)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(h),[2 1])); assert(isgraphics(h(1))); assert(~isgraphics(h(2)));" },
    },
    BuiltinExample {
        id: "example-2",
        title: "Use a size vector",
        program: "h = gobjects([1 3]);\nnumel(h)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(h),[1 3])); assert(numel(h)==3); assert(all(isnan(h)));" },
    },
];
