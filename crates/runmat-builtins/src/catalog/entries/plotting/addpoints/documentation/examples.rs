use crate::*;
pub(super) const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Append samples to an animated line",
        program: "an = animatedline('MaximumNumPoints', 3);\naddpoints(an, [1 2], [10 20]);\naddpoints(an, [3 4], [30 40]);\nget(an, 'YData')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(get(an,'YData'),[20 30 40])); assert(strcmp(get(an,'Type'),'animatedline'));" },
    },
    BuiltinExample {
        id: "example-2",
        title: "Append 3-D samples",
        program: "an = animatedline();\naddpoints(an, [1 2], [3 4], [5 6]);\nget(an, 'Type')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(get(an,'XData'),[1 2])); assert(isequal(get(an,'YData'),[3 4])); assert(isequal(get(an,'ZData'),[5 6]));" },
    },
];
