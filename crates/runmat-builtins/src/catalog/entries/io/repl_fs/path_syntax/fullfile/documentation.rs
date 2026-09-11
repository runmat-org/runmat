use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Path assembly", paragraphs: &["`fullfile(filepart1, ..., filepartN)` joins path components with the separator for the current platform. It performs lexical path assembly and does not require the result to exist.", "Character rows produce a character row. If any input is a string array, the result is a string array. Otherwise, a cell-array input produces a cell array of character rows. Scalar components expand across a nonscalar container; nonscalar containers must have the same shape."] },
    BuiltinDocumentationSection { heading: "Separators and execution", paragraphs: &["On Windows, forward slashes are normalized to backslashes. Repeated inner separators and nonterminal `.` components are collapsed; leading and trailing separators are retained where they describe the requested path.", "Path assembly is host-owned and is not accelerated. RunMat compatibility mode additionally accepts dense real numeric character-code rows, including an eligible provider-resident row gathered through its owner."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "relative", title: "Join relative path components", program: "p = fullfile('data', 'raw', 'sample.dat');", display_output: Some("A platform-correct path ending in sample.dat"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(p)); assert(endsWith(p, 'sample.dat')); assert(contains(p, 'data')); assert(contains(p, 'raw'));" } },
    BuiltinExample { id: "current-folder", title: "Build below the current folder", program: "config = fullfile(pwd(), 'config', 'settings.json');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(startsWith(config, pwd())); assert(endsWith(config, 'settings.json'));" } },
    BuiltinExample { id: "string-array", title: "Build several paths with a string array", program: "names = [\"first.m\" \"second.m\"];\npaths = fullfile(\"src\", names);", display_output: Some("paths is a 1-by-2 string array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(paths)); assert(isequal(size(paths), [1 2])); assert(endsWith(paths(1), \"first.m\")); assert(endsWith(paths(2), \"second.m\"));" } },
    BuiltinExample { id: "cell-array", title: "Build paths from a cell array", program: "names = {'first.m'; 'second.m'};\npaths = fullfile('src', names);", display_output: Some("paths is a 2-by-1 cell array of character rows"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(paths)); assert(isequal(size(paths), [2 1])); assert(endsWith(paths{1}, 'first.m')); assert(endsWith(paths{2}, 'second.m'));" } },
    BuiltinExample { id: "trailing-separator", title: "Retain a requested trailing separator", program: "folder = fullfile('data', 'raw', filesep());", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(endsWith(folder, filesep()));" } },
    BuiltinExample { id: "numeric-codes", title: "Use numeric character codes in RunMat mode", program: "p = fullfile(uint8([100 97 116 97]), 'sample.dat');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(p, 'data')); assert(endsWith(p, 'sample.dat'));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does fullfile create a folder or file?", answer: "No. It only assembles path text." },
    BuiltinDocumentationFaq { question: "How does fullfile choose its output representation?", answer: "A string input selects a string result; otherwise a cell input selects a cell result; otherwise the result is a character row." },
    BuiltinDocumentationFaq { question: "Does fullfile expand a leading tilde?", answer: "No. It does not resolve or inspect the path." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "fileparts", target: BuiltinDocumentationLinkTarget::Builtin("fileparts") },
    BuiltinDocumentationLink { label: "filesep", target: BuiltinDocumentationLinkTarget::Builtin("filesep") },
    BuiltinDocumentationLink { label: "pathsep", target: BuiltinDocumentationLinkTarget::Builtin("pathsep") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/fullfile/mod.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fullfile"),
    slug: Some("fullfile"),
    summary: "Build platform-correct paths from character, string, or cell components.",
    description: "`fullfile` joins one or more path components and preserves MATLAB-compatible text-container representation and shape.",
    keywords: &["fullfile", "join paths", "path assembly", "filesystem", "filesep"],
    related: &["fileparts", "filesep", "pathsep", "pwd", "tempdir", "mkdir"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Lexical path runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/path_syntax/fullfile/mod.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Representation, shape, separator, extension, and rejection behavior", location: "builtins::io::repl_fs::path_syntax::fullfile::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable native and browser examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
