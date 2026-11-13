use crate::*;

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "refresh-path",
    title: "Refresh the active source search cache",
    program: "status = rehash('path');\nassert(status == 0);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(status == 0);",
    },
}];

pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rehash"),
    slug: Some("rehash"),
    summary: "Refresh the active session's source-function lookup cache.",
    description: "`rehash` asks the active execution session to invalidate its source-function cache and advance its search-path revision. Accepted modes are `path`, `toolbox`, and `toolboxcache`. RunMat currently has one live source cache rather than separate MATLAB toolbox and persisted toolbox-cache inventories.",
    keywords: &["rehash", "path", "functions", "cache"],
    related: &["path", "addpath", "clear"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Modes",
            paragraphs: &[
                "`rehash` and `rehash('path')` refresh the active source lookup cache. `rehash('toolbox')` and `rehash('toolboxcache')` refresh the same RunMat source inventory; there is no separate persisted toolbox cache file to update.",
                "Refresh does not clear workspace values or unload native extension modules. The source cache belongs to the active execution session; an embedding without an installed refresh service reports an unavailable-service error instead of silently succeeding.",
                "RunMat permits an explicit zero status output. At most one result may be requested; malformed modes reject before cache invalidation.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Current limits",
            paragraphs: &[
                "Immutable packaged/project source catalogs, already compiled artifacts, and live class objects retain their own revisions. A full MATLAB-style toolbox file inventory and persisted toolbox cache are not yet modeled.",
            ],
        },
    ],
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does rehash clear variables or unload native modules?",
        answer: "No. It refreshes source-function lookup state in the active session, leaving workspace values and native modules intact.",
    }],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB rehash reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/rehash.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime mode preparation",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs"),
            },
            BuiltinDocumentationLink {
                label: "Core source-cache refresh owner",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-core/src/session/run.rs"),
            },
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Changed path function resolves after rehash without clearing workspace",
            location: "runmat_core::tests::rehash_path_invalidates_session_source_cache_without_clearing_workspace",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
