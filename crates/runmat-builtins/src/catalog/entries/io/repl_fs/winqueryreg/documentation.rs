use crate::*;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("winqueryreg"),
    slug: Some("winqueryreg"),
    summary: "Query values from the Windows registry when running on Windows.",
    description: "`winqueryreg` reads a named value or lists value names from the Windows registry.",
    keywords: &[
        "winqueryreg"
    ],
    related: &[
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Registry root, key, and value-name selectors must be host string scalars or character row vectors.",
                "`REG_SZ` and `REG_EXPAND_SZ` values return text. `REG_DWORD` values return an exact `int32`, including values whose bit pattern represents a negative `int32`.",
                "The leading `name` form lists value names under a key and returns them as text entries.",
                "Native integer and resident numeric selectors reject before provider or registry access.",
                "The builtin reports that the Windows registry is unavailable when called on another operating system."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Host and accelerator behavior",
            paragraphs: &[
                "Registry access is host-side on Windows. Numeric and resident selectors are rejected without provider access."
            ]
        },
        BuiltinDocumentationSection {
            heading: "Authority, resource policy, and limits",
            paragraphs: &[
                "The omitted or empty value-name form requests the default value with reg.exe /ve. A named value uses /v with the exact selector as a separate argument; the name-list form queries the key without /v or /ve.",
                "REG_SZ and REG_EXPAND_SZ retain payload whitespace after the reg.exe column delimiter. REG_DWORD parses its 32-bit payload without conversion through double; all 32 bits are preserved in the returned int32.",
                "The selected RuntimeContext process service executes one exact HostCommand through the shared ProcessInvocation owner. Its selected environment, native working-directory authority, cancellation, and pipe ownership are retained. There is no ambient std::process fallback.",
                "A finite host-memory policy rejects complete unbounded process output before child creation. A missing service or stale directory authority reports its typed cause instead of silently using another host.",
                "The portable examples exercise deterministic admission failures. Native Windows successful registry queries and localized reg.exe output still require dedicated qualification; those examples do not certify successful registry access."
            ]
        }
    ],
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Why can a REG_DWORD result be negative?",
            answer: "The 32-bit registry payload is returned with the documented `int32` class, so its bit pattern is interpreted as a signed 32-bit integer."
        }
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MATLAB winqueryreg reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/winqueryreg.html")
        }
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime winqueryreg retained admission",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/compat.rs")
            }
        ],
        verification: &[
        ],
        notes: &[
            "Source migration only: listed legacy test locations and independently asserted standalone examples are evidence pointers, not claims of fresh execution. Central integration must run focused canonical/runtime/Core and example qualification.",
            "Contract maturity remains Incomplete and documentation status Partial. Host-only placement and fusion boundaries do not certify full MATLAB compatibility, browser execution, positive Windows registry qualification, or finite resource completion."
        ]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
