use super::*;

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "MAT object methods and whole-variable protocol", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/matfile.rs") },
    BuiltinDocumentationLink { label: "MAT receiver and Writable option admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/matfile/receiver.rs") },
    BuiltinDocumentationLink { label: "MAT member read and write steps", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/matfile/steps.rs") },
    BuiltinDocumentationLink { label: "Sealed receiver and filesystem owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/authority/handle.rs") },
    BuiltinDocumentationLink { label: "Shared MAT value conversion and writer", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/encoding.rs") },
    BuiltinDocumentationLink { label: "MAT value conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/encoding/value.rs") },
];
const MAT_LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "load",
        target: BuiltinDocumentationLinkTarget::Builtin("load"),
    },
    BuiltinDocumentationLink {
        label: "save",
        target: BuiltinDocumentationLinkTarget::Builtin("save"),
    },
];

pub(in super::super) const MATFILE: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matfile"),
    slug: Some("matfile"),
    summary: "Create a Level-5 MAT-file whole-variable compatibility object.",
    description: "`matfile(filename)` returns a `matlab.io.MatFile` object. `m.Properties` reports `Source` and `Writable`; `m.A` reads a complete variable. Construct with `matfile(filename, 'Writable', true)` to enable whole-variable dot assignments.",
    keywords: &["matfile", "MAT-file", "Writable", "whole variable"],
    related: &["load", "save"],
    sections: &[
        BuiltinDocumentationSection { heading: "Construction and ownership", paragraphs: &[
            "The filename is scalar character or string text and gains `.mat` when no extension is present. `Writable` accepts a scalar logical or numeric value; omitted means false. Complete repeated `Writable` pairs are accepted and the last value wins. The object retains the selected filesystem provider and resolved source path across later calls, even if the ambient provider or working folder changes. Public `Source` and `Writable` properties report state but do not authorize or retarget I/O.",
        ] },
        BuiltinDocumentationSection { heading: "Reads and writes", paragraphs: &[
            "Dot reads decode a complete variable with the same Level-5 classes and host-resident results as `load`. Writable dot assignment reads existing variables, replaces or appends a whole value, and writes the refreshed Level-5 file. A read-only object rejects assignment.",
            "Partial reads/writes that avoid materializing a whole variable, HDF5-backed v7.3 MAT files, and `who(m)`/`whos(m)` are not yet supported. The object is session-bound in runtime calls and cannot be forged from public class or property text.",
        ] },
    ],
    examples: &[
        example("read", "Read a whole variable", "A = [1 2; 3 4];\nsave('data.mat', 'A');\nm = matfile('data.mat');\nloaded = m.A", Some("A contains the variable loaded from data.mat."), "assert(isequal(loaded, A)); assert(~m.Properties.Writable);"),
        example("write", "Write whole variables", "m = matfile('data.mat', 'Writable', true);\nm.A = ones(3);\nm.flag = true;\nS = load('data.mat')", Some("The MAT-file is rewritten with updated whole variables."), "assert(isequal(S.A, ones(3))); assert(S.flag); assert(m.Properties.Writable);"),
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `matfile` read part of a variable?", answer: "No. The current Level-5 object implementation loads and writes complete variables; partial indexed I/O is not implemented." },
        BuiltinDocumentationFaq { question: "Can a read-only object write?", answer: "No. Construct with the `Writable` name-value pair set to a truthy scalar logical or numeric value." },
        BuiltinDocumentationFaq { question: "What are `m.Properties` fields?", answer: "`Source` keeps the normalized caller path spelling (adding `.mat` when needed), while `Writable` reports the selected Boolean mode. The private I/O owner separately retains the captured absolute path." },
    ],
    links: MAT_LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: IMPLEMENTATION,
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Whole-variable object reads and writes", location: "builtins::io::mat::matfile::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(in super::super) const SUBSREF: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.io.MatFile.subsref"),
    slug: Some("matlab.io.MatFile.subsref"),
    summary: "Read a MAT-file property or complete variable.",
    description: "`matlab.io.MatFile.subsref(m, S)` is the object protocol behind `m.Properties` and `m.A`. The receiver must be a live, sealed `matlab.io.MatFile` object; `S` is a standard substruct member path.",
    keywords: &["matfile", "subsref", "MAT-file", "variable read"],
    related: &["matfile", "load"],
    sections: &[BuiltinDocumentationSection { heading: "Whole-member reads", paragraphs: &[
        "`Properties` returns a struct with `Source` and `Writable`. Another member name opens the captured provider's Level-5 file, restores complete variables and object `loadobj` callbacks, then selects that name. Unsupported partial indexing and missing variables raise errors; returned arrays are host-resident.",
    ] }],
    examples: &[example("member", "Read a complete MAT variable", "A = [2 4 6];\nsave('member.mat', 'A');\nm = matfile('member.mat');\nvalue = m.A", None, "assert(isequal(value, A));")],
    example_exemption: None,
    faqs: &[],
    links: MAT_LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "MAT subsref receiver and selected values", location: "builtins::io::mat::matfile::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

pub(in super::super) const SUBSASGN: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matlab.io.MatFile.subsasgn"),
    slug: Some("matlab.io.MatFile.subsasgn"),
    summary: "Write one complete variable through a writable MAT-file object.",
    description: "`matlab.io.MatFile.subsasgn(m, S, rhs)` is the object protocol behind whole-variable `m.A = rhs`. The sealed receiver must be writable, and `S` must designate one valid variable member.",
    keywords: &["matfile", "subsasgn", "MAT-file", "variable write"],
    related: &["matfile", "save"],
    sections: &[BuiltinDocumentationSection { heading: "Whole-member writes", paragraphs: &[
        "The operation reads existing Level-5 variables through the object's captured provider, gathers resident rhs data from its owner, encodes a replacement MAT file, and writes it through the same captured provider. A same-name variable is replaced; a new name is appended. Read-only or partial assignments reject before publication. The provider write is not a rollback transaction.",
    ] }],
    examples: &[example("member", "Assign a complete MAT variable", "m = matfile('assignment.mat', 'Writable', true);\nm.A = [5 6];\nS = load('assignment.mat')", None, "assert(isequal(S.A, [5 6]));")],
    example_exemption: None,
    faqs: &[],
    links: MAT_LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "MAT subsasgn exact replacement", location: "builtins::io::mat::matfile::tests" }], notes: &[] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
