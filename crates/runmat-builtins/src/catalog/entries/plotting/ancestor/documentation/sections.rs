use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Syntax",
        paragraphs: &[
            "p = ancestor(h, type)",
            "p = ancestor(h, type, 'toplevel')",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`type` can be a character vector, string scalar, string array, or cell array of type names.",
            "`ancestor(h, type)` returns the nearest matching object, including `h` itself when its type matches.",
            "`ancestor(h, type, 'toplevel')` returns the highest matching object among the requested types.",
            "Invalid or non-graphics scalar handles return an empty handle array instead of throwing.",
            "RunMat graphics handle arrays use an opaque host-double representation and are processed elementwise. Integer scalars and arrays are ordinary numeric data, not graphics handles, and return an empty handle array without exposing the internal encoding.",
            "Supported graphics types include `root`, `figure`, `axes`, `numericruler`, `text`, `legend`, and RunMat plot child types such as `line`, `scatter`, `bar`, `surface`, `patch`, and `contour`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`ancestor` inspects host-side graphics handle metadata and does not execute numeric GPU kernels.",
            "`ancestor` only reads host-managed graphics handle metadata. It does not inspect, gather, or transform GPU-resident plot data.",
            "`ancestor` performs no provider dispatch and does not gather or upload GPU arrays.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
            "The current implementation builds a typed parent chain, including the input object, and matches canonical plot-child class identities. Public type-name text is decoded only at the selector boundary. Unknown nonempty names simply produce no match. Nonscalar successful results are filtered host-double rows; unmatched positions are omitted, not replaced by placeholders.",
        ],
    },
];
