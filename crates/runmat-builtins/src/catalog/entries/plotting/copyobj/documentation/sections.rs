use crate::*;
pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`copyobj(h, parent)` copies each plot-child handle in `h` to a compatible axes parent. Plot children such as lines do not accept a figure or panel as their destination parent.",
            "Copied objects are inserted into the destination axes identified by `parent`.",
            "A scalar parent is applied to every source handle; equal-length source and parent vectors copy each source into the corresponding parent regardless of row/column orientation; a scalar source can be copied into each parent in a parent array. Existing RunMat matrix and N-D handle arrays pair in linear storage order with equal element counts, without claiming MATLAB matrix-form parity.",
            "Handle arrays preserve the controlling input shape and receive fresh handle values: source shape controls when the parent is scalar or both inputs are nonscalar; otherwise parent shape controls. Copying the same object more than once to the same parent in one call rejects before mutation.",
            "Because RunMat currently encodes graphics handles numerically, compatibility mode can separately enable exact host typed-integer handle aliases for all eight integer classes. Negative and nonexact values reject before lookup; exactly representable aligned wide integers proceed to ordinary handle lookup. Resident integer arrays always reject without provider gather.",
            "RunMat validates the whole source/parent copy plan before mutating graphics state, so invalid handle arrays do not leave partial copies behind.",
            "Copied plot children preserve their plotted geometry and readable/writable plot-child properties such as `Type`, `Parent`, data vectors, line styling, and display names.",
            "Figure, axes, text, legend, ruler, root-object, UI-container, and composite-chart deep copies are rejected with `RunMat:copyobj:InvalidArgument` rather than silently returning shallow placeholders.",
            "The optional `copyobj(obj, parent, 'legacy')` callback/application-data form is not implemented. Callback, application-data, context-menu, toolbar, axes-interaction, and data-tip-template copy rules remain general object-copy gaps.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and residency",
        paragraphs: &[
            "`copyobj` clones host-managed graphics state and does not execute numeric GPU kernels.",
            "`copyobj` returns host-side numeric graphics handles. It does not gather, transform, or upload GPU arrays.",
            "`copyobj` performs no provider dispatch because graphics object copying is a host graphics-registry operation.",
            "When copied plot elements were originally built from GPU-resident input data, the existing plot-state representation is cloned; no additional CPU/GPU transfer is initiated by `copyobj` itself.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution, authority, and current limits",
        paragraphs: &[
            "The canonical binding uses the same runtime implementation in interpreter and compiled execution. Requested output arity is checked before this operation can mutate graphics state or gather coordinate data; discarding an output does not suppress the operation.",
            "Graphics handles encode local registry identities, not transferable figure, axes, or chart state. Copying a numeric handle to a worker, foreign runtime, or serialized value does not transfer its graphics authority. This host operation is a fusion boundary; no numeric GPU kernel or distributed implementation is supplied.",
            "Finite Managed allocation remains an L25 obligation: complete scratch, metadata, retained graphics-state, and output ownership are not yet quoted by these implementations. The shared graphics admission owner refuses Managed policy before unowned work, captures cancellation authority, and checks it before publication; that safety fence is not finite-allocation completion. Unbounded host scratch may still contain synchronous loops.",
            "Only supported plot children can be copied to axes. Plan validation precedes copying, but resource failure or external authority rotation during a multi-copy is terminal and is not a transaction rollback of already committed copies. Retained callback metadata is copied through its existing owner; this operation does not invoke user callback code.",
        ],
    },
];
