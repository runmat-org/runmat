use crate::BuiltinDocumentationFaq;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which path values can I pass to dir?", answer: "Use a character vector or string scalar containing a local file, folder, or wildcard path. Numeric values, cells, character matrices, and nonscalar string arrays are rejected before filesystem access." },
    BuiltinDocumentationFaq { question: "Why do folder listings contain . and ..?", answer: "The compatible folder-listing form includes the current-folder and parent-folder entries. Wildcard results contain only entries matched by the pattern." },
    BuiltinDocumentationFaq { question: "Does dir support recursive wildcards?", answer: "The `**` form traverses subfolders when it appears as a complete path component. Ordinary `*` and `?` patterns match within one directory level." },
    BuiltinDocumentationFaq { question: "Does dir support remote URLs?", answer: "Not yet. This implementation reads through RunMat's native or browser filesystem service and currently resolves local filesystem paths only." },
    BuiltinDocumentationFaq { question: "Can dir inspect a gpuArray path?", answer: "No. Paths are host text. Numeric and provider-resident values are rejected without an implicit device transfer." },
];
