use crate::BuiltinDocumentationFaq;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which path values can I pass to ls?", answer: "Use a character vector or string scalar. Numeric values, cells, character matrices, and nonscalar string arrays are rejected before filesystem access." },
    BuiltinDocumentationFaq { question: "How does ls represent directories?", answer: "A directory name ends with the platform's file separator in RunMat's portable row layout." },
    BuiltinDocumentationFaq { question: "How does the output differ by compatibility mode?", answer: "RunMat mode uses one padded character row per entry on every platform. MATLAB compatibility mode follows the platform-specific MATLAB layout: a separated character vector on Unix-like systems and padded rows on Windows." },
    BuiltinDocumentationFaq { question: "Does ls support operating-system flags?", answer: "Not currently. The input is interpreted as a file, folder, or wildcard name through RunMat's filesystem service rather than passed to an operating-system shell." },
    BuiltinDocumentationFaq { question: "Does ls support remote URLs?", answer: "No. The current implementation resolves local paths in the active native or browser filesystem." },
];
