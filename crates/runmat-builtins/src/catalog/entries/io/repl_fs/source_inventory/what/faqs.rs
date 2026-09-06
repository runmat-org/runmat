use crate::BuiltinDocumentationFaq;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which files does what report?", answer: "The returned structure groups `.m` source files, `.mat` data files, platform MEX extensions, `@` class folders, and `+` package folders found directly in the selected folder." },
    BuiltinDocumentationFaq { question: "Does what search subfolders?", answer: "No. It summarizes direct children of the current or named folder. Use path and source-discovery tools when you need recursive lookup." },
    BuiltinDocumentationFaq { question: "Can what inspect a gpuArray path?", answer: "No. The optional folder is host text. Numeric and provider-resident inputs reject before filesystem or accelerator-provider access." },
];
