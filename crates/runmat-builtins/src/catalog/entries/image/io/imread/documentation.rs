use crate::*;

const PPM: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId {
        local_name: "two-pixel-ppm",
    },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[
        BuiltinFilesystemEntry::File {
            relative_path: "pixels.dat",
            content: BuiltinFixtureContent::Utf8("P3\n2 1\n255\n255 0 0 0 255 0\n"),
        },
        BuiltinFilesystemEntry::File {
            relative_path: "pixels.ppm",
            content: BuiltinFixtureContent::Utf8("P3\n2 1\n255\n255 0 0 0 255 0\n"),
        },
    ],
});

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("imread"), slug: Some("imread"),
    summary: "Decode a local or HTTP(S) raster image into host numeric arrays.",
    description: "`imread(filename)` detects an image format from its bytes; `imread(filename, fmt)` uses an explicit format hint. Requested outputs select the image, empty direct-color colormap, and optional alpha plane.",
    keywords: &["imread", "image", "read image", "png", "jpeg", "jpg", "tiff", "bmp", "gif", "webp", "upload", "URL"],
    related: &["image", "imagesc", "class", "size", "gpuArray", "imwrite", "imfinfo"],
    sections: &[
        BuiltinDocumentationSection { heading: "Sources and formats", paragraphs: &[
            "The source and optional format are scalar text. Local paths and local `file://` URLs use the active filesystem; `http://` and `https://` URLs use the configured HTTP transport. A browser needs the corresponding filesystem bridge or network permission and may additionally be constrained by CORS. Unsupported URL schemes reject. Common explicit hints include png, jpg/jpeg, tif/tiff, bmp, gif, and webp; byte-stream detection is used when the hint is omitted. A format hint selects a decoder but does not search for a missing `filename.fmt` path.",
            "Reading can suspend and fail before decoding. The HTTP path has a finite request timeout and reports status, timeout, connection, and header failures distinctly. The file path retains an I/O cause. There is no provider kernel or fusion route; output remains host-owned, and numeric gpuArray arguments are not filenames.",
        ]},
        BuiltinDocumentationSection { heading: "Arrays and requested outputs", paragraphs: &[
            "Grayscale images return M-by-N; direct RGB/RGBA images return M-by-N-by-3 with column-major planes. The decoder preserves uint8 and uint16 samples and single floating-point RGB samples when exposed. Other decoder variants can be converted to uint8 RGBA before publication. Indexed colormap extraction and animated frame selection are not implemented: the current decoder returns direct pixel data for the decoded frame.",
            "`[I,map]` returns an empty double 0-by-0 map for direct decoded color. `[I,map,alpha]` returns an M-by-N alpha plane when present or an empty double 0-by-0 array otherwise. More than three requested outputs fail before file or network access. With no requested outputs, the call still performs its read/decode effects and discards the result.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "local-ppm", title: "Read a closed two-pixel image",
            program: "I = imread('pixels.ppm');\nsize(I)", display_output: Some("ans = [1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem, fixture: PPM,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isa(I,'uint8')); assert(isequal(size(I),[1 2 3])); assert(I(1,1,1)==255); assert(I(1,2,2)==255);" },
        },
        BuiltinExample {
            id: "explicit-format", title: "Decode bytes without a format extension",
            program: "[I,map] = imread('pixels.dat','pnm');\nsize(map)", display_output: Some("ans = [0 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem, fixture: PPM,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(I),[1 2 3])); assert(isempty(map));" },
        },
        BuiltinExample {
            id: "alpha", title: "Read a separately returned alpha plane",
            program: "rgba = reshape(uint8([1 2 3 4]),[1 1 4]); imwrite(rgba,'alpha.png'); [rgb,map,alpha] = imread('alpha.png');\nsize(alpha)", display_output: Some("ans = [1 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(rgb),[1 1 3])); assert(isempty(map)); assert(isequal(alpha,uint8(4)));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What shape does imread return for RGB images?", answer: "RGB images return an M-by-N-by-3 tensor with red, green, and blue planes in RunMat's column-major layout." },
        BuiltinDocumentationFaq { question: "Where is the alpha channel returned?", answer: "Use `[I,map,alpha] = imread(filename)`. For direct color, map is empty; alpha is M-by-N when present and empty otherwise." },
        BuiltinDocumentationFaq { question: "Can imread load file:// or HTTP URLs?", answer: "Yes, through the configured filesystem or HTTP transport. Browser filesystem bridges, network permission, and CORS policy may limit availability; unsupported schemes fail." },
        BuiltinDocumentationFaq { question: "Does imread return gpuArray data?", answer: "No. Decoding publishes host tensors; use gpuArray afterward if device residency is needed. A resident numeric value is not a filename." },
        BuiltinDocumentationFaq { question: "Are animated GIF frames supported?", answer: "Frame selection is not supported; the decoder returns the first decoded image frame as direct pixel data." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "image", target: BuiltinDocumentationLinkTarget::Builtin("image") },
        BuiltinDocumentationLink { label: "imagesc", target: BuiltinDocumentationLinkTarget::Builtin("imagesc") },
        BuiltinDocumentationLink { label: "class", target: BuiltinDocumentationLinkTarget::Builtin("class") },
        BuiltinDocumentationLink { label: "imwrite", target: BuiltinDocumentationLinkTarget::Builtin("imwrite") },
        BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Image reader and decoder", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/imread/decode.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Local, URL, multi-output and precision behavior", location: "crates/runmat-runtime/src/builtins/image/imread/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: None,
};
