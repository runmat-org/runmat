use crate::*;

const PPM: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId {
        local_name: "two-pixel-ppm",
    },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File {
        relative_path: "two_pixels.ppm",
        content: BuiltinFixtureContent::Utf8("P3\n2 1\n255\n255 0 0 0 255 0\n"),
    }],
});

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("imfinfo"), slug: Some("imfinfo"),
    summary: "Read image file format, dimensions, color type, and byte-size metadata.",
    description: "`imfinfo(filename)` reads an image through the active filesystem and returns a host metadata struct. `imfinfo(filename, fmt)` uses a supported format hint and, if the named file is missing, also tries `filename.fmt`.",
    keywords: &["imfinfo", "image", "metadata", "width", "height", "format"],
    related: &["imread", "imwrite"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and result", paragraphs: &[
            "The filename and optional format are scalar text. An explicit format hint must name a format recognized by the image decoder. If the original filename is not found, a supported hint also tries the path with that extension appended. The result is a scalar struct with `Filename`, `FileModDate`, `FileSize`, `Format`, `FormatVersion`, `Width`, `Height`, `BitDepth`, and `ColorType`. The two version/date fields are currently empty character rows; byte counts and dimensions are double-valued metadata.",
            "The decoder reads the image to establish metadata. This is not a header-only query; malformed image bytes can fail. No numeric image array or resident provider handle is accepted as a path or format.",
        ]},
        BuiltinDocumentationSection { heading: "Filesystem and portability", paragraphs: &[
            "The call may suspend while reading the active RunMat filesystem and can fail for invalid text, unsupported format, I/O, or decode errors. `~` path expansion consults the environment. The result is host-owned and is a fusion boundary; browser use requires an available filesystem bridge and compatible image decoder. A path names the caller's filesystem, not a transferable file snapshot, so distributed workers need a separately staged file and local invocation; this call is not automatically retried elsewhere.",
        ]},
    ],
    examples: &[BuiltinExample {
        id: "ppm-metadata", title: "Inspect a two-pixel PPM image",
        program: "info = imfinfo('two_pixels.ppm');\n[info.Width info.Height]",
        display_output: Some("ans = [2 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: PPM, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(info)); assert(info.Width == 2); assert(info.Height == 1); assert(info.FileSize > 0); assert(strcmp(info.Format,'pnm'));" },
    }, BuiltinExample {
        id: "format-extension-fallback", title: "Find a file by its format extension",
        program: "info = imfinfo('two_pixels','ppm');\ninfo.Width",
        display_output: Some("ans = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: PPM, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(info.Width == 2); assert(info.Height == 1); assert(strcmp(info.Format,'pnm'));" },
    }],
    example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "imread", target: BuiltinDocumentationLinkTarget::Builtin("imread") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Metadata reader", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/imfinfo.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Metadata, argument and output effects", location: "crates/runmat-runtime/src/builtins/image/imfinfo.rs" }],
        notes: &[],
    },
    introduced: None, status: None,
};
