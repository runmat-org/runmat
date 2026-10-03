use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("imwrite"),
    slug: Some("imwrite"),
    summary: "Encode an image array and write PNG, JPEG, GIF, BMP, or TIFF bytes.",
    description: "`imwrite(A,filename)` writes direct grayscale or color data. An optional explicit format, indexed colormap, and format-specific name-value controls select other supported forms.",
    keywords: &["imwrite", "image", "PNG", "JPEG", "GIF", "BMP", "TIFF", "indexed image"],
    related: &["imread", "imfinfo", "gpuArray"],
    sections: &[
        BuiltinDocumentationSection { heading: "Data and formats", paragraphs: &[
            "Direct images may be M-by-N grayscale, M-by-N-by-3 RGB, or M-by-N-by-4 RGBA. Logical, double, single, uint8, and uint16 image data are accepted. Floating-point samples are interpreted in [0,1] and clamped when encoded; uint8 retains its byte scale. uint16 is preserved by supported PNG/TIFF encoders and projected to eight bits for other formats. Other integer image classes reject before writing.",
            "`imwrite(X,map,filename,...)` uses an Nx3 finite double colormap with entries in [0,1]. Floating indices are one-based; uint8 and uint16 indices are zero-based. The filename is scalar text; the extension chooses PNG, JPEG, GIF, BMP, or TIFF unless a supported explicit format is supplied.",
            "A resident numeric image or colormap is downloaded non-destructively through its owning provider before host encoding. No provider kernel or fusion route writes image files. Input class and owner metadata are checked before readback; decoded bytes are not sent back to the provider.",
        ]},
        BuiltinDocumentationSection { heading: "Options and effects", paragraphs: &[
            "`Alpha` supplies an M-by-N double, single, uint8, or uint16 plane for formats that support alpha. JPEG `Quality` is 0 to 100 (default 75). GIF `DelayTime` is a non-negative number of seconds; `LoopCount` is non-negative or Inf. `WriteMode` is `overwrite` or `append`; append is supported only for GIF and reads existing frames before rewriting the destination.",
            "Unsupported advanced controls such as Compression and BitDepth fail explicitly. Image, option, and encoder validation occurs before the final write, but a filesystem write failure can leave a partial destination. GIF append is read-modify-write without a portable file lock, so concurrent writers must coordinate externally. This call returns no outputs and does not print an automatic result.",
            "Direct single-precision GIF and TIFF writes are a RunMat extension and require that compatibility extensions be enabled. The destination uses the active filesystem service. Native and browser runtimes need a writable filesystem or browser filesystem bridge. A local destination is not implicitly transferable to a remote worker; run the effect where its filesystem is available.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "rgb-png", title: "Write and read a small RGB PNG",
            program: "rgb = uint8(zeros(2,3,3)); rgb(:,:,1) = 255; imwrite(rgb,'red.png'); reread = imread('red.png');\nsize(reread)",
            display_output: Some("ans = [2 3 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(reread,'uint8')); assert(isequal(size(reread),[2 3 3])); assert(reread(1,1,1)==255); assert(reread(1,1,2)==0);",
            },
        },
        BuiltinExample {
            id: "explicit-jpeg-quality", title: "Choose JPEG and its quality",
            program: "rgb = uint8(zeros(2,3,3)); imwrite(rgb,'quality.dat','jpg','Quality',90); info = imfinfo('quality.dat','jpg');\ninfo.Width",
            display_output: Some("ans = 3"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(info.Width==3); assert(info.Height==2); assert(info.FileSize>0);",
            },
        },
        BuiltinExample {
            id: "indexed-gif", title: "Write zero-based indexed data",
            program: "X = uint8([0 1]); map = [1 0 0; 0 1 0]; imwrite(X,map,'indexed.gif'); info = imfinfo('indexed.gif');\ninfo.Width",
            display_output: Some("ans = 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(info.Width==2); assert(info.Height==1); assert(info.FileSize>0);",
            },
        },
        BuiltinExample {
            id: "gif-append", title: "Append a second GIF frame",
            program: "frame1 = uint8([0 255]); frame2 = uint8([255 0]); imwrite(frame1,'motion.gif','LoopCount',Inf,'DelayTime',0.2); imwrite(frame2,'motion.gif','WriteMode','append','DelayTime',0.2); info = imfinfo('motion.gif');\ninfo.Width",
            display_output: Some("ans = 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(info.Width==2); assert(info.Height==1); assert(info.FileSize>0);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can imwrite use gpuArray inputs?", answer: "Yes. Numeric resident operands are downloaded through their owning provider for the host encoder; the caller's resident value remains resident." },
        BuiltinDocumentationFaq { question: "Is GIF append atomic?", answer: "No. It reads existing frames and rewrites the GIF without a portable lock." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "imread", target: BuiltinDocumentationLinkTarget::Builtin("imread") },
        BuiltinDocumentationLink { label: "imfinfo", target: BuiltinDocumentationLinkTarget::Builtin("imfinfo") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Image writer and encoders", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/imwrite.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Image classes, options, GIF append, and resident owner behavior", location: "crates/runmat-runtime/src/builtins/image/imwrite.rs" }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
