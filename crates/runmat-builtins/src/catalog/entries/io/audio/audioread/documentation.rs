use crate::*;

const WAV_FILE: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
    relative_path: "speech.wav",
    content: BuiltinFixtureContent::Bytes(super::super::fixtures::MONO_WAV),
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("audioread"),
    slug: Some("audioread"),
    summary: "Read audio samples from a file.",
    description: "`audioread(filename)` decodes an N-by-C sample tensor, where rows are frames and columns are channels. `[y, Fs] = audioread(filename)` also returns the double sample rate in Hz.",
    keywords: &["audioread", "audio", "read audio", "wav", "rf64", "pcm", "float", "sample rate", "channels"],
    related: &["audioinfo", "gpuArray"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and range", paragraphs: &[
            "Filenames are scalar strings, singleton string arrays, or one-row character arrays read through the active filesystem provider. The optional two-element double sample range is one-based and inclusive: `audioread(filename, [first last])`. `Inf` as the last endpoint reads through the final frame; typed-integer ranges reject.",
            "The datatype argument is `double` (default) or `native`. It may follow the filename directly or a sample range. At most two outputs are available; a zero-output statement still reads and decodes.",
        ] },
        BuiltinDocumentationSection { heading: "Decoding and classes", paragraphs: &[
            "WAV/RF64 PCM 8-bit unsigned, PCM 16/24/32-bit signed, IEEE 32/64-bit float, and WAVE_FORMAT_EXTENSIBLE PCM/float subformats are decoded. FLAC, MP3, Ogg/Vorbis, and AIFF metadata can be inspected by `audioinfo`, but their samples are not decoded here.",
            "Default PCM output is normalized double: unsigned 8-bit maps around zero and signed PCM divides by signed full scale. Native integer storage follows the sample container width: uint8 for 8-bit unsigned PCM, int16 for 16-bit signed PCM, and int32 for 24/32-bit signed PCM; 24-bit containers are left-aligned in int32. Extensible PCM may have fewer valid bits left-aligned inside a wider container, such as 24 valid bits in a 32-bit int32 sample. Native IEEE float32 returns single and float64 returns double. Sample tensors are column-major N-by-C, including N-by-1 mono.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Filesystem I/O and decoding execute on the host, terminate fusion, and create host-resident tensors. No provider kernel runs; GPU-resident arguments reject before provider access. Use `gpuArray` explicitly after reading if subsequent computation should reside on a device.",
            "The active native, browser, sandbox, or remote filesystem provider must supply the input bytes. A filename is not a transferable file snapshot, so distributed workers require their own accessible file and filesystem capability.",
            "With a seekable filesystem handle, WAV/RF64 sample ranges are read from selected frames in bounded input chunks. Providers that support whole-file reads but not seekable handles use the read fallback. The returned tensor occupies memory proportional to requested frames and channels; providers may have additional internal buffering costs.",
            "When a finite host-memory budget is active, the complete read-and-decode call is admitted before opening the file. If the installed host resource service cannot establish complete bounds for the captured filesystem and decoder, the call fails without reading the file. Calls without a finite host-memory budget retain the ordinary reading path when no measured route is selected.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "samples-and-rate", title: "Read samples and sample rate",
            program: "[y, Fs] = audioread('speech.wav');\nsize(y)\nFs",
            display_output: Some("ans = 2 1\nFs = 8000"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "audio-mono-wav" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: WAV_FILE }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y), [2 1])); assert(Fs == 8000); assert(abs(y(1)) < 1e-12); assert(abs(y(2) - 0.5) < 1e-12);" },
        },
        BuiltinExample {
            id: "frame-range", title: "Read a one-based inclusive frame range",
            program: "clip = audioread('speech.wav', [2 Inf]);\nclip",
            display_output: Some("clip = 0.5"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "audio-mono-wav" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: WAV_FILE }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(clip), [1 1])); assert(abs(clip(1) - 0.5) < 1e-12);" },
        },
        BuiltinExample {
            id: "native-class", title: "Preserve native PCM16 storage",
            program: "raw = audioread('speech.wav', 'native');\nclass(raw)",
            display_output: Some("ans = int16"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "audio-mono-wav" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: WAV_FILE }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(raw), 'int16')); assert(isequal(size(raw), [2 1])); assert(raw(1) == int16(0)); assert(raw(2) == int16(16384));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What shape does `audioread` return?", answer: "An N-by-C tensor: N decoded sample frames and C channels; mono is N-by-1." },
        BuiltinDocumentationFaq { question: "Does it use the active filesystem provider?", answer: "Yes. Native, sandbox, remote, and browser providers can supply file bytes when installed by the host." },
        BuiltinDocumentationFaq { question: "Does it decode MP3 or FLAC samples?", answer: "Not yet. `audioinfo` reads metadata for those containers; `audioread` currently decodes WAV/RF64 PCM and IEEE float." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "audioinfo", target: BuiltinDocumentationLinkTarget::Builtin("audioinfo") },
        BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Audio read entrypoint and controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/read.rs") },
            BuiltinDocumentationLink { label: "Captured filesystem source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/input.rs") },
            BuiltinDocumentationLink { label: "Complete-call admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/admission.rs") },
            BuiltinDocumentationLink { label: "Streaming WAV decoder", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/stream.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Normalized PCM16", location: "builtins::io::audio::tests::samples::audioread_reads_mono_pcm16_as_normalized_double" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Channel-major storage", location: "builtins::io::audio::tests::samples::audioread_stereo_output_is_column_major_by_channel" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Inclusive sample range", location: "builtins::io::audio::tests::samples::audioread_sample_range_is_one_based_inclusive_and_returns_fs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact native uint8", location: "builtins::io::audio::tests::samples::audioread_native_pcm8_uses_exact_uint8_storage" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed integer range rejection", location: "builtins::io::audio::tests::samples::audioread_sample_range_rejects_all_typed_integer_classes" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Provider filesystem", location: "builtins::io::audio::tests::samples::audioread_reads_via_active_filesystem_provider" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Finite admission before file effects", location: "builtins::io::audio::tests::admission::cases::finite_missing_and_unknown_evidence_reject_both_identities_before_file_effects" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
