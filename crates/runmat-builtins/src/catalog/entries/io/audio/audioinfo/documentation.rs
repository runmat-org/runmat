use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("audioinfo"),
    slug: Some("audioinfo"),
    summary: "Read metadata from an audio file.",
    description: "`info = audioinfo(filename)` returns a metadata structure without decoding sample data.",
    keywords: &["audioinfo", "audio", "metadata", "wav", "flac", "aiff", "mp3", "ogg", "sample rate", "channels"],
    related: &["audioread", "fileread", "fread"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and fields", paragraphs: &[
            "The filename is scalar string, singleton string array, or one-row character array. The active filesystem provider supplies the file. The result is a host scalar structure with `Filename`, `CompressionMethod`, `NumChannels`, `SampleRate`, `TotalSamples`, `Duration`, `BitsPerSample`, `BitRate`, `FileSize`, and `Format` fields.",
            "When headers do not establish a field, `TotalSamples`, `Duration`, `BitsPerSample`, or `BitRate` is `NaN`; the field is retained rather than omitted.",
            "When the caller explicitly discards the result, `audioinfo` still reads and validates the file but does not construct or retain the metadata structure.",
            "For WAVE_FORMAT_EXTENSIBLE, `BitsPerSample` reports valid sample precision; the encoded container may be wider. `audioread(..., 'native')` chooses its integer storage from the container width.",
        ] },
        BuiltinDocumentationSection { heading: "Containers and limits", paragraphs: &[
            "WAV/RF64 RIFF, FLAC STREAMINFO, AIFF/AIFC COMM, Ogg/Vorbis identification headers, and MPEG audio frame headers are recognized. Malformed or unsupported data raises a format error rather than partial metadata.",
            "MP3 information is estimated from the first valid MPEG frame header following an optional ID3v2 tag; variable-rate duration can differ from decoded duration. Inspection reads at most a 1 MiB prefix and, for larger Ogg files, an additional 64 KiB tail.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Metadata inspection is host filesystem I/O, a fusion boundary, and never a GPU kernel. It consumes no GPU-resident numeric input and returns a host struct. `audioinfo` does not decode samples; use `audioread` for supported WAV/RF64 sample payloads.",
            "When a finite host-memory budget is active, the complete metadata call is admitted before opening the file. If the installed host resource service cannot establish complete bounds for the captured filesystem and metadata work, the call fails without reading the file. Calls without a finite host-memory budget retain the ordinary metadata path when no measured route is selected.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "wav-metadata",
        title: "Read WAV metadata and duration",
        program: "info = audioinfo('sample.wav');\ninfo.SampleRate\ninfo.NumChannels\nseconds = info.Duration;\nframes = info.TotalSamples;",
        display_output: Some("ans = 8000\nans = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "audio-mono-wav" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "sample.wav", content: BuiltinFixtureContent::Bytes(super::super::fixtures::MONO_WAV) }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(info.SampleRate == 8000); assert(info.NumChannels == 1); assert(frames == 2); assert(abs(seconds - 0.00025) < 1e-12); assert(strcmp(info.Format, 'WAV'));" },
    }, BuiltinExample {
        id: "flac-duration",
        title: "Inspect FLAC duration and sample count",
        program: "info = audioinfo('speech.flac');\nseconds = info.Duration;\nframes = info.TotalSamples;\nseconds\nframes",
        display_output: Some("seconds = 2\nframes = 96000"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "audio-flac-streaminfo" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "speech.flac", content: BuiltinFixtureContent::Bytes(super::super::fixtures::STEREO_FLAC_METADATA) }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(info.Format, 'FLAC')); assert(info.SampleRate == 48000); assert(info.NumChannels == 2); assert(info.BitsPerSample == 24); assert(frames == 96000); assert(seconds == 2);" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `audioinfo` decode audio samples?", answer: "No. It reads container metadata only; sample decoding belongs to `audioread`." },
        BuiltinDocumentationFaq { question: "Which formats are supported?", answer: "WAV/RF64, FLAC, AIFF/AIFC, Ogg/Vorbis, and MPEG audio frame headers, including common MP3 files." },
        BuiltinDocumentationFaq { question: "Why are some fields `NaN`?", answer: "Some compressed formats do not expose exact sample count, bit depth, or duration in their primary headers. RunMat preserves the field shape and uses `NaN` for unavailable metadata." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "fileread", target: BuiltinDocumentationLinkTarget::Builtin("fileread") },
        BuiltinDocumentationLink { label: "fread", target: BuiltinDocumentationLinkTarget::Builtin("fread") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Audio metadata reader", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/info.rs") },
            BuiltinDocumentationLink { label: "Captured filesystem source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/input.rs") },
            BuiltinDocumentationLink { label: "Complete-call admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/audio/admission.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "WAV metadata", location: "builtins::io::audio::tests::metadata::audioinfo_reads_wav_metadata" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "FLAC STREAMINFO", location: "builtins::io::audio::tests::metadata::audioinfo_reads_flac_streaminfo" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "MP3 frame header", location: "builtins::io::audio::tests::metadata::audioinfo_parses_mp3_frame_header" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Unknown data error", location: "builtins::io::audio::tests::metadata::audioinfo_rejects_unknown_data" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Finite admission before file effects", location: "builtins::io::audio::tests::admission::cases::finite_missing_and_unknown_evidence_reject_both_identities_before_file_effects" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Discarded metadata output", location: "builtins::io::audio::tests::admission::zero_output::discarded_audioinfo_has_no_retained_value_but_still_needs_scratch" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Metadata scan cancellation", location: "builtins::io::audio::tests::admission::zero_output::discarded_audioinfo_cancellation_during_synchronous_read_never_publishes" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
