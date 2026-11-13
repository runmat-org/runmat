use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("read"), slug: Some("read"),
    summary: "Read bytes or typed values from a `tcpclient` connection with MATLAB-compatible behavior.", description: "`read(t)` consumes data waiting on the TCP/IP client returned by `tcpclient` (or `accept`). It follows MATLAB behavior for timeout handling, byte-order decoding, and optional datatype conversion so networking code ports directly.",
    keywords: &["read", "tcpclient", "networking", "socket", "binary data"],
    related: &["tcpclient", "accept", "write", "readline", "close", "tcpserver"],
    sections: &[
        BuiltinDocumentationSection { heading: "Read forms and output classes", paragraphs: &[
            "`data = read(t)` waits until at least one byte becomes available (subject to the client timeout) and then drains the socket buffer, returning a row `uint8` vector. If the peer closes the connection without sending data, the result is an empty `uint8` row vector.",
            "`data = read(t, count)` blocks until exactly `count` `uint8` values are available (again honouring `Timeout`). The count may use any native integer class and is read exactly; when the peer closes the socket before satisfying the request the builtin raises `RunMat:read:ConnectionClosed`.",
            "`data = read(t, count, datatype)` interprets the values using the requested MATLAB datatype. Supported tokens are `\"uint8\"` (default), `\"int8\"`, `\"uint16\"`, `\"int16\"`, `\"uint32\"`, `\"int32\"`, `\"uint64\"`, `\"int64\"`, `\"single\"`, `\"double\"`, `\"char\"`, and `\"string\"`. Numeric results retain the requested native class, including exact 64-bit integer values and physical single precision; `\"char\"` produces a character row vector and `\"string\"` returns a scalar string.",
            "Every call honours the byte order configured when the live client was constructed. Little-endian is the default, but `\"big-endian\"` is respected for data written in network byte order; editing a visible copied property does not reconfigure the socket.",
            "The builtin gathers GPU-resident arguments automatically, executes the socket read on the CPU, and returns host values.",
            "Errors are raised with stable identifiers: invalid or stale client objects trigger `RunMat:read:InvalidTcpClient`, timeouts emit `RunMat:read:Timeout`, and connection closures before a requested count is met raise `RunMat:read:ConnectionClosed`.",
        ] },
        BuiltinDocumentationSection { heading: "Host placement and compatibility", paragraphs: &[
            "Networking is a host-only subsystem. Resident client metadata or count arguments gather transparently before socket I/O. No acceleration-provider hook participates, and the result is always a host value in its selected native numeric class, a character array, or a string.",
            "The socket operation terminates fusion and produces host-owned values. No GPU kernel or provider execution route exists for this identity.",
            "The documented integer count and exact numeric payload classes are preserved; bytes are decoded directly into requested storage without binary64 conversion of 64-bit integers.",
        ] },
        BuiltinDocumentationSection { heading: "Further illustrative programs", paragraphs: &[
            "Reading a fixed number of bytes from a TCP echo service\nclient = tcpclient(\"127.0.0.1\", 50000);\nwrite(client, uint8(1:6));\npayload = read(client, 6)\nExpected display: payload =\n     1     2     3     4     5     6",
            "Reading ASCII text as characters\nclient = tcpclient(\"127.0.0.1\", 50001);\nwrite(client, \"RunMat TCP\");\nchars = read(client, 10, \"char\")\nExpected display: chars =\n    'RunMat TCP'",
            "Reading doubles written in big-endian byte order\nclient = tcpclient(\"localhost\", 50002, \"ByteOrder\", \"big-endian\");\nwrite(client, swapbytes([1 2 3], \"double\"));\nvalues = read(client, 3, \"double\")\nExpected display: values =\n     1     2     3",
            "Reading all available data without specifying a count\nclient = tcpclient(\"127.0.0.1\", 50003);\nwrite(client, uint8([10 20 30]));\nburst = read(client)\nExpected display: burst =\n    10    20    30",
            "Handling read timeouts gracefully\nclient = tcpclient(\"example.com\", 12345, \"Timeout\", 0.5);\ntry\n    data = read(client, 64);\ncatch err\n    disp(err.identifier)\nend\nExpected display: RunMat:read:Timeout",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "loopback-exact-u8", title: "Read exact bytes from a local peer",
        program: "client = tcpclient('__RUNMAT_LOOPBACK_HOST__', __RUNMAT_LOOPBACK_PORT__);\nwrite(client, uint8(9));\ndata = read(client, 3, 'uint8');\nassert(isa(data, 'uint8'));\nassert(isequal(double(data), [1 2 3]));\nclose(client);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeLoopbackNetwork,
        fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
            id: BuiltinExampleFixtureId { local_name: "read-exact-u8" },
            scenario: BuiltinLoopbackScenario::Tcp(BuiltinTcpScenario { exchanges: &[BuiltinTcpExchange { client_bytes: &[9], server_bytes: &[1, 2, 3] }] }),
            endpoint_substitutions: &[BuiltinEndpointSubstitution::LoopbackHost, BuiltinEndpointSubstitution::LoopbackPort],
        }),
        requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(data), [1 3])); assert(isa(data, 'uint8')); assert(isequal(double(data), [1 2 3]));" },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `read` modify the client object?", answer: "No. The builtin uses its sealed session-owned socket authority; visible object properties are not mutated in place." },
        BuiltinDocumentationFaq { question: "What happens when the remote host closes the connection?", answer: "If the peer closes the connection before the requested count is satisfied, `read` raises `RunMat:read:ConnectionClosed`. When no specific count is requested (`read(t)`), the builtin returns whatever data was available before the closure (possibly an empty vector)." },
        BuiltinDocumentationFaq { question: "Does `read` support infinite timeouts?", answer: "Yes. Pass `\"Timeout\", inf` when constructing the client to leave its socket in blocking mode. Editing the visible client property does not reconfigure the captured live socket." },
        BuiltinDocumentationFaq { question: "How are multibyte integers decoded?", answer: "RunMat honours the client’s `ByteOrder` property (`\"little-endian\"` or `\"big-endian\"`). For example, `read(t, 4, \"uint16\")` consumes eight bytes and interprets each pair in the configured byte order." },
        BuiltinDocumentationFaq { question: "Can I read UTF-8 strings directly?", answer: "The current `\"string\"` form maps each received byte directly to a Unicode character of the same value; it does not perform UTF-8 decoding. Use a separate decoder when the peer sends multibyte UTF-8." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tcpclient", target: BuiltinDocumentationLinkTarget::Builtin("tcpclient") },
        BuiltinDocumentationLink { label: "accept", target: BuiltinDocumentationLinkTarget::Builtin("accept") },
        BuiltinDocumentationLink { label: "write", target: BuiltinDocumentationLinkTarget::Builtin("write") },
        BuiltinDocumentationLink { label: "readline", target: BuiltinDocumentationLinkTarget::Builtin("readline") },
        BuiltinDocumentationLink { label: "close", target: BuiltinDocumentationLinkTarget::Builtin("close") },
        BuiltinDocumentationLink { label: "tcpserver", target: BuiltinDocumentationLinkTarget::Builtin("tcpserver") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime TCP reader", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/net/read.rs") }, BuiltinDocumentationLink { label: "Typed TCP datatype authority", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/io/net/datatype.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Socket read tests", location: "builtins::io::net::read::tests" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
