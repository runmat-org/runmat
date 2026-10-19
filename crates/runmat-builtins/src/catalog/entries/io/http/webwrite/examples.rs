use crate::*;

pub(super) const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "loopback-post",
    title: "Post text and read a response",
    program: "reply = webwrite('__RUNMAT_HTTP_BASE_URL__/echo', 'hello', 'MediaType', 'text/plain', 'ContentType', 'text');\ndisp(reply);",
    display_output: Some("accepted"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "post-response" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario {
            exchanges: &[BuiltinHttpExchange {
                request: BuiltinHttpRequest { method: BuiltinHttpMethod::Post, path: "/echo", body: Some(b"hello") },
                response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "text/plain" }], body: b"accepted" },
            }],
        }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(reply)); assert(strcmp(reply, 'accepted'));" },
},
BuiltinExample {
    id: "loopback-form-fields",
    title: "Post form fields and decode JSON",
    program: "payload = struct('name', 'Ada', 'score', 42);\nopts = struct('ContentType', 'json');\nreply = webwrite('__RUNMAT_HTTP_BASE_URL__/submit', payload, opts);\ndisp(reply.status);",
    display_output: Some("ok"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "form-fields" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Post, path: "/submit", body: Some(b"name=Ada&score=42") },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/json" }], body: br#"{"status":"ok"}"# },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(reply)); assert(strcmp(reply.status, 'ok'));" },
},
BuiltinExample {
    id: "loopback-json-body",
    title: "Send a JSON body",
    program: "body = struct('title', 'RunMat', 'stars', 5);\nopts = struct('MediaType', 'application/json', 'ContentType', 'json');\nresp = webwrite('__RUNMAT_HTTP_BASE_URL__/projects', body, opts);\ndisp(resp.ok);",
    display_output: Some("1"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "json-body" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Post, path: "/projects", body: Some(br#"{"stars":5,"title":"RunMat"}"#) },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/json" }], body: br#"{"ok":true}"# },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(resp)); assert(resp.ok);" },
},
BuiltinExample {
    id: "loopback-binary-upload",
    title: "Upload raw bytes",
    program: "bytes = uint8([1 2 3 4 5]);\nreply = webwrite('__RUNMAT_HTTP_BASE_URL__/upload', bytes, 'ContentType', 'binary', 'MediaType', 'application/octet-stream');\nassert(isequal(reply, [115 116 111 114 101 100]));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "binary-upload" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Post, path: "/upload", body: Some(&[1, 2, 3, 4, 5]) },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "text/plain" }], body: b"stored" },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(reply, [115 116 111 114 101 100]));" },
},
BuiltinExample {
    id: "loopback-auth-query",
    title: "Write with credentials, headers, and query parameters",
    program: "headers = struct('X-Client', 'RunMat', 'Accept', 'application/json');\nopts = struct('Username', 'ada', 'Password', 'lovelace', 'HeaderFields', headers, 'QueryParameters', struct('verbose', true), 'ContentType', 'json');\nprofile = webwrite('__RUNMAT_HTTP_BASE_URL__/me', struct(), opts);\ndisp(profile.name);",
    display_output: Some("Ada"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "auth-query" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Post, path: "/me?verbose=true", body: Some(b"") },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/json" }], body: br#"{"name":"Ada"}"# },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(profile)); assert(strcmp(profile.name, 'Ada'));" },
}];
