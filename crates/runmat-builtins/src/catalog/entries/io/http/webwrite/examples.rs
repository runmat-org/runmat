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
}];
