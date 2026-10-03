use crate::*;

pub(super) const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "loopback-text",
    title: "Read an HTTP text response",
    program:
        "data = webread('__RUNMAT_HTTP_BASE_URL__/message', 'ContentType', 'text');\ndisp(data);",
    display_output: Some("RunMat HTTP fixture"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId {
            local_name: "text-response",
        },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario {
            exchanges: &[BuiltinHttpExchange {
                request: BuiltinHttpRequest {
                    method: BuiltinHttpMethod::Get,
                    path: "/message",
                    body: None,
                },
                response: BuiltinHttpResponse {
                    status: 200,
                    headers: &[BuiltinHttpHeader {
                        name: "Content-Type",
                        value: "text/plain",
                    }],
                    body: b"RunMat HTTP fixture",
                },
            }],
        }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements {
        host: BuiltinExampleHostRequirement::NativeOnly,
        ..BuiltinExampleRequirements::NONE
    },
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(ischar(data)); assert(strcmp(data, 'RunMat HTTP fixture'));",
    },
}];
