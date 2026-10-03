use crate::*;

pub(super) const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "loopback-save",
    title: "Save an HTTP response to a file",
    program: "saved = websave('report.txt', '__RUNMAT_HTTP_BASE_URL__/report');\ntext = fileread(saved);\ndisp(text);",
    display_output: Some("RunMat report"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "download-response" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario {
            exchanges: &[BuiltinHttpExchange {
                request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/report", body: None },
                response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "text/plain" }], body: b"RunMat report" },
            }],
        }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(saved)); assert(strcmp(saved, 'report.txt')); assert(strcmp(text, 'RunMat report'));" },
}];
