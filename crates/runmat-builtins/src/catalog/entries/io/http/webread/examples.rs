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
},
BuiltinExample {
    id: "loopback-json-query-options",
    title: "Read JSON with an options structure and query value",
    program: "opts = weboptions('ContentType', 'json', 'Timeout', 15);\nweather = webread('__RUNMAT_HTTP_BASE_URL__/weather', opts, 'city', 'Reykjavik');\ndisp(weather.temperatureC);",
    display_output: Some("2.3"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "json-query-options" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/weather?city=Reykjavik", body: None },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/json" }], body: br#"{"temperatureC":2.3}"# },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(weather)); assert(abs(weather.temperatureC - 2.3) < 1e-12);" },
},
BuiltinExample {
    id: "loopback-text-extract",
    title: "Extract text from a downloaded character vector",
    program: "html = webread('__RUNMAT_HTTP_BASE_URL__/index.txt', 'Timeout', 5);\nextract = html(1:6);\ndisp(extract);",
    display_output: Some("RunMat"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "text-extract" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/index.txt", body: None },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "text/plain" }], body: b"RunMat reference text" },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(html)); assert(strcmp(extract, 'RunMat'));" },
},
BuiltinExample {
    id: "loopback-binary",
    title: "Retrieve exact binary response bytes",
    program: "bytes = webread('__RUNMAT_HTTP_BASE_URL__/logo.png', 'ContentType', 'binary');\nassert(isa(bytes, 'uint8'));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "binary-response" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/logo.png", body: None },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/octet-stream" }], body: &[1, 2, 3, 254, 255] },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(bytes), [1 5])); assert(isequal(double(bytes), [1 2 3 254 255]));" },
},
BuiltinExample {
    id: "loopback-auth-headers",
    title: "Supply headers and HTTP basic credentials",
    program: "headers = struct('Accept', 'application/json', 'X-Client', 'RunMat');\ndata = webread('__RUNMAT_HTTP_BASE_URL__/me', 'Username', 'ada', 'Password', 'secret', 'HeaderFields', headers, 'ContentType', 'json');\ndisp(data.name);",
    display_output: Some("Ada"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "auth-headers" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/me", body: None },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/json" }], body: br#"{"name":"Ada"}"# },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(data)); assert(strcmp(data.name, 'Ada'));" },
},
BuiltinExample {
    id: "loopback-query-struct",
    title: "Pass query values through a structure",
    program: "query = struct('limit', 25, 'sort', 'name');\nresponse = webread('__RUNMAT_HTTP_BASE_URL__/resources', query, 'ContentType', 'json');\ndisp(response.count);",
    display_output: Some("1"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeLoopbackNetwork,
    fixture: BuiltinExampleFixture::Loopback(BuiltinLoopbackFixture {
        id: BuiltinExampleFixtureId { local_name: "query-struct" },
        scenario: BuiltinLoopbackScenario::Http(BuiltinHttpScenario { exchanges: &[BuiltinHttpExchange {
            request: BuiltinHttpRequest { method: BuiltinHttpMethod::Get, path: "/resources?limit=25&sort=name", body: None },
            response: BuiltinHttpResponse { status: 200, headers: &[BuiltinHttpHeader { name: "Content-Type", value: "application/json" }], body: br#"{"count":1}"# },
        }] }),
        endpoint_substitutions: &[BuiltinEndpointSubstitution::HttpBaseUrl],
    }),
    requirements: BuiltinExampleRequirements { host: BuiltinExampleHostRequirement::NativeOnly, ..BuiltinExampleRequirements::NONE },
    verification: BuiltinExampleVerification::Assertions { source: "assert(isstruct(response)); assert(response.count == 1);" },
}];
