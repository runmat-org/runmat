# Frozen native artifact compatibility fixture

`native-object-manifest-4.json` preserves the final native-object manifest 4 field layout. The admission test embeds these exact historical metadata bytes in a newly hashed `NativeObjectPayload` and `ProgramArtifact`. This proves that valid outer digests do not admit stale nested metadata. The fixture is read directly and is not produced by mutating a current manifest during the test.

`runtime-archive-manifest-3.json` preserves the prior runtime-archive manifest with Native IR 5 and native-object manifest 4 nested revisions. Admission rejects its archive schema before target-specific validation, so the fixture behaves identically on every host.
