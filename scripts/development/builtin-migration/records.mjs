export function createRecords() {
  const records = new Map();
  const record = (name) => {
    const identity = name.toLowerCase();
    if (!records.has(identity)) records.set(identity, newRecord(identity));
    return records.get(identity);
  };
  return { records, record };
}

function newRecord(identity) {
  return {
    identity,
    spellings: new Set(),
    runtime: [],
    catalogPaths: new Set(),
    catalogDocumentationPaths: new Set(),
    sidecarPaths: new Set(),
    shadowPaths: new Set(),
    providerPaths: new Set(),
    fusionPaths: new Set(),
    testPaths: new Set(),
    catalogResolverPaths: new Set(),
    nativeLinkPaths: new Set(),
    wasmRegistrations: new Set(),
    documentationEvidence: [],
    input: emptyClassification(),
  };
}

export function emptyClassification() {
  return { disposition: null, canonical: null, domain: null, family: null, reason: null, review: { status: "unreviewed", evidence: [] } };
}
