const KEY_WRAP_INFO = new TextEncoder().encode("runmat-package-key-wrap-v1");

export interface PrivatePackageEnvelopeContext {
  registry: string;
  namespace: string;
  name: string;
  version: string;
  recipientKeyId: string;
  recipientKeyFingerprint: string;
}

export interface BrowserPrivatePackageKeyProvider {
  privateKeyForEnvelope(
    context: PrivatePackageEnvelopeContext
  ): CryptoKey | null | Promise<CryptoKey | null>;
}

interface EncryptionMetadata {
  schemaVersion: number;
  contentCipher: string;
  keyVersion: number;
  plaintextDigest: string;
  plaintextByteLen: number;
  nonce: string;
  aadDigest: string;
}

interface KeyEnvelope {
  schemaVersion: number;
  algorithm: string;
  recipientKeyId: string;
  recipientKeyFingerprint: string;
  ephemeralPublicKey: string;
  nonce: string;
  wrappedKey: string;
  contextDigest: string;
}

interface PrivateRelease {
  registry: string;
  namespace: string;
  name: string;
  version: string;
  encryption?: EncryptionMetadata | null;
  artifact: {
    digest: string;
    treeDigest: string;
    keyEnvelopes?: KeyEnvelope[];
  };
}

export async function decryptRegistryPrivateArtifact(
  releaseValue: unknown,
  ciphertext: Uint8Array,
  keys: BrowserPrivatePackageKeyProvider
): Promise<Uint8Array> {
  const release = releaseValue as PrivateRelease;
  const metadata = release.encryption;
  if (!metadata) {
    throw new Error("registry release has no private artifact metadata");
  }
  validateMetadata(release, metadata);
  await verifyDigest(release.artifact.digest, ciphertext, "ciphertext");
  const aad = canonicalArtifactAad(release, metadata);
  await verifyDigest(metadata.aadDigest, aad, "artifact associated data");
  const envelopes = release.artifact.keyEnvelopes ?? [];
  if (envelopes.length === 0) {
    throw new Error("registry release has no authorized package key envelope");
  }
  const subtle = cryptoApi();
  for (const envelope of envelopes) {
    validateEnvelope(envelope);
    const privateKey = await keys.privateKeyForEnvelope({
      registry: release.registry,
      namespace: release.namespace,
      name: release.name,
      version: release.version,
      recipientKeyId: envelope.recipientKeyId,
      recipientKeyFingerprint: envelope.recipientKeyFingerprint
    });
    if (!privateKey) continue;
    validatePrivateKey(privateKey);
    const context = canonicalEnvelopeContext(metadata, envelope);
    await verifyDigest(envelope.contextDigest, context, "key envelope context");
    const ephemeral = await subtle.importKey(
      "raw",
      ownedBuffer(decodeBase64Url(envelope.ephemeralPublicKey, 65)),
      { name: "ECDH", namedCurve: "P-256" },
      false,
      []
    );
    const shared = new Uint8Array(
      await subtle.deriveBits({ name: "ECDH", public: ephemeral }, privateKey, 256)
    );
    try {
      const wrappingKey = await deriveWrappingKey(subtle, shared, envelope.contextDigest);
      const contentKeyBytes = new Uint8Array(
        await subtle.decrypt(
          {
            name: "AES-GCM",
            iv: ownedBuffer(decodeBase64Url(envelope.nonce, 12)),
            additionalData: ownedBuffer(context),
            tagLength: 128
          },
          wrappingKey,
          ownedBuffer(decodeBase64Url(envelope.wrappedKey, 48))
        )
      );
      try {
        if (contentKeyBytes.byteLength !== 32) {
          throw new Error("package content key has an invalid length");
        }
        const contentKey = await subtle.importKey(
          "raw",
          ownedBuffer(contentKeyBytes),
          { name: "AES-GCM" },
          false,
          ["decrypt"]
        );
        const plaintext = new Uint8Array(
          await subtle.decrypt(
            {
              name: "AES-GCM",
              iv: ownedBuffer(decodeBase64Url(metadata.nonce, 12)),
              additionalData: ownedBuffer(aad),
              tagLength: 128
            },
            contentKey,
            ownedBuffer(ciphertext)
          )
        );
        if (plaintext.byteLength !== metadata.plaintextByteLen) {
          plaintext.fill(0);
          throw new Error("private artifact plaintext length is invalid");
        }
        try {
          await verifyDigest(metadata.plaintextDigest, plaintext, "plaintext");
        } catch (error) {
          plaintext.fill(0);
          throw error;
        }
        return plaintext;
      } finally {
        contentKeyBytes.fill(0);
      }
    } catch (error) {
      if (error instanceof Error && error.message.includes("digest")) throw error;
    } finally {
      shared.fill(0);
    }
  }
  throw new Error("no local private key could decrypt an authorized package key envelope");
}

function validateMetadata(release: PrivateRelease, value: EncryptionMetadata): void {
  if (
    value.schemaVersion !== 1 ||
    value.contentCipher !== "aes256-gcm" ||
    !Number.isSafeInteger(value.keyVersion) ||
    value.keyVersion <= 0 ||
    !Number.isSafeInteger(value.plaintextByteLen) ||
    value.plaintextByteLen <= 0 ||
    typeof release.registry !== "string" ||
    typeof release.namespace !== "string" ||
    typeof release.name !== "string" ||
    typeof release.version !== "string" ||
    typeof release.artifact?.treeDigest !== "string"
  ) {
    throw new Error("registry release contains invalid private artifact metadata");
  }
  decodeBase64Url(value.nonce, 12);
}

function validateEnvelope(value: KeyEnvelope): void {
  if (
    value.schemaVersion !== 1 ||
    value.algorithm !== "p256-hkdf-sha256-aes256-gcm" ||
    !value.recipientKeyId
  ) {
    throw new Error("registry release contains an invalid package key envelope");
  }
  decodeBase64Url(value.ephemeralPublicKey, 65);
  decodeBase64Url(value.nonce, 12);
  decodeBase64Url(value.wrappedKey, 48);
}

function validatePrivateKey(value: CryptoKey): void {
  const algorithm = value.algorithm as EcKeyAlgorithm;
  if (
    value.type !== "private" ||
    value.extractable ||
    algorithm.name !== "ECDH" ||
    algorithm.namedCurve !== "P-256" ||
    !value.usages.includes("deriveBits")
  ) {
    throw new Error(
      "private package keys must be non-extractable P-256 ECDH keys with deriveBits usage"
    );
  }
}

function canonicalArtifactAad(
  release: PrivateRelease,
  metadata: EncryptionMetadata
): Uint8Array {
  return jsonBytes({
    format: "runmat-private-artifact-aad-v1",
    registry: release.registry,
    namespace: release.namespace,
    name: release.name,
    version: release.version,
    treeDigest: release.artifact.treeDigest,
    contentCipher: metadata.contentCipher,
    keyVersion: metadata.keyVersion,
    plaintextDigest: metadata.plaintextDigest,
    plaintextByteLen: metadata.plaintextByteLen
  });
}

function canonicalEnvelopeContext(
  metadata: EncryptionMetadata,
  envelope: KeyEnvelope
): Uint8Array {
  return jsonBytes({
    format: "runmat-package-key-envelope-context-v1",
    artifactAadDigest: metadata.aadDigest,
    keyVersion: metadata.keyVersion,
    recipientKeyId: envelope.recipientKeyId,
    recipientKeyFingerprint: envelope.recipientKeyFingerprint,
    ephemeralPublicKey: envelope.ephemeralPublicKey,
    algorithm: envelope.algorithm
  });
}

async function deriveWrappingKey(
  subtle: SubtleCrypto,
  shared: Uint8Array,
  contextDigest: string
): Promise<CryptoKey> {
  const material = await subtle.importKey("raw", ownedBuffer(shared), "HKDF", false, [
    "deriveKey"
  ]);
  return subtle.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: ownedBuffer(digestBytes(contextDigest)),
      info: ownedBuffer(KEY_WRAP_INFO)
    },
    material,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"]
  );
}

async function verifyDigest(
  expected: string,
  bytes: Uint8Array,
  label: string
): Promise<void> {
  const actual = new Uint8Array(
    await cryptoApi().digest("SHA-256", ownedBuffer(bytes))
  );
  const expectedBytes = digestBytes(expected);
  let difference = actual.byteLength ^ expectedBytes.byteLength;
  for (let index = 0; index < actual.byteLength; index += 1) {
    difference |= actual[index]! ^ (expectedBytes[index] ?? 0);
  }
  if (difference !== 0) {
    throw new Error(`private artifact ${label} digest is invalid`);
  }
}

function digestBytes(value: string): Uint8Array {
  const match = /^sha256:([0-9a-f]{64})$/.exec(value);
  if (!match) throw new Error("private artifact digest is invalid");
  const hex = match[1]!;
  return Uint8Array.from(
    { length: 32 },
    (_, index) => Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16)
  );
}

function decodeBase64Url(value: string, expectedLength: number): Uint8Array {
  if (!/^[A-Za-z0-9_-]+$/.test(value) || value.includes("=")) {
    throw new Error("private artifact base64url value is invalid");
  }
  const padded = value.replaceAll("-", "+").replaceAll("_", "/").padEnd(
    Math.ceil(value.length / 4) * 4,
    "="
  );
  const decoded = Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
  if (decoded.byteLength !== expectedLength) {
    throw new Error("private artifact base64url value has an invalid length");
  }
  return decoded;
}

function jsonBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(value));
}

function ownedBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.slice().buffer as ArrayBuffer;
}

function cryptoApi(): SubtleCrypto {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto is required for private registry packages");
  }
  return globalThis.crypto.subtle;
}
