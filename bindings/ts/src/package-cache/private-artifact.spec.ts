import { describe, expect, it } from "vitest";
import { decryptRegistryPrivateArtifact } from "./private-artifact.js";

const encoder = new TextEncoder();

describe("private registry artifact WebCrypto adapter", () => {
  it("decrypts the portable P-256/HKDF/AES-GCM envelope and artifact format", async () => {
    const subtle = globalThis.crypto.subtle;
    const generatedRecipient = (await subtle.generateKey(
      { name: "ECDH", namedCurve: "P-256" },
      true,
      ["deriveBits"]
    )) as CryptoKeyPair;
    const recipientJwk = await subtle.exportKey("jwk", generatedRecipient.privateKey);
    const recipientPrivate = await subtle.importKey(
      "jwk",
      recipientJwk,
      { name: "ECDH", namedCurve: "P-256" },
      false,
      ["deriveBits"]
    );
    const recipientPublic = new Uint8Array(
      await subtle.exportKey("raw", generatedRecipient.publicKey)
    );
    const recipientFingerprint = await digest(recipientPublic);
    const plaintext = encoder.encode('{"schema_version":1,"entries":[]}');
    const treeDigest = await digest(encoder.encode("tree"));
    const artifactNonce = Uint8Array.from({ length: 12 }, (_, index) => index + 1);
    const metadata = {
      schemaVersion: 1,
      contentCipher: "aes256-gcm",
      keyVersion: 7,
      plaintextDigest: await digest(plaintext),
      plaintextByteLen: plaintext.byteLength,
      nonce: base64Url(artifactNonce),
      aadDigest: ""
    };
    const release = {
      registry: "https://packages.runmat.test",
      namespace: "acme",
      name: "private",
      version: "1.2.3",
      encryption: metadata,
      artifact: {
        digest: "",
        treeDigest,
        keyEnvelopes: [] as unknown[]
      }
    };
    const aad = jsonBytes({
      format: "runmat-private-artifact-aad-v1",
      registry: release.registry,
      namespace: release.namespace,
      name: release.name,
      version: release.version,
      treeDigest,
      contentCipher: metadata.contentCipher,
      keyVersion: metadata.keyVersion,
      plaintextDigest: metadata.plaintextDigest,
      plaintextByteLen: metadata.plaintextByteLen
    });
    metadata.aadDigest = await digest(aad);
    const contentKeyBytes = crypto.getRandomValues(new Uint8Array(32));
    const contentKey = await subtle.importKey(
      "raw",
      contentKeyBytes,
      "AES-GCM",
      false,
      ["encrypt"]
    );
    const ciphertext = new Uint8Array(
      await subtle.encrypt(
        { name: "AES-GCM", iv: artifactNonce, additionalData: aad },
        contentKey,
        plaintext
      )
    );
    release.artifact.digest = await digest(ciphertext);

    const ephemeral = (await subtle.generateKey(
      { name: "ECDH", namedCurve: "P-256" },
      true,
      ["deriveBits"]
    )) as CryptoKeyPair;
    const ephemeralPublic = new Uint8Array(
      await subtle.exportKey("raw", ephemeral.publicKey)
    );
    const shared = new Uint8Array(
      await subtle.deriveBits(
        { name: "ECDH", public: generatedRecipient.publicKey },
        ephemeral.privateKey,
        256
      )
    );
    const envelope = {
      schemaVersion: 1,
      algorithm: "p256-hkdf-sha256-aes256-gcm",
      recipientKeyId: "pkr_test",
      recipientKeyFingerprint: recipientFingerprint,
      ephemeralPublicKey: base64Url(ephemeralPublic),
      nonce: base64Url(Uint8Array.from({ length: 12 }, (_, index) => index + 21)),
      wrappedKey: "",
      contextDigest: ""
    };
    const context = jsonBytes({
      format: "runmat-package-key-envelope-context-v1",
      artifactAadDigest: metadata.aadDigest,
      keyVersion: metadata.keyVersion,
      recipientKeyId: envelope.recipientKeyId,
      recipientKeyFingerprint: recipientFingerprint,
      ephemeralPublicKey: envelope.ephemeralPublicKey,
      algorithm: envelope.algorithm
    });
    envelope.contextDigest = await digest(context);
    const hkdf = await subtle.importKey("raw", shared, "HKDF", false, ["deriveKey"]);
    const wrappingKey = await subtle.deriveKey(
      {
        name: "HKDF",
        hash: "SHA-256",
        salt: digestBytes(envelope.contextDigest),
        info: encoder.encode("runmat-package-key-wrap-v1")
      },
      hkdf,
      { name: "AES-GCM", length: 256 },
      false,
      ["encrypt"]
    );
    envelope.wrappedKey = base64Url(
      new Uint8Array(
        await subtle.encrypt(
          {
            name: "AES-GCM",
            iv: decodeBase64Url(envelope.nonce),
            additionalData: context
          },
          wrappingKey,
          contentKeyBytes
        )
      )
    );
    release.artifact.keyEnvelopes = [envelope];

    const result = await decryptRegistryPrivateArtifact(release, ciphertext, {
      privateKeyForEnvelope: ({ recipientKeyId }) =>
        recipientKeyId === "pkr_test" ? recipientPrivate : null
    });
    expect(new TextDecoder().decode(result)).toBe(new TextDecoder().decode(plaintext));
  });

  it("rejects extractable private keys", async () => {
    const key = (await crypto.subtle.generateKey(
      { name: "ECDH", namedCurve: "P-256" },
      true,
      ["deriveBits"]
    )) as CryptoKeyPair;
    const encryption = {
      schemaVersion: 1,
      contentCipher: "aes256-gcm",
      keyVersion: 1,
      plaintextDigest: `sha256:${"0".repeat(64)}`,
      plaintextByteLen: 1,
      nonce: base64Url(new Uint8Array(12)),
      aadDigest: ""
    };
    const release = {
          registry: "https://packages.runmat.test",
          namespace: "acme",
          name: "private",
          version: "1.0.0",
          encryption,
          artifact: {
            digest: await digest(new Uint8Array()),
            treeDigest: `sha256:${"1".repeat(64)}`,
            keyEnvelopes: [
              {
                schemaVersion: 1,
                algorithm: "p256-hkdf-sha256-aes256-gcm",
                recipientKeyId: "pkr_test",
                recipientKeyFingerprint: `sha256:${"2".repeat(64)}`,
                ephemeralPublicKey: base64Url(
                  new Uint8Array(await crypto.subtle.exportKey("raw", key.publicKey))
                ),
                nonce: base64Url(new Uint8Array(12)),
                wrappedKey: base64Url(new Uint8Array(48)),
                contextDigest: `sha256:${"3".repeat(64)}`
              }
            ]
          }
        };
    encryption.aadDigest = await digest(
      jsonBytes({
        format: "runmat-private-artifact-aad-v1",
        registry: release.registry,
        namespace: release.namespace,
        name: release.name,
        version: release.version,
        treeDigest: release.artifact.treeDigest,
        contentCipher: encryption.contentCipher,
        keyVersion: encryption.keyVersion,
        plaintextDigest: encryption.plaintextDigest,
        plaintextByteLen: encryption.plaintextByteLen
      })
    );
    await expect(
      decryptRegistryPrivateArtifact(
        release,
        new Uint8Array(),
        { privateKeyForEnvelope: () => key.privateKey }
      )
    ).rejects.toThrow("non-extractable");
  });
});

async function digest(bytes: Uint8Array): Promise<string> {
  const value = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return `sha256:${Array.from(value, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function digestBytes(value: string): Uint8Array {
  const hex = value.slice("sha256:".length);
  return Uint8Array.from(
    { length: 32 },
    (_, index) => Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16)
  );
}

function base64Url(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

function decodeBase64Url(value: string): Uint8Array {
  const padded = value.replaceAll("-", "+").replaceAll("_", "/").padEnd(
    Math.ceil(value.length / 4) * 4,
    "="
  );
  return Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
}

function jsonBytes(value: unknown): Uint8Array {
  return encoder.encode(JSON.stringify(value));
}
