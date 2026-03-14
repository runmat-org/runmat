## RunMat Cloud Infrastructure

This directory provisions the RunMat services on Google Cloud Platform.

The following services are currently provisioned:

- Cloud DNS zones for `telemetry.runmat.org` and `telemetry.runmat.com`
- Cloud Run service that hosts an HTTP telemetry ingress endpoint (forwards to collector in runmat-server)
- UDP forwarder running on a regional Managed Instance Group + UDP load balancer

### Remote state

Terraform state is stored in a GCS bucket via the `gcs` backend. Initialize locally with:

```bash
GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/terraform-sa.json \
terraform init \
  -backend-config="bucket=$GCS_TF_STATE_BUCKET" \
  -backend-config="prefix=infra/terraform.tfstate"
```

Set the following environment variables (or prefix `terraform plan/apply` with them):

- `GOOGLE_APPLICATION_CREDENTIALS`: path to a service-account JSON with DNS/Run/Compute access.
- `TF_VAR_project_id`: GCP project hosting telemetry.
- `TF_VAR_region`: region for Cloud Run + Compute (defaults to `us-central1`).
- `TF_VAR_telemetry_collector_endpoint` (for example `https://api.runmat.com/v1/t`)
- `TF_VAR_telemetry_collector_key` (optional key forwarded by compatibility ingress)
- `TF_VAR_telemetry_ingestion_key`
- `TF_VAR_worker_image`: container image URI for the Cloud Run worker (e.g. `us-docker.pkg.dev/<project>/telemetry/worker:latest`).
- `TF_VAR_udp_forwarder_image`: container image URI for the UDP forwarder.

### Building/publishing containers

Both the HTTP worker and the UDP forwarder live under `infra/worker/` and `infra/udp-forwarder/`. Build and push them to Artifact Registry (or another registry) whenever you change the code:

```bash
# one-time bootstrap (builds images, provisions SA/bucket, writes infra/.env)
./infra/scripts/bootstrap_gcp.sh

# or run manually if you prefer:
docker build -t us-docker.pkg.dev/$PROJECT/telemetry/worker:$TAG infra/worker
docker push us-docker.pkg.dev/$PROJECT/telemetry/worker:$TAG

docker build -t us-docker.pkg.dev/$PROJECT/telemetry/udp-forwarder:$TAG infra/udp-forwarder
docker push us-docker.pkg.dev/$PROJECT/telemetry/udp-forwarder:$TAG
```

Supply the same image URIs to Terraform via `TF_VAR_worker_image` / `TF_VAR_udp_forwarder_image`.

The compatibility worker intentionally does minimal legacy mapping for older runtime payloads: it preserves only event occurrence (`runtime_started`/`runtime_finished` -> canonical names), `arch`, and stable user id (`cid` -> `distinctId`). Other legacy envelope fields are not preserved.

> The GitHub Actions workflow automatically builds/pushes both images for every commit and injects the resulting tags into Terraform (`worker:${{ github.sha }}` and `udp-forwarder:${{ github.sha }}`). The local bootstrap script mirrors that flow so `terraform plan` works outside CI.

> **Release builds:** set `RUNMAT_TELEMETRY_KEY=$TELEMETRY_INGESTION_KEY` before invoking `cargo build` (the release workflow does this automatically) so the CLI bakes the header it must send to the ingestion service.

### GitHub Actions deployment

`.github/workflows/terraform.yml` authenticates with GCP and runs plan/apply when `infra/` changes:

- **Plan job** executes on same-repo PRs and publishes the plan artifact.
- **Apply job** runs on pushes to `main`.

Repository secrets required by the workflow:

| Secret | Purpose |
| --- | --- |
| `GCP_CREDENTIALS` | Base64/JSON for the Terraform service account |
| `GCP_PROJECT_ID` | Passed through to `TF_VAR_project_id` |
| `GCP_REGION` | Default region for Cloud Run/Compute |
| `GCS_TF_STATE_BUCKET` | Bucket that stores Terraform state |
| `TELEMETRY_COLLECTOR_ENDPOINT` | Compatibility ingress forward target (usually `https://api.runmat.com/v1/t`) |
| `TELEMETRY_COLLECTOR_KEY` | Optional key included when forwarding to collector |
| `TELEMETRY_INGESTION_KEY` | Required shared secret (`x-telemetry-key`); also baked into release builds |

Delegate `telemetry.runmat.org` and `telemetry.runmat.com` to the Cloud DNS name servers emitted by `terraform output telemetry_name_servers`. Once delegated, Cloud Run’s domain mapping automatically issues TLS for both HTTPS endpoints (e.g. `https://telemetry.runmat.com/ingest`), and the UDP forwarding rule is reachable at both `udp.telemetry.runmat.org:7846` and `udp.telemetry.runmat.com:7846`.
