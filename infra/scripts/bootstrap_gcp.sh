#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-runmat}
REGION=${REGION:-us-central1}
STATE_BUCKET=${STATE_BUCKET:-${PROJECT}-terraform-state}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-runmat-terraform}
ARTIFACT_REPO=${ARTIFACT_REPO:-telemetry}
TAG=${TAG:-$(git rev-parse --short HEAD)}
CONFIG_DIR=${CONFIG_DIR:-$HOME/.config/gcloud}
KEY_PATH=${KEY_PATH:-$CONFIG_DIR/${SERVICE_ACCOUNT}.json}

WORKER_IMAGE="us-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/worker:${TAG}"
UDP_IMAGE="us-docker.pkg.dev/${PROJECT}/${ARTIFACT_REPO}/udp-forwarder:${TAG}"

echo "=> Using project ${PROJECT} (${REGION}), tag ${TAG}"
gcloud config set project "${PROJECT}" >/dev/null
gcloud config set run/region "${REGION}" >/dev/null

echo "=> Ensuring service account ${SERVICE_ACCOUNT}"
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT}" --display-name "RunMat Terraform"
fi

for role in roles/run.developer roles/compute.instanceAdmin.v1 roles/compute.networkAdmin roles/dns.admin roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com" \
    --role="${role}" >/dev/null
done

echo "=> Writing key to ${KEY_PATH}"
mkdir -p "$(dirname "${KEY_PATH}")"
gcloud iam service-accounts keys create "${KEY_PATH}" \
  --iam-account="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com" >/dev/null

echo "=> Ensuring state bucket ${STATE_BUCKET}"
gsutil mb -l "${REGION%%-*}" "gs://${STATE_BUCKET}" >/dev/null 2>&1 || true
gcloud storage buckets add-iam-policy-binding "gs://${STATE_BUCKET}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin" >/dev/null

echo "=> Ensuring Artifact Registry ${ARTIFACT_REPO}"
gcloud artifacts repositories describe "${ARTIFACT_REPO}" --location="${REGION%%-*}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${REGION%%-*}" \
    --description="RunMat telemetry images"
gcloud artifacts repositories add-iam-policy-binding "${ARTIFACT_REPO}" \
  --location="${REGION%%-*}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" >/dev/null

echo "=> Authenticating docker"
gcloud auth configure-docker us-docker.pkg.dev -q

echo "=> Building worker image ${WORKER_IMAGE}"
docker build -t "${WORKER_IMAGE}" "$(git rev-parse --show-toplevel)/infra/worker"
docker push "${WORKER_IMAGE}"

echo "=> Building UDP forwarder image ${UDP_IMAGE}"
docker build -t "${UDP_IMAGE}" "$(git rev-parse --show-toplevel)/infra/udp-forwarder"
docker push "${UDP_IMAGE}"

ENV_FILE="$(git rev-parse --show-toplevel)/infra/.env"
echo "=> Writing ${ENV_FILE}"
cat > "${ENV_FILE}" <<EOF
GCP_PROJECT_ID=${PROJECT}
GCP_REGION=${REGION}
GCS_TF_STATE_BUCKET=${STATE_BUCKET}
GOOGLE_APPLICATION_CREDENTIALS=${KEY_PATH}

TF_VAR_project_id=${PROJECT}
TF_VAR_region=${REGION}
TF_VAR_worker_image=${WORKER_IMAGE}
TF_VAR_udp_forwarder_image=${UDP_IMAGE}

# Fill these in before running terraform plan/apply:
POSTHOG_API_KEY=
POSTHOG_HOST=https://us.i.posthog.com
TELEMETRY_INGESTION_KEY=
GA_MEASUREMENT_ID=
GA_API_SECRET=
EOF

echo "Bootstrap complete. Next steps:"
echo "1. Populate telemetry secrets inside infra/.env."
echo "2. Source the file before running terraform (e.g. 'set -a && source infra/.env && set +a')."
echo "3. Upload ${KEY_PATH} contents to the GitHub secret 'GCP_CREDENTIALS' and copy the other values into repo secrets."

