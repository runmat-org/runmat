variable "project_id" {
  description = "GCP project that hosts telemetry infra"
  type        = string
}

variable "region" {
  description = "Primary region for Cloud Run + Compute resources"
  type        = string
  default     = "us-central1"
}

variable "telemetry_domain" {
  description = "Delegated subdomain served by Cloud DNS (e.g. telemetry.runmat.org)"
  type        = string
  default     = "telemetry.runmat.org"
}

variable "telemetry_udp_subdomain" {
  description = "Subdomain used for UDP intake (e.g. udp.telemetry.runmat.org)"
  type        = string
  default     = "udp.telemetry.runmat.org"
}

variable "posthog_api_key" {
  description = "PostHog project ingestion key (phc_xxx)"
  type        = string
}

variable "posthog_host" {
  description = "PostHog ingestion host"
  type        = string
  default     = "https://us.i.posthog.com"
}

variable "telemetry_ingestion_key" {
  description = "Optional shared secret clients must send via x-telemetry-key"
  type        = string
  default     = ""
}

variable "ga_measurement_id" {
  description = "Optional GA4 measurement ID for forwarding telemetry"
  type        = string
  default     = ""
}

variable "ga_api_secret" {
  description = "Optional GA4 Measurement API secret"
  type        = string
  default     = ""
}

variable "worker_image" {
  description = "Container image for the HTTP telemetry worker (Cloud Run)"
  type        = string
  default     = ""
}

variable "udp_forwarder_image" {
  description = "Container image for the UDP forwarder"
  type        = string
  default     = ""
}

variable "udp_min_instances" {
  description = "Number of UDP forwarder instances to keep running"
  type        = number
  default     = 1
}

variable "udp_machine_type" {
  description = "Machine type for UDP forwarder instances"
  type        = string
  default     = "e2-micro"
}
