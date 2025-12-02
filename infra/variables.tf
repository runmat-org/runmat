variable "cloudflare_account_id" {
  description = "Cloudflare account id that owns runmat.org"
  type        = string
}

variable "telemetry_domain" {
  description = "Delegated subdomain that Cloudflare will host (e.g. telemetry.runmat.org)"
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