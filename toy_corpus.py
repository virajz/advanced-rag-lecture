DOCS = [
  {
    "id": "ssl_formats",
    "text": "SSL certificates are commonly encoded in PEM or DER formats. PEM files often contain a certificate chain."
  },
  {
    "id": "dev_setup_ssl_mention",
    "text": "For local development, you can run the app on http://localhost:3000. Some teams mention SSL in dev, but it is not required."
  },
  {
    "id": "rotation_policy",
    "text": "Certificate rotation policy: rotate every 90 days. Automate renewal and monitor expiry dates."
  },
  {
    "id": "ssl_cert_location",
    "text": "Production SSL Configuration - Step 1 of 4: Place your SSL certificate files (fullchain.pem and privkey.pem) in the /etc/myapp/tls/ directory."
  },
  {
    "id": "container_port_note",
    "text": "For containerized deployments behind a reverse proxy, expose 8443 internally and terminate TLS at the proxy. Do not bind 443 in the container."
  },
  {
    "id": "tls_env_vars",
    "text": "Production SSL Configuration - Step 2 of 4: Set environment variables TLS_CERT_PATH and TLS_KEY_PATH to point to your certificate files."
  },
  {
    "id": "compliance_req",
    "text": "Compliance requires TLS 1.2+ and disallows weak ciphers. All production endpoints must support modern cipher suites."
  },
  {
    "id": "https_port_config",
    "text": "Production SSL Configuration - Step 3 of 4: Configure HTTPS_PORT=443 in your environment for standard HTTPS traffic."
  },
  {
    "id": "infra_constraint",
    "text": "Infrastructure constraint: only the reverse proxy can bind privileged ports (e.g., 443). Application containers must use high ports."
  },
  {
    "id": "service_restart_note",
    "text": "Production SSL Configuration - Step 4 of 4: Restart the service to apply all SSL configuration changes."
  },
]
