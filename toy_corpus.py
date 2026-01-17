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
    "id": "prod_ssl_steps",
    "text": "Production SSL setup steps: 1) Put fullchain.pem and privkey.pem in /etc/myapp/tls/. 2) Set TLS_CERT_PATH and TLS_KEY_PATH. 3) Set HTTPS_PORT=443. 4) Restart service."
  },
  {
    "id": "container_port_note",
    "text": "For containerized deployments behind a reverse proxy, expose 8443 internally and terminate TLS at the proxy. Do not bind 443 in the container."
  },
  {
    "id": "compliance_req",
    "text": "Compliance requires TLS 1.2+ and disallows weak ciphers. All production endpoints must support modern cipher suites."
  },
  {
    "id": "infra_constraint",
    "text": "Infrastructure constraint: only the reverse proxy can bind privileged ports (e.g., 443). Application containers must use high ports."
  },
]
