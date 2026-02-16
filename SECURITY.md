# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x (latest) | :white_check_mark: |

## Reporting a Vulnerability

EdgeNN is a bare-metal inference library with no network stack, but buffer overflow or
integer overflow vulnerabilities in operator code could be relevant for safety-critical deployments.

To report a vulnerability:
1. **Do NOT open a public issue**
2. Email: dimitrioskafetzisd@gmail.com with subject "EdgeNN Security"
3. Include: version, affected code, reproduction steps, potential impact
4. Expected response time: 48 hours

## Scope

Security-relevant issues include:
- Buffer overflows in operator implementations
- Integer overflow in quantization/accumulation code
- Arena allocator boundary violations
- Undefined behavior detectable by sanitizers
