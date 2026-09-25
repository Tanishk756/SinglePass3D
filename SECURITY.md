# Security policy

## Supported version

Security fixes target the latest tagged release.

## Reporting

Use GitHub private vulnerability reporting when enabled. Do not include secrets, private drone footage, exact sensitive coordinates, or restricted mission data in a public issue.

SinglePass3D runs local tools and model code with the current user's permissions. Review third-party model licenses and checkpoints before operational deployment. The local web application binds to loopback by default and is not designed for exposure to an untrusted network.

