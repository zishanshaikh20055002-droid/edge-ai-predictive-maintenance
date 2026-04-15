# Project Timeline and Major Changes

## Phase 1: Core Platform
- Built streaming ingestion and machine telemetry processing
- Implemented predictive outputs (RUL, stage, risk)
- Added persistence and baseline dashboard

## Phase 2: Diagnosis and Policy
- Added fault localization logic and diagnosis APIs
- Added alarm policy and maintenance priority strategy
- Improved machine-level and fleet-level visibility

## Phase 3: Security and Operations
- Integrated JWT auth and RBAC controls
- Added rate limiting
- Added metrics export and Grafana dashboard stack

## Phase 4: Runtime Hardening and Stability Updates
- Validated full Docker Compose bring-up end-to-end
- Improved quantized inference handling in runtime paths
- Tuned defaults for safer and more interpretable replay behavior
- Clarified stage-versus-alarm interpretation in operations

## Phase 5: Dashboard Upgrade
- Added richer diagnosis cards
- Added probability and component score visualizations
- Improved operator-facing context with model source/version display

## Phase 6: Training and Data Tooling Expansion
- Added multisource data builder and fusion logic
- Added imbalance utilities and focused loss strategies
- Added evaluation metrics helpers
- Added dataset verification CLI for pre-training integrity checks

## Phase 7: Next-generation Extension Work
- Added domain adaptation module for cross-domain robustness
- Added PLC integration bridge design with protocol abstraction
- Added integration example for implementation guidance

## What This Means for Viva Discussion
You can confidently explain that the project evolved from a predictive model demo into a full platform covering:
- ingestion
- inference
- diagnosis
- policy
- security
- observability
- operator UX
- expansion pathways for hardware and domain transfer
