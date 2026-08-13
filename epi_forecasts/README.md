# Google Research EpiHub

This repository serves as a public archive for prospective epidemiological
forecasts generated as part of ongoing research. The forecasts cover three
public health targets: US jurisdiction-level quantile forecasts for weekly
influenza, COVID-19 and RSV hospitalizations.

## Repository Structure

Forecasts are organized by pathogen and season. The general structure
is as follows:

```
forecasts/
│
└── [target_hub]/
    │
    └── [season_identifier]/
        │
        ├── YYYY-MM-DD-[model_name].csv
        └── ...
```

-   **`[target_hub]`**: The name of the forecasting challenge (e.g., `flu-hub`, `covid19-hub`).
-   **`[season_identifier]`**: The season the forecast applies to (e.g., `2025-2026`).
-   **`YYYY-MM-DD-[model_name].csv`**: The forecast file, named with its reference date and an internal model identifier.

## Data Format

All forecasts are provided in a standardized CSV format, consistent with the
requirements of public health forecasting hubs.
