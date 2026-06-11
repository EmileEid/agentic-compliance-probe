"""Local development stub for `google.generativeai`.

Provides a minimal surface so imports resolve in editors. At runtime the
stub raises an informative error directing users to install
`google-generativeai` or to use `--dry-run`.
"""

def configure(api_key: str | None = None) -> None:
    # No-op for static usage; real configuration requires the official SDK.
    return None


class GenerativeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_content(self, prompt: str, generation_config: dict | None = None):
        raise RuntimeError(
            "google-generativeai SDK not installed. Install with `pip install google-generativeai` "
            "or run the pipeline with `--dry-run`."
        )
