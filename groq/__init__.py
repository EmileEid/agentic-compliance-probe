"""Local development stub for the `groq` SDK.

This stub exists purely to satisfy editor/static import resolution during
development. It raises clear runtime errors if used; install the real
`groq` package for production runs.
"""

class Groq:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.chat = self.Chat()

    class Chat:
        class Completions:
            def create(self, *args, **kwargs):
                raise RuntimeError(
                    "Groq SDK not installed. Install with `pip install groq` or run with --dry-run."
                )

        completions = Completions()
