from backend.main import app

# This file is intentionally not used as a Vercel route.
# It exists as a lightweight import smoke-test target for local tooling.
assert app is not None
