
#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
# minimal import from notebook logic: we assume this file is placed where `docs` & idx & functions available,
# so for simplicity this script expects repository root and imports a small helper module `src.vsm_ir` if present.
try:
    from vsm_ir_cli_helpers import rank_with_model  # optional helper if you created one
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["boolean","vsm","bm25"], default="vsm")
    parser.add_argument("--weight", choices=["tfidf","tfidf_sublinear"], default="tfidf")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    # fallback: print instructions
    print("This CLI is a thin wrapper. Use the notebook for full control.")
    print("Query:", args.query)
    print("Model:", args.model, "Weight:", args.weight, "Top-k:", args.k)

if __name__ == '__main__':
    main()
