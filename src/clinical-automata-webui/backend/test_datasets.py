"""Guards that the display manifest stays honest.

1. datasets.yaml keys must match the DATA_DIR keys in the backend harness's
   prepare.py — otherwise the showcase and the whitelist drift apart.
2. No real filesystem paths may survive sanitization.
"""
import os
import re

os.environ.setdefault("CAWEBUI_access_code", "demo-access")
# Validate the manifest in full (internal datasets included); the visibility
# guard at the bottom then checks that PUBLIC deployments really do hide them.
os.environ.setdefault("CAWEBUI_visibility", "internal")

from app import datasets  # noqa: E402
from app.claude_runner import _confidentiality_prompt  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.events import sanitize  # noqa: E402


def set_visibility(v: str) -> None:
    os.environ["CAWEBUI_visibility"] = v
    get_settings.cache_clear()


def prepare_py_keys() -> set[str]:
    path = os.path.join(get_settings().backend_dir, "prepare.py")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    # isolate the DATA_DIR = { ... } block, then pull quoted keys before ':'
    m = re.search(r"DATA_DIR\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert m, "could not find DATA_DIR in prepare.py"
    return set(re.findall(r'"([^"]+)"\s*:', m.group(1)))


def main():
    manifest_keys = set(datasets.keys())
    backend_keys = prepare_py_keys()
    assert manifest_keys == backend_keys, (
        f"datasets.yaml keys drifted from prepare.py\n"
        f"  only in manifest: {manifest_keys - backend_keys}\n"
        f"  only in prepare : {backend_keys - manifest_keys}"
    )
    print(f"[ok] {len(manifest_keys)} dataset keys match prepare.py")

    # sanitization must scrub the real dataset root + home + api keys
    data_root = get_settings().data_root.rstrip("/")
    samples = [
        f"{data_root}/ISIC2019/train",
        os.path.expanduser("~/.claude/settings.json"),
        os.path.expanduser("~/projects/run1/train.py"),
        "using key sk-ant-abcdef0123456789ABCDEF",
    ]
    for s in samples:
        out = sanitize(s)
        assert data_root not in out, out
        assert os.path.expanduser("~") not in out, out
        assert "sk-ant-abcdef" not in out, out
    print("[ok] sanitization scrubs dataset paths, home, and api keys")

    # cards never expose a 'path' field — nor the raw i18n block
    for card in datasets.cards():
        assert "path" not in card
        assert "i18n" not in card
    print("[ok] dataset cards carry no real paths")

    # every dataset must be fully translated into every offered language, and
    # the clinical fields must actually differ from English (catches a card
    # that was added without translations and silently fell back).
    en = {c["key"]: c for c in datasets.cards("en")}
    for lang in ("de", "ja", "zh"):
        for card in datasets.cards(lang):
            k = card["key"]
            for field in ("display_name", "modality", "annotations", "label_categories", "description"):
                if field not in en[k]:
                    continue  # optional field, absent in English too
                assert card[field] != en[k][field], f"{k}: {field} not translated into {lang}"
                assert card[field], f"{k}: {field} empty in {lang}"
            assert len(card["label_categories"]) == len(en[k]["label_categories"]), (
                f"{k}: {lang} label_categories count differs from English"
            )
            assert len(card["annotations"]) == len(en[k]["annotations"]), (
                f"{k}: {lang} annotations count differs from English"
            )
            # proper nouns must survive untranslated
            assert card["dataset"] == en[k]["dataset"], f"{k}: dataset name must not be translated"
            assert card.get("license") == en[k].get("license"), f"{k}: license must not be translated"
    print(f"[ok] all {len(en)} datasets fully translated into de/ja/zh; dataset name + license kept verbatim")

    # an unknown language degrades to English rather than blowing up
    assert datasets.cards("fr") == datasets.cards("en")
    print("[ok] unknown language falls back to English")

    # --- visibility: in-house patient data must not leak to external users ---
    # Spelled out rather than counted, so adding a card can never silently widen
    # what a public deployment shows.
    EXPECTED_PUBLIC = [
        "dermoscopy",
        "wrist x-ray",
        "chest x-ray (pneumothorax)",
        "chest x-ray (14 diseases)",
        "chest x-ray (anatomy segmentation)",
    ]
    EXPECTED_WITHHELD = [
        "pelvic ct (fracture)",     # challenge data, CC BY-NC
    ]
    set_visibility("public")  # the DEFAULT
    public = [c["key"] for c in datasets.cards()]
    for k in EXPECTED_WITHHELD:
        assert k not in public, f"withheld dataset '{k}' leaked into the public cards!"
    assert public == EXPECTED_PUBLIC, f"public cards changed: {public}"
    assert sorted(datasets.withheld_keys()) == sorted(EXPECTED_WITHHELD)
    # ...and the agent is explicitly told to withhold them (via --append-system-prompt,
    # so the harness itself is never modified)
    prompt = _confidentiality_prompt(datasets.withheld_keys())
    for k in EXPECTED_WITHHELD:
        assert k in prompt
    assert "not use" in prompt and "not list" in prompt
    print(f"[ok] public mode: {len(EXPECTED_WITHHELD)} datasets hidden from cards AND withheld from the agent")

    set_visibility("internal")
    for k in EXPECTED_WITHHELD:
        assert k in [c["key"] for c in datasets.cards()]
    assert datasets.withheld_keys() == []
    print("[ok] internal mode: every dataset visible, nothing withheld")
    print("\nDATASET GUARDS PASSED")


if __name__ == "__main__":
    main()
