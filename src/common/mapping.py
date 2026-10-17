"""Conservative source-label to SegreSmart class mapping."""
WET = {
    "biodegradable", "organic", "coffee tea bags", "egg shells",
    "food scraps", "kitchen waste", "yard trimmings",
}
DRY = {
    "paper", "plastic", "glass", "metal", "cardboard", "cloth",
    "recyclable", "non-recyclable", "cans all type", "glass containers",
    "paper products", "plastic bottles", "ceramic product", "diapers",
    "plastics bags wrappers", "platics bags wrappers", "stroform product",
    "sanitary napkin",
}
REJECT = {
    "e-waste", "ewaste", "hazardous", "batteries", "paints", "pesticides",
}

def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").split())

def map_label(label: str, enable_reject: bool = True) -> str | None:
    key = normalize_label(label)
    if key in WET: return "wet"
    if key in DRY: return "dry"
    if key in REJECT: return "reject" if enable_reject else None
    return None
