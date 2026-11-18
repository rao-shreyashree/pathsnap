from sklearn.ensemble import HistGradientBoostingClassifier


def build_model() -> HistGradientBoostingClassifier:
    """Create the untuned PathSnap boosting baseline."""
    return HistGradientBoostingClassifier(
        early_stopping=False,
        random_state=42,
    )