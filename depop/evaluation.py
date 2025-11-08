from __future__ import annotations

import pandas as pd
from IPython.display import HTML
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from .settings import NotebookSettings


class EvaluationSuite:
    """Policy evaluation, threshold sweep, gallery, and legacy review helpers."""

    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    def threshold_sweep(self, df: pd.DataFrame) -> tuple[float, float, float]:
        best_score = -1.0
        best_review = self.settings.policy.review_threshold
        best_demote = self.settings.policy.demote_threshold
        for review in self.settings.policy.review_grid:
            for demote in self.settings.policy.demote_grid:
                if demote <= review:
                    continue
                y_pred = []
                for row in df.itertuples(index=False):
                    if bool(row.is_spam_pred) and row.confidence_pred >= demote:
                        y_pred.append(1)
                    elif bool(row.is_spam_pred) and row.confidence_pred >= review:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
                score = precision_recall_fscore_support(df["label"], y_pred, average="macro", zero_division=0)[2]
                if score > best_score:
                    best_score = score
                    best_review = review
                    best_demote = demote
        return best_review, best_demote, best_score

    def evaluate(self, df: pd.DataFrame, review_threshold: float, demote_threshold: float) -> dict[str, object]:
        y_true = df["label"].astype(int).tolist()
        y_pred = df["is_spam_pred"].astype(int).tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)

        decisions = []
        for row in df.itertuples(index=False):
            if bool(row.is_spam_pred) and row.confidence_pred >= demote_threshold:
                decisions.append("demote")
            elif bool(row.is_spam_pred) and row.confidence_pred >= review_threshold:
                decisions.append("review")
            else:
                decisions.append("keep")
        df["decision"] = decisions

        return {
            "accuracy": accuracy,
            "macro_f1": f1,
            "macro_precision": precision,
            "macro_recall": recall,
            "classification_report": classification_report(y_true, y_pred, digits=3),
        }

    def build_gallery(self, predictions: pd.DataFrame, source_df: pd.DataFrame, max_rows: int = 4) -> str:
        categories = {
            "True Positive": lambda r: r.label == 1 and r.is_spam_pred,
            "True Negative": lambda r: r.label == 0 and not r.is_spam_pred,
            "False Positive": lambda r: r.label == 0 and r.is_spam_pred,
            "False Negative": lambda r: r.label == 1 and not r.is_spam_pred,
        }
        rows = []
        missing = 0
        for name, predicate in categories.items():
            subset = [r for r in predictions.itertuples(index=False) if predicate(r)]
            for row in subset[:max_rows]:
                matches = source_df.loc[source_df.product_id == row.product_id]
                if matches.empty:
                    missing += 1
                    continue
                original = matches.iloc[0]
                rows.append(
                    {
                        "Category": name,
                        "Product ID": row.product_id,
                        "True Label": row.label,
                        "Predicted": row.is_spam_pred,
                        "Confidence": f"{row.confidence_pred:.2f}",
                        "Reason": row.reason_pred,
                        "Description": original.get("description", "")[:200],
                    }
                )
        if not rows:
            return "<p>No examples available.</p>"
        html = pd.DataFrame(rows).to_html(index=False)
        if missing:
            html = f"<p>Skipped {missing} rows without source data.</p>" + html
        return html

    def show_legacy_failures(self, df: pd.DataFrame, max_rows: int = 9) -> HTML:
        rows = []
        for row in df.itertuples(index=False):
            reason = self._legacy_reason(row.description)
            if reason:
                rows.append(
                    {
                        "product_id": row.product_id,
                        "label": int(row.label),
                        "matched_reason": reason,
                        "description": str(row.description)[:180],
                    }
                )
            if len(rows) >= max_rows:
                break
        return HTML(pd.DataFrame(rows).to_html(index=False)) if rows else HTML("<p>No legacy failures sampled.</p>")

    def _legacy_reason(self, text: str) -> str | None:
        if not text:
            return None
        tokens = str(text).split()
        hashtag_ratio = sum(1 for t in tokens if t.startswith("#")) / len(tokens) if tokens else 0.0
        if hashtag_ratio > 0.2:
            return "hashtag-heavy description"
        lower = str(text).lower()
        cta_terms = ("dm", "whatsapp", "contact", "inbox", "email", "text me")
        if any(term in lower for term in cta_terms):
            return "call-to-action present"
        brand_terms = ("nike", "adidas", "gucci")
        if any(term in lower for term in brand_terms):
            return "brand keyword mentioned"
        return None
