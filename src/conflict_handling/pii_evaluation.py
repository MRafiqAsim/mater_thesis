"""
PII Anonymization Evaluation

Evaluates the accuracy of PII detection and anonymization
by comparing against manually labeled ground truth.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Protocol

from .models import PIIAnnotation, PIIType, PIIEvaluationResult


class PIIDetector(Protocol):
    """Protocol for PII detection systems"""

    def detect(self, text: str) -> List[PIIAnnotation]:
        """Detect PII in text and return annotations"""
        ...


@dataclass
class AnnotatedDocument:
    """Document with ground truth PII annotations"""

    doc_id: str
    text: str
    annotations: List[PIIAnnotation]
    language: str = "en"
    source_file: Optional[str] = None


@dataclass
class DocumentEvaluationResult:
    """Evaluation result for a single document"""

    doc_id: str
    true_positives: List[PIIAnnotation] = field(default_factory=list)
    false_positives: List[PIIAnnotation] = field(default_factory=list)
    false_negatives: List[PIIAnnotation] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


class PIIEvaluator:
    """
    Evaluates PII detection accuracy against ground truth.

    Supports:
    - Document-level and corpus-level metrics
    - Breakdown by PII type
    - Configurable overlap threshold for matching
    """

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        require_type_match: bool = True
    ):
        """
        Initialize the evaluator.

        Args:
            overlap_threshold: Minimum character overlap for matching (0-1)
            require_type_match: Require PII type to match for TP
        """
        self.overlap_threshold = overlap_threshold
        self.require_type_match = require_type_match

    def evaluate(
        self,
        ground_truth: List[AnnotatedDocument],
        detector: PIIDetector
    ) -> PIIEvaluationResult:
        """
        Evaluate a PII detector against ground truth.

        Args:
            ground_truth: List of annotated documents
            detector: PII detection system to evaluate

        Returns:
            PIIEvaluationResult with metrics
        """
        all_tp = []
        all_fp = []
        all_fn = []

        doc_results = []
        metrics_by_type: Dict[str, Dict[str, int]] = {}

        for doc in ground_truth:
            # Run detection
            detected = detector.detect(doc.text)

            # Get ground truth (sensitive items only)
            gt_annotations = [a for a in doc.annotations if a.is_sensitive]

            # Match detected against ground truth
            tp, fp, fn = self._match_annotations(detected, gt_annotations)

            # Aggregate
            all_tp.extend(tp)
            all_fp.extend(fp)
            all_fn.extend(fn)

            # Track by type
            self._update_type_metrics(metrics_by_type, tp, fp, fn)

            # Document-level result
            doc_results.append(DocumentEvaluationResult(
                doc_id=doc.doc_id,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn
            ))

        # Calculate overall metrics
        total_tp = len(all_tp)
        total_fp = len(all_fp)
        total_fn = len(all_fn)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate per-type metrics
        type_metrics = self._calculate_type_metrics(metrics_by_type)

        return PIIEvaluationResult(
            total_ground_truth=total_tp + total_fn,
            total_detected=total_tp + total_fp,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            metrics_by_type=type_metrics,
            tp_details=all_tp,
            fp_details=all_fp,
            fn_details=all_fn
        )

    def evaluate_single_document(
        self,
        doc: AnnotatedDocument,
        detected: List[PIIAnnotation]
    ) -> DocumentEvaluationResult:
        """
        Evaluate a single document.

        Args:
            doc: Annotated document
            detected: Detected PII annotations

        Returns:
            DocumentEvaluationResult
        """
        gt_annotations = [a for a in doc.annotations if a.is_sensitive]
        tp, fp, fn = self._match_annotations(detected, gt_annotations)

        return DocumentEvaluationResult(
            doc_id=doc.doc_id,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )

    def _match_annotations(
        self,
        detected: List[PIIAnnotation],
        ground_truth: List[PIIAnnotation]
    ) -> Tuple[List[PIIAnnotation], List[PIIAnnotation], List[PIIAnnotation]]:
        """
        Match detected annotations against ground truth.

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        tp = []
        fp = []
        fn = []

        matched_gt_indices = set()

        # Match each detected annotation
        for det in detected:
            best_match_idx = None
            best_overlap = 0

            for i, gt in enumerate(ground_truth):
                if i in matched_gt_indices:
                    continue

                # Check type match if required
                if self.require_type_match:
                    if det.pii_type != gt.pii_type:
                        continue

                # Calculate overlap
                overlap = self._calculate_overlap(det, gt)

                if overlap >= self.overlap_threshold and overlap > best_overlap:
                    best_match_idx = i
                    best_overlap = overlap

            if best_match_idx is not None:
                tp.append(det)
                matched_gt_indices.add(best_match_idx)
            else:
                fp.append(det)

        # Unmatched ground truth = false negatives
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt_indices:
                fn.append(gt)

        return tp, fp, fn

    def _calculate_overlap(
        self,
        detected: PIIAnnotation,
        ground_truth: PIIAnnotation
    ) -> float:
        """Calculate character overlap ratio"""
        # Calculate intersection
        overlap_start = max(detected.start_char, ground_truth.start_char)
        overlap_end = min(detected.end_char, ground_truth.end_char)

        if overlap_end <= overlap_start:
            return 0.0

        overlap_len = overlap_end - overlap_start
        gt_len = ground_truth.end_char - ground_truth.start_char

        return overlap_len / gt_len if gt_len > 0 else 0.0

    def _update_type_metrics(
        self,
        metrics: Dict[str, Dict[str, int]],
        tp: List[PIIAnnotation],
        fp: List[PIIAnnotation],
        fn: List[PIIAnnotation]
    ) -> None:
        """Update per-type metrics"""
        for ann in tp:
            type_name = ann.pii_type.value if isinstance(ann.pii_type, PIIType) else str(ann.pii_type)
            if type_name not in metrics:
                metrics[type_name] = {"tp": 0, "fp": 0, "fn": 0}
            metrics[type_name]["tp"] += 1

        for ann in fp:
            type_name = ann.pii_type.value if isinstance(ann.pii_type, PIIType) else str(ann.pii_type)
            if type_name not in metrics:
                metrics[type_name] = {"tp": 0, "fp": 0, "fn": 0}
            metrics[type_name]["fp"] += 1

        for ann in fn:
            type_name = ann.pii_type.value if isinstance(ann.pii_type, PIIType) else str(ann.pii_type)
            if type_name not in metrics:
                metrics[type_name] = {"tp": 0, "fp": 0, "fn": 0}
            metrics[type_name]["fn"] += 1

    def _calculate_type_metrics(
        self,
        raw_metrics: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, F1 per type"""
        result = {}

        for type_name, counts in raw_metrics.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            result[type_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn  # Total ground truth count
            }

        return result


class GroundTruthLoader:
    """Load and manage ground truth annotations"""

    @staticmethod
    def load_from_json(file_path: str) -> List[AnnotatedDocument]:
        """
        Load ground truth from JSON file.

        Expected format:
        {
            "documents": [
                {
                    "doc_id": "...",
                    "text": "...",
                    "language": "en",
                    "annotations": [
                        {
                            "text": "John Smith",
                            "type": "PERSON",
                            "start": 10,
                            "end": 20,
                            "is_sensitive": true
                        }
                    ]
                }
            ]
        }
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        for doc_data in data.get("documents", []):
            annotations = []

            for ann_data in doc_data.get("annotations", []):
                pii_type = PIIType(ann_data["type"]) if ann_data["type"] in [t.value for t in PIIType] else PIIType.PERSON

                annotation = PIIAnnotation(
                    text=ann_data["text"],
                    pii_type=pii_type,
                    start_char=ann_data["start"],
                    end_char=ann_data["end"],
                    confidence=ann_data.get("confidence", 1.0),
                    is_sensitive=ann_data.get("is_sensitive", True)
                )
                annotations.append(annotation)

            doc = AnnotatedDocument(
                doc_id=doc_data["doc_id"],
                text=doc_data["text"],
                annotations=annotations,
                language=doc_data.get("language", "en"),
                source_file=doc_data.get("source_file")
            )
            documents.append(doc)

        return documents

    @staticmethod
    def save_to_json(documents: List[AnnotatedDocument], file_path: str) -> None:
        """Save annotated documents to JSON file"""
        data = {
            "created_at": datetime.now().isoformat(),
            "document_count": len(documents),
            "documents": []
        }

        for doc in documents:
            doc_data = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "language": doc.language,
                "source_file": doc.source_file,
                "annotations": [
                    {
                        "text": ann.text,
                        "type": ann.pii_type.value if isinstance(ann.pii_type, PIIType) else str(ann.pii_type),
                        "start": ann.start_char,
                        "end": ann.end_char,
                        "confidence": ann.confidence,
                        "is_sensitive": ann.is_sensitive
                    }
                    for ann in doc.annotations
                ]
            }
            data["documents"].append(doc_data)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def create_empty_template(
        texts: List[str],
        output_path: str,
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """
        Create an empty ground truth template for manual annotation.

        Args:
            texts: List of texts to annotate
            output_path: Where to save the template
            doc_ids: Optional document IDs (generated if not provided)
        """
        documents = []

        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else f"doc_{i:04d}"

            doc = AnnotatedDocument(
                doc_id=doc_id,
                text=text,
                annotations=[],
                language="en"
            )
            documents.append(doc)

        GroundTruthLoader.save_to_json(documents, output_path)
        print(f"Created template with {len(documents)} documents at {output_path}")
        print("Please annotate the PII in each document and set is_sensitive=true for items to anonymize.")


def evaluate_pii_detector(
    detector: PIIDetector,
    ground_truth_path: str
) -> PIIEvaluationResult:
    """
    Convenience function to evaluate a PII detector.

    Args:
        detector: PII detection system
        ground_truth_path: Path to ground truth JSON file

    Returns:
        PIIEvaluationResult with metrics
    """
    # Load ground truth
    ground_truth = GroundTruthLoader.load_from_json(ground_truth_path)

    # Evaluate
    evaluator = PIIEvaluator()
    result = evaluator.evaluate(ground_truth, detector)

    # Print report
    print(result.to_report())

    return result


# Example ground truth template
EXAMPLE_GROUND_TRUTH = {
    "documents": [
        {
            "doc_id": "email_001",
            "text": "Hi John, please contact Sarah Johnson at sarah.johnson@company.com or call +32 123 456 789. Meeting at Leuven office tomorrow.",
            "language": "en",
            "annotations": [
                {
                    "text": "John",
                    "type": "PERSON",
                    "start": 3,
                    "end": 7,
                    "is_sensitive": True
                },
                {
                    "text": "Sarah Johnson",
                    "type": "PERSON",
                    "start": 24,
                    "end": 37,
                    "is_sensitive": True
                },
                {
                    "text": "sarah.johnson@company.com",
                    "type": "EMAIL",
                    "start": 41,
                    "end": 66,
                    "is_sensitive": True
                },
                {
                    "text": "+32 123 456 789",
                    "type": "PHONE",
                    "start": 79,
                    "end": 94,
                    "is_sensitive": True
                },
                {
                    "text": "Leuven",
                    "type": "LOCATION",
                    "start": 107,
                    "end": 113,
                    "is_sensitive": False
                }
            ]
        }
    ]
}
