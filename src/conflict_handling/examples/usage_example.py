"""
Usage Examples for Conflict Handling Module

This file demonstrates how to use the conflict handling components
in the data processing and retrieval pipeline.
"""

from datetime import datetime, timedelta
import sys
sys.path.append("..")

from models import (
    DocumentMetadata,
    EnrichedChunk,
    EntityMention,
    PIIAnnotation,
    PIIType,
    SourceType,
    RetrievalResult,
)
from temporal_extractor import TemporalExpressionExtractor
from metadata_extractor import MetadataExtractor
from conflict_detector import ConflictDetector
from temporal_decay import TemporalDecayScorer, DecayConfig, AdaptiveDecayScorer
from entity_versioning import EntityVersionManager
from pii_evaluation import PIIEvaluator, AnnotatedDocument, GroundTruthLoader


# =============================================================================
# Example 1: Extract Temporal Information from Text
# =============================================================================

def example_temporal_extraction():
    """Demonstrate temporal signal extraction from document content"""
    print("\n" + "="*60)
    print("Example 1: Temporal Expression Extraction")
    print("="*60)

    extractor = TemporalExpressionExtractor()

    # Sample text with various temporal expressions
    text = """
    As of January 15, 2025, the project budget has been revised.
    This document supersedes the previous version dated December 2024.
    The deadline is Q2 2025, and the project started 6 months ago.
    Version 2.3 includes all updates from the last quarterly review.
    """

    signals = extractor.extract(text)

    print(f"\nInput text:\n{text}")
    print(f"\nExtracted signals:")
    print(f"  Explicit dates: {[d.strftime('%Y-%m-%d') for d in signals.explicit_dates]}")
    print(f"  Date expressions: {signals.date_expressions}")
    print(f"  Version indicators: {signals.version_indicators}")
    print(f"  Supersedes references: {signals.supersedes_references}")
    print(f"  Relative references: {signals.relative_references}")
    print(f"  Inferred date: {signals.inferred_date}")
    print(f"  Confidence: {signals.confidence:.2f}")


# =============================================================================
# Example 2: Entity Versioning and Conflict Detection
# =============================================================================

def example_entity_versioning():
    """Demonstrate entity versioning and conflict detection"""
    print("\n" + "="*60)
    print("Example 2: Entity Versioning and Conflict Detection")
    print("="*60)

    # Create entity manager
    manager = EntityVersionManager()

    # Create an entity
    project = manager.get_or_create_entity(
        name="Project Alpha",
        entity_type="PROJECT",
        source_doc_id="email_001"
    )

    # Add attributes from different sources over time
    # First mention: January 2025
    manager.add_attribute(
        entity=project,
        attribute_name="budget",
        value="$50,000",
        timestamp=datetime(2025, 1, 15),
        source_doc_id="email_001",
        confidence=0.9,
        evidence_text="Initial budget allocation of $50,000"
    )

    manager.add_attribute(
        entity=project,
        attribute_name="status",
        value="Planning",
        timestamp=datetime(2025, 1, 15),
        source_doc_id="email_001",
        confidence=0.95
    )

    # Second mention: March 2025 (budget changed!)
    manager.add_attribute(
        entity=project,
        attribute_name="budget",
        value="$75,000",
        timestamp=datetime(2025, 3, 20),
        source_doc_id="email_042",
        confidence=0.95,
        evidence_text="Budget increased to $75,000 after scope expansion"
    )

    manager.add_attribute(
        entity=project,
        attribute_name="status",
        value="In Progress",
        timestamp=datetime(2025, 3, 20),
        source_doc_id="email_042",
        confidence=0.95
    )

    # Detect conflicts
    detector = ConflictDetector()
    conflicts = detector.detect_conflicts(project)

    print(f"\nEntity: {project.name}")
    print(f"Current attributes: {project.current_attributes}")
    print(f"\nAttribute history:")
    for attr_name, history in project.attribute_history.items():
        print(f"  {attr_name}:")
        for h in history:
            print(f"    - {h.value} ({h.timestamp.strftime('%Y-%m-%d')}, source: {h.source_doc_id})")

    print(f"\nConflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"  - {c.attribute_name}: '{c.value_a}' vs '{c.value_b}'")
        print(f"    Resolution: {c.resolved_value} ({c.resolution_strategy.value})")


# =============================================================================
# Example 3: Temporal Decay in Retrieval
# =============================================================================

def example_temporal_decay():
    """Demonstrate temporal decay scoring in retrieval"""
    print("\n" + "="*60)
    print("Example 3: Temporal Decay in Retrieval")
    print("="*60)

    # Create scorer with 1-year half-life
    scorer = TemporalDecayScorer(DecayConfig(
        half_life_days=365,
        min_weight=0.1,
        reference_date=datetime(2025, 6, 1)  # Fixed reference for demo
    ))

    # Create sample retrieval results with different dates
    results = [
        RetrievalResult(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="Recent information about the project",
            base_score=0.85,
            doc_date=datetime(2025, 5, 15)  # 2 weeks ago
        ),
        RetrievalResult(
            chunk_id="chunk_2",
            doc_id="doc_2",
            text="Older information about the project",
            base_score=0.90,  # Higher base score
            doc_date=datetime(2024, 6, 1)   # 1 year ago
        ),
        RetrievalResult(
            chunk_id="chunk_3",
            doc_id="doc_3",
            text="Very old information",
            base_score=0.95,  # Highest base score
            doc_date=datetime(2023, 1, 1)   # 2.5 years ago
        ),
        RetrievalResult(
            chunk_id="chunk_4",
            doc_id="doc_4",
            text="Information with unknown date",
            base_score=0.80,
            doc_date=None  # Unknown date
        ),
    ]

    print("\nBefore temporal decay (sorted by base_score):")
    for r in sorted(results, key=lambda x: x.base_score, reverse=True):
        date_str = r.doc_date.strftime('%Y-%m-%d') if r.doc_date else 'Unknown'
        print(f"  {r.chunk_id}: base={r.base_score:.2f}, date={date_str}")

    # Apply temporal decay
    decayed_results = scorer.apply_decay_to_results(results)

    print("\nAfter temporal decay (sorted by final_score):")
    for r in decayed_results:
        date_str = r.doc_date.strftime('%Y-%m-%d') if r.doc_date else 'Unknown'
        print(f"  {r.chunk_id}: base={r.base_score:.2f}, weight={r.temporal_weight:.2f}, final={r.final_score:.2f}, date={date_str}")

    # Show decay curve
    print("\nDecay curve (days -> weight):")
    curve = scorer.get_decay_curve(max_age_days=730, step_days=180)
    for age, weight in curve:
        print(f"  {age:4d} days: {weight:.2f}")


# =============================================================================
# Example 4: Adaptive Temporal Decay
# =============================================================================

def example_adaptive_decay():
    """Demonstrate query-adaptive temporal decay"""
    print("\n" + "="*60)
    print("Example 4: Adaptive Temporal Decay")
    print("="*60)

    scorer = AdaptiveDecayScorer()

    queries = [
        "What is the current status of Project Alpha?",  # Time-sensitive
        "What was the original budget for Project Alpha?",  # Historical
        "Who is working on Project Alpha?",  # Neutral
    ]

    for query in queries:
        query_type = scorer.classify_query(query)
        print(f"\nQuery: \"{query}\"")
        print(f"  Classification: {query_type}")


# =============================================================================
# Example 5: PII Evaluation
# =============================================================================

def example_pii_evaluation():
    """Demonstrate PII evaluation against ground truth"""
    print("\n" + "="*60)
    print("Example 5: PII Evaluation")
    print("="*60)

    # Create ground truth document
    ground_truth = [
        AnnotatedDocument(
            doc_id="email_001",
            text="Hi John, please contact Sarah Johnson at sarah.johnson@company.com or call +32 123 456 789.",
            language="en",
            annotations=[
                PIIAnnotation(
                    text="John",
                    pii_type=PIIType.PERSON,
                    start_char=3,
                    end_char=7,
                    confidence=1.0,
                    is_sensitive=True
                ),
                PIIAnnotation(
                    text="Sarah Johnson",
                    pii_type=PIIType.PERSON,
                    start_char=24,
                    end_char=37,
                    confidence=1.0,
                    is_sensitive=True
                ),
                PIIAnnotation(
                    text="sarah.johnson@company.com",
                    pii_type=PIIType.EMAIL,
                    start_char=41,
                    end_char=66,
                    confidence=1.0,
                    is_sensitive=True
                ),
                PIIAnnotation(
                    text="+32 123 456 789",
                    pii_type=PIIType.PHONE,
                    start_char=79,
                    end_char=94,
                    confidence=1.0,
                    is_sensitive=True
                ),
            ]
        )
    ]

    # Simulated detector results (as if from Presidio)
    class MockDetector:
        def detect(self, text):
            # Simulating: detected 3 of 4, with 1 false positive
            return [
                PIIAnnotation(
                    text="John",
                    pii_type=PIIType.PERSON,
                    start_char=3,
                    end_char=7,
                    confidence=0.95,
                    is_sensitive=True
                ),
                PIIAnnotation(
                    text="Sarah Johnson",
                    pii_type=PIIType.PERSON,
                    start_char=24,
                    end_char=37,
                    confidence=0.92,
                    is_sensitive=True
                ),
                PIIAnnotation(
                    text="sarah.johnson@company.com",
                    pii_type=PIIType.EMAIL,
                    start_char=41,
                    end_char=66,
                    confidence=0.99,
                    is_sensitive=True
                ),
                # Missed the phone number (False Negative)
                # Added a false positive
                PIIAnnotation(
                    text="company",
                    pii_type=PIIType.ORGANIZATION,
                    start_char=56,
                    end_char=63,
                    confidence=0.6,
                    is_sensitive=True
                ),
            ]

    # Evaluate
    evaluator = PIIEvaluator()
    result = evaluator.evaluate(ground_truth, MockDetector())

    print("\nGround truth: 4 PII items (John, Sarah Johnson, email, phone)")
    print("Detected: 4 items (John, Sarah Johnson, email, 'company' as org)")
    print("\nEvaluation Results:")
    print(f"  True Positives: {result.true_positives}")
    print(f"  False Positives: {result.false_positives}")
    print(f"  False Negatives: {result.false_negatives}")
    print(f"  Precision: {result.precision:.2%}")
    print(f"  Recall: {result.recall:.2%}")
    print(f"  F1 Score: {result.f1_score:.2%}")

    if result.fn_details:
        print(f"\nMissed PII (False Negatives):")
        for fn in result.fn_details:
            print(f"  - '{fn.text}' ({fn.pii_type.value})")


# =============================================================================
# Example 6: Complete Pipeline Integration
# =============================================================================

def example_pipeline_integration():
    """Demonstrate full pipeline with all components"""
    print("\n" + "="*60)
    print("Example 6: Complete Pipeline Integration")
    print("="*60)

    # Initialize components
    entity_manager = EntityVersionManager()
    temporal_extractor = TemporalExpressionExtractor()
    conflict_detector = ConflictDetector()

    # Simulate processing multiple documents
    documents = [
        {
            "doc_id": "email_001",
            "text": "As of January 15, 2025, Project Alpha has a budget of $50,000. John Smith is the project lead.",
            "date": datetime(2025, 1, 15)
        },
        {
            "doc_id": "email_042",
            "text": "Update: Project Alpha budget increased to $75,000 due to scope expansion. Effective March 20, 2025.",
            "date": datetime(2025, 3, 20)
        },
        {
            "doc_id": "email_089",
            "text": "Project Alpha status: In Progress. Team size is 8 people. Manager: John Smith.",
            "date": datetime(2025, 4, 1)
        }
    ]

    print("\nProcessing documents...")

    for doc in documents:
        print(f"\n  Processing: {doc['doc_id']}")

        # Extract temporal signals
        signals = temporal_extractor.extract(doc["text"])
        print(f"    Temporal signals: {len(signals.explicit_dates)} dates found")

        # Create/update entities (simplified - normally from NER)
        project = entity_manager.get_or_create_entity(
            name="Project Alpha",
            entity_type="PROJECT",
            source_doc_id=doc["doc_id"]
        )

        # Extract and add attributes (simplified)
        if "$50,000" in doc["text"]:
            entity_manager.add_attribute(
                entity=project,
                attribute_name="budget",
                value="$50,000",
                timestamp=doc["date"],
                source_doc_id=doc["doc_id"],
                confidence=0.9
            )
        if "$75,000" in doc["text"]:
            entity_manager.add_attribute(
                entity=project,
                attribute_name="budget",
                value="$75,000",
                timestamp=doc["date"],
                source_doc_id=doc["doc_id"],
                confidence=0.95
            )
        if "8 people" in doc["text"]:
            entity_manager.add_attribute(
                entity=project,
                attribute_name="team_size",
                value=8,
                timestamp=doc["date"],
                source_doc_id=doc["doc_id"],
                confidence=0.9
            )

    # Detect conflicts
    conflicts = conflict_detector.detect_conflicts(project)

    print("\n" + "-"*40)
    print("Final Entity State:")
    print("-"*40)
    print(f"Entity: {project.name}")
    print(f"Current attributes: {project.current_attributes}")

    print(f"\nConflicts: {len(conflicts)}")
    print(conflict_detector.get_conflict_summary(conflicts))


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    example_temporal_extraction()
    example_entity_versioning()
    example_temporal_decay()
    example_adaptive_decay()
    example_pii_evaluation()
    example_pipeline_integration()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
