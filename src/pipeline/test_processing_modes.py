#!/usr/bin/env python3
"""
Test Processing Modes

Demonstrates all three processing modes: OPENAI, LOCAL, and HYBRID.

Usage:
    python test_processing_modes.py --mode openai
    python test_processing_modes.py --mode local
    python test_processing_modes.py --mode hybrid
    python test_processing_modes.py --all
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PipelineConfig, ProcessingMode, init_config


# Test texts
TEST_TEXTS = [
    {
        "text": "Contact John Smith at john.smith@email.com or call +1 555-123-4567. He works at Microsoft in Seattle.",
        "language": "en",
        "description": "English text with person, email, phone, organization, location"
    },
    {
        "text": "Beste Jan van der Berg, uw IBAN NL91 ABNA 0417 1643 00 is geregistreerd. Uw BSN is 123456789.",
        "language": "nl",
        "description": "Dutch text with person, IBAN, BSN"
    },
    {
        "text": """Dear Dr. Sarah Johnson,

Thank you for your email dated January 15, 2024. I've reviewed the patient records
you sent regarding case #MED-2024-0542. The patient, Michael Brown (DOB: 03/15/1985,
SSN: 123-45-6789), was admitted to Memorial Hospital on December 28, 2023.

Please contact me at sarah.j@hospital.org or call my office at (555) 987-6543.

Best regards,
Dr. Robert Williams
Chief Medical Officer
123 Medical Center Drive
Boston, MA 02115""",
        "language": "en",
        "description": "Complex medical document with multiple PII types"
    }
]


def test_local_mode():
    """Test LOCAL mode (Presidio/spaCy/regex only)"""
    print("\n" + "=" * 70)
    print("TESTING LOCAL MODE (Presidio/spaCy/Regex)")
    print("=" * 70)

    config = init_config(mode="local")

    from anonymization.unified_processor import UnifiedProcessor
    processor = UnifiedProcessor(config)

    for i, test in enumerate(TEST_TEXTS, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Input: {test['text'][:100]}...")

        result = processor.process(
            test["text"],
            language=test["language"],
            include_summary=False  # No summarization in local mode
        )

        print(f"\nDetected PII ({len(result.pii_entities)} entities):")
        for entity in result.pii_entities:
            print(f"  - {entity.text} ({entity.pii_type.value}, conf={entity.confidence:.2f}, method={entity.detection_method})")

        print(f"\nAnonymized: {result.anonymized_text[:150]}...")


def test_openai_mode():
    """Test OPENAI mode"""
    print("\n" + "=" * 70)
    print("TESTING OPENAI MODE")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Skipping OpenAI mode test.")
        print("   Set the environment variable and try again:")
        print("   export OPENAI_API_KEY='your-api-key'")
        return

    config = init_config(mode="openai")

    from anonymization.unified_processor import UnifiedProcessor
    processor = UnifiedProcessor(config)

    # Test with first example only (to save API costs)
    test = TEST_TEXTS[0]
    print(f"\n--- Test: {test['description']} ---")
    print(f"Input: {test['text']}")

    result = processor.process(
        test["text"],
        language=test["language"],
        include_summary=True
    )

    print(f"\nDetected PII ({len(result.pii_entities)} entities):")
    for entity in result.pii_entities:
        print(f"  - {entity.text} ({entity.pii_type.value}, conf={entity.confidence:.2f})")

    print(f"\nAnonymized: {result.anonymized_text}")

    if result.summary:
        print(f"\nSummary: {result.summary}")
        print(f"Key entities: {result.key_entities}")
        print(f"Key topics: {result.key_topics}")


def test_hybrid_mode():
    """Test HYBRID mode"""
    print("\n" + "=" * 70)
    print("TESTING HYBRID MODE (Local + OpenAI verification)")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Skipping Hybrid mode test.")
        print("   Set the environment variable and try again:")
        print("   export OPENAI_API_KEY='your-api-key'")
        return

    config = init_config(mode="hybrid")

    from anonymization.unified_processor import UnifiedProcessor
    processor = UnifiedProcessor(config)

    # Test with complex example
    test = TEST_TEXTS[2]
    print(f"\n--- Test: {test['description']} ---")
    print(f"Input: {test['text'][:200]}...")

    result = processor.process(
        test["text"],
        language=test["language"],
        include_summary=True
    )

    print(f"\nDetected PII ({len(result.pii_entities)} entities):")
    for entity in result.pii_entities:
        print(f"  - {entity.text} ({entity.pii_type.value}, conf={entity.confidence:.2f}, method={entity.detection_method})")

    print(f"\nAnonymized:\n{result.anonymized_text}")

    if result.summary:
        print(f"\nSummary: {result.summary}")

    if result.metadata:
        print(f"\nMetadata: {result.metadata}")


def main():
    parser = argparse.ArgumentParser(
        description="Test processing modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_processing_modes.py --mode local
  python test_processing_modes.py --mode openai
  python test_processing_modes.py --mode hybrid
  python test_processing_modes.py --all

Environment:
  OPENAI_API_KEY - Required for openai and hybrid modes
  PIPELINE_MODE - Default mode (openai, local, hybrid)
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["openai", "local", "hybrid"],
        help="Processing mode to test"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Test all modes"
    )

    args = parser.parse_args()

    if args.all:
        test_local_mode()
        test_openai_mode()
        test_hybrid_mode()
    elif args.mode == "local":
        test_local_mode()
    elif args.mode == "openai":
        test_openai_mode()
    elif args.mode == "hybrid":
        test_hybrid_mode()
    else:
        # Default: test local (doesn't require API key)
        print("No mode specified. Testing LOCAL mode (no API key required).")
        print("Use --mode openai|local|hybrid or --all to test other modes.")
        test_local_mode()


if __name__ == "__main__":
    main()
