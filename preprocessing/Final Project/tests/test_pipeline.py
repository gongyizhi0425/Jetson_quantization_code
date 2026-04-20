from __future__ import annotations

import unittest

from ehr_retention.baselines import build_context
from ehr_retention.metrics import evidence_recall
from ehr_retention.qa_generation import generate_qa
from ehr_retention.retention import score_event
from ehr_retention.synthetic import generate_events
from ehr_retention.timeline import build_timelines, validate_timeline


def _small_dataset():
    events = list(generate_events(patients=3, seed=7, noise_level="high", timeline_length="long"))
    timelines = build_timelines(events)
    samples = generate_qa(timelines)
    return events, timelines, samples


class PipelineTests(unittest.TestCase):
    def test_synthetic_generator_is_deterministic(self):
        first = list(generate_events(patients=2, seed=123))
        second = list(generate_events(patients=2, seed=123))
        self.assertEqual(first, second)

    def test_timelines_are_sorted_and_have_event_ids(self):
        _, timelines, _ = _small_dataset()
        for timeline in timelines:
            validate_timeline(timeline)

    def test_qa_samples_have_answer_and_evidence(self):
        _, _, samples = _small_dataset()
        self.assertTrue(samples)
        self.assertTrue(all(sample["answer"] for sample in samples))
        self.assertTrue(all(sample["gold_evidence_event_ids"] for sample in samples))

    def test_high_value_events_score_above_noise(self):
        important = {
            "event_id": "A",
            "date": "2020-01-01",
            "event_type": "hospitalization",
            "text": "Hospitalized for severe hypoglycemia.",
        }
        noise = {
            "event_id": "B",
            "date": "2020-01-02",
            "event_type": "normal_lab",
            "text": "normal lab panel reviewed",
        }
        self.assertGreater(
            score_event(important, "Why hospitalized?", 0, 2),
            score_event(noise, "Why hospitalized?", 1, 2),
        )

    def test_selective_keeps_more_evidence_than_sliding_window_with_same_event_budget(self):
        _, timelines, samples = _small_dataset()
        timeline_by_patient = {timeline["patient_id"]: timeline for timeline in timelines}
        sample = next(s for s in samples if s["task_type"] == "diagnosis_history")
        timeline = timeline_by_patient[sample["patient_id"]]

        _, sliding_events = build_context(
            timeline,
            sample["question"],
            strategy="sliding_window",
            top_k_events=5,
        )
        _, selective_events = build_context(
            timeline,
            sample["question"],
            strategy="selective",
            top_k_events=5,
        )
        gold = list(sample["gold_evidence_event_ids"])
        sliding_ids = [event["event_id"] for event in sliding_events]
        selective_ids = [event["event_id"] for event in selective_events]

        self.assertGreaterEqual(evidence_recall(selective_ids, gold), evidence_recall(sliding_ids, gold))
        self.assertEqual(evidence_recall(selective_ids, gold), 1.0)


if __name__ == "__main__":
    unittest.main()
